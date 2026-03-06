# Implementation Plan: Realized Volatility Forecasting with TSFMs

**Last updated:** 2026-03-04
**Status:** Pre-implementation

---

## 0. Data Summary

**Source file:** `data/raw/RV_March2024.xlsx`

| Property | Value |
|----------|-------|
| Assets | 30 DJIA stocks (AAPL, AMGN, AMZN, AXP, BA, CAT, CRM, CSCO, CVX, DIS, DOW, GS, HD, HON, IBM, INTC, JNJ, JPM, KO, MCD, MMM, MRK, MSFT, NKE, PG, TRV, UNH, V, VZ, WMT) |
| Observations | 5,346 trading days per full-sample stock |
| Date range | ~2003-01-02 to 2024-12-31 |
| 1-min measures | RV, BPV, Good (RS+), Bad (RS-), RQ |
| 5-min measures | RV_5, BPV_5, Good_5, Bad_5, RQ_5 |
| Missing encoding | Zeros (replace with NaN before analysis) |
| Short-history stocks | DOW (1,266 obs from ~2019), CRM (4,976), TRV (4,302), V (4,035) |

**Not in dataset (must source externally):**
- Daily returns (required for Realized GARCH / Realized EGARCH)
- Risk-free rate (not needed for our models)

---

## 2A. Econometric Baselines

### Tier 1: Must Have

---

#### Model 1: HAR (Corsi, 2009)

**Specification:**

```
RV_t = beta_0 + beta_1 * RV_{t-1} + beta_2 * RV^{(w)}_{t-1} + beta_3 * RV^{(m)}_{t-1} + eps_t
```

where:
- `RV_{t-1}` = previous day's realized variance
- `RV^{(w)}_{t-1} = (1/5) * sum_{i=1}^{5} RV_{t-i}` = weekly average RV
- `RV^{(m)}_{t-1} = (1/22) * sum_{i=1}^{22} RV_{t-i}` = monthly average RV

**Required data columns:** `RV` (1-min or 5-min; we use 5-min as primary, 1-min for robustness)

**Python implementation:**

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

def construct_har_features(rv_series: pd.Series) -> pd.DataFrame:
    """Construct HAR regressors from an RV series.

    Parameters
    ----------
    rv_series : pd.Series
        Daily realized variance, indexed by date. Must be pre-cleaned
        (NaN for missing, no zeros).

    Returns
    -------
    pd.DataFrame with columns: RV_d, RV_w, RV_m, RV_target
        Aligned so row t has RV_target = RV_t and regressors = info at t-1.
    """
    rv = rv_series.copy()
    df = pd.DataFrame(index=rv.index)
    df['RV_target'] = rv
    df['RV_d'] = rv.shift(1)                              # RV_{t-1}
    df['RV_w'] = rv.shift(1).rolling(5).mean()             # (1/5)*sum RV_{t-1}..RV_{t-5}
    df['RV_m'] = rv.shift(1).rolling(22).mean()            # (1/22)*sum RV_{t-1}..RV_{t-22}
    return df.dropna()


def fit_har(rv_series: pd.Series, train_end: str, nw_lags: int = 5):
    """Fit HAR model on training data up to train_end.

    Returns
    -------
    statsmodels OLS results object fitted with HAC (Newey-West) SEs.
    """
    df = construct_har_features(rv_series)
    train = df.loc[:train_end]
    X = sm.add_constant(train[['RV_d', 'RV_w', 'RV_m']])
    y = train['RV_target']
    model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})
    return model


def forecast_har(model, rv_series: pd.Series, forecast_date: str) -> float:
    """Produce one-step-ahead HAR forecast for forecast_date."""
    df = construct_har_features(rv_series)
    row = df.loc[forecast_date]
    X = np.array([1.0, row['RV_d'], row['RV_w'], row['RV_m']])
    return model.predict(X)[0]
```

**Package:** `statsmodels>=0.14.0`

**Estimation details:**
- Window: expanding (re-estimate every day or every N days; every 22 days is common for speed)
- Minimum training window: 500 observations (ensures stable monthly lag estimates)
- Standard errors: Newey-West HAC with 5 lags (for inference tables only; forecasts use point estimates)
- For h>1 forecasts: use direct projection, i.e., regress `RV_{t+h}` (or `(1/h)*sum_{i=1}^{h} RV_{t+i}` for aggregated) on the same HAR regressors

**Data transformations:** None for baseline. Log-HAR variant uses log(RV) as both LHS and RHS.

**Known pitfalls:**
- The weekly and monthly averages must be constructed from *lagged* data only. A common bug is to include the current day's RV in the weekly/monthly averages, which creates look-ahead bias.
- `rolling(5).mean()` after `shift(1)` correctly computes the average of days t-1 through t-5 (when there are no gaps). If there are missing days (NaN), pandas `rolling` with `min_periods` should be set carefully.
- For multi-step-ahead (h=5, h=22): use direct projection HAR as in Corsi (2009) Section 3.2, regressing the h-day-ahead average RV on today's daily/weekly/monthly RV. Do NOT iterate the 1-step model.
- HAR can produce negative forecasts for low-volatility periods. Two options: (a) floor at zero, or (b) use log-HAR as robustness check.

---

#### Model 2: HAR-J (Andersen, Bollerslev & Diebold, 2007)

**Specification:**

```
RV_t = beta_0 + beta_1 * RV_{t-1} + beta_2 * RV^{(w)}_{t-1} + beta_3 * RV^{(m)}_{t-1}
       + beta_4 * J_{t-1} + eps_t
```

where:
- `J_{t-1} = max(RV_{t-1} - BPV_{t-1}, 0)` = truncated jump variation

**Required data columns:** `RV`, `BPV` (both available at 1-min and 5-min)

**Python implementation:**

```python
def construct_harj_features(rv_series: pd.Series,
                            bpv_series: pd.Series) -> pd.DataFrame:
    """Construct HAR-J regressors."""
    rv = rv_series.copy()
    bpv = bpv_series.copy()
    df = construct_har_features(rv)  # inherits RV_d, RV_w, RV_m, RV_target
    jump = np.maximum(rv - bpv, 0)
    df['J_d'] = jump.shift(1)
    return df.dropna()
```

**Package:** `statsmodels>=0.14.0` (same OLS + HAC)

**Estimation details:** Identical to HAR but with one additional regressor. Same expanding window, same NW standard errors.

**Known pitfalls:**
- BPV can occasionally exceed RV due to sampling noise. The `max(..., 0)` truncation handles this, following ABD (2007). Do not drop these observations.
- Some papers use `J_t = (RV_t - BPV_t) * I(z_t > Phi_{1-alpha})` with a formal jump test statistic. The simple truncation is the standard in the HAR-J literature and is what we implement here.
- Weekly and monthly jump averages are sometimes included as additional regressors (HAR-CJ). We keep HAR-J parsimonious with only the daily jump.

---

#### Model 3: HAR-RS (Patton & Sheppard, 2015)

**Specification:**

```
RV_t = beta_0 + beta_1 * RS+_{t-1} + beta_2 * RS-_{t-1}
       + beta_3 * RS+^{(w)}_{t-1} + beta_4 * RS-^{(w)}_{t-1}
       + beta_5 * RS+^{(m)}_{t-1} + beta_6 * RS-^{(m)}_{t-1} + eps_t
```

where:
- `RS+` = realized semivariance from positive returns (Good)
- `RS-` = realized semivariance from negative returns (Bad)
- Weekly/monthly averages computed analogously to HAR

Note: The data exploration confirms `Good + Bad approx RV` (ratio mean ~1.0), validating the decomposition.

**Required data columns:** `Good`, `Bad` (both available; alternatively `Good_5`, `Bad_5`)

**Python implementation:**

```python
def construct_har_rs_features(good_series: pd.Series,
                              bad_series: pd.Series,
                              rv_series: pd.Series) -> pd.DataFrame:
    """Construct HAR-RS (semivariance) regressors."""
    df = pd.DataFrame(index=rv_series.index)
    df['RV_target'] = rv_series

    df['RS_pos_d'] = good_series.shift(1)
    df['RS_neg_d'] = bad_series.shift(1)
    df['RS_pos_w'] = good_series.shift(1).rolling(5).mean()
    df['RS_neg_w'] = bad_series.shift(1).rolling(5).mean()
    df['RS_pos_m'] = good_series.shift(1).rolling(22).mean()
    df['RS_neg_m'] = bad_series.shift(1).rolling(22).mean()
    return df.dropna()
```

**Package:** `statsmodels>=0.14.0`

**Estimation details:** Same as HAR. Six slope coefficients + intercept = 7 parameters.

**Known pitfalls:**
- The key test is whether `beta_2 > beta_1` (bad semivariance predicts more than good), capturing the leverage/asymmetry effect. Report individual coefficients, not just forecast accuracy.
- Ensure the semivariance decomposition is consistent: if `Good + Bad != RV` by more than 1% on average, investigate the data construction. Our exploration shows this holds.
- For h>1 direct projection: regress multi-day average RV on daily/weekly/monthly semivariance lags.

---

#### Model 4: HARQ (Bollerslev, Patton & Quaedvlieg, 2016)

**Specification:**

```
RV_t = beta_0 + (beta_1 + beta_1Q * sqrt(RQ_{t-1})) * RV_{t-1}
       + beta_2 * RV^{(w)}_{t-1} + beta_3 * RV^{(m)}_{t-1} + eps_t
```

Equivalently (in OLS-estimable form):

```
RV_t = beta_0 + beta_1 * RV_{t-1} + beta_1Q * (RV_{t-1} * sqrt(RQ_{t-1}))
       + beta_2 * RV^{(w)}_{t-1} + beta_3 * RV^{(m)}_{t-1} + eps_t
```

where:
- `RQ_{t-1}` = realized quarticity (measures the variance of the variance estimator)
- The interaction term `RV_{t-1} * sqrt(RQ_{t-1})` attenuates the daily coefficient when measurement error is high

**Required data columns:** `RV`, `RQ` (both available at 1-min and 5-min)

**Python implementation:**

```python
def construct_harq_features(rv_series: pd.Series,
                            rq_series: pd.Series) -> pd.DataFrame:
    """Construct HARQ regressors (BPQ 2016)."""
    df = construct_har_features(rv_series)  # inherits RV_d, RV_w, RV_m, RV_target
    rq_sqrt = np.sqrt(rq_series)
    df['RV_d_x_sqrtRQ'] = rv_series.shift(1) * rq_sqrt.shift(1)
    return df.dropna()
```

**Package:** `statsmodels>=0.14.0`

**Estimation details:** Same as HAR. Five slope coefficients + intercept = 6 parameters. NW-HAC standard errors.

**Known pitfalls:**
- RQ values can be very large (heavy-tailed). Check for extreme outliers in `sqrt(RQ)` and consider winsorizing at the 99.9th percentile if numerical issues arise.
- The interaction term can be highly collinear with `RV_d`. Monitor VIFs. BPQ (2016) argue this is not a problem for forecasting, but report it for transparency.
- BPQ (2016) also propose HARQ-F (full version) where all three HAR components have RQ interactions. Start with the parsimonious HARQ and add HARQ-F as a robustness check if time permits.
- The key finding from BPQ is that `beta_1Q < 0`: when measurement error is high, the model discounts the noisy daily RV in favor of smoother weekly/monthly averages. Verify this sign in our results.

---

#### Model 5: Realized GARCH (Hansen, Huang & Shek, 2012)

**Specification (log-linear form):**

Return equation:
```
r_t = mu + sqrt(h_t) * z_t,    z_t ~ N(0,1)
```

GARCH equation:
```
log(h_t) = omega + beta * log(h_{t-1}) + gamma * log(x_{t-1})
```

Measurement equation:
```
log(x_t) = xi + phi * log(h_t) + tau(z_t) + u_t,    u_t ~ N(0, sigma_u^2)
```

where:
- `r_t` = daily return
- `h_t` = conditional variance (latent)
- `x_t` = realized measure (RV)
- `tau(z) = tau_1 * z + tau_2 * (z^2 - 1)` = leverage function

**Required data columns:**
- `RV` (or `RV_5`) for the realized measure x_t
- **Daily returns** -- NOT in the dataset, must source externally

**Sourcing daily returns:**

```python
import yfinance as yf

def download_returns(tickers: list, start: str = '2003-01-01',
                     end: str = '2025-01-01') -> pd.DataFrame:
    """Download adjusted close prices and compute log returns.

    Returns
    -------
    pd.DataFrame of log returns, indexed by date, columns = tickers.
    """
    prices = yf.download(tickers, start=start, end=end,
                         auto_adjust=True, progress=False)['Close']
    log_returns = np.log(prices / prices.shift(1)) * 100  # percentage log returns
    return log_returns.dropna()
```

**Python implementation:**

```python
from arch import arch_model
from arch.univariate import RealizedGARCH

# The `arch` package (version >= 7.0) supports Realized GARCH natively.
# CRITICAL: As of arch 7.1.1 (Feb 2026), the RealizedGARCH class is available
# but the interface requires careful setup.

def fit_realized_garch(returns: pd.Series,
                       rv: pd.Series,
                       train_end: str):
    """Fit a Realized GARCH(1,1) model.

    Parameters
    ----------
    returns : pd.Series
        Daily percentage log returns (r_t * 100).
    rv : pd.Series
        Daily realized variance (not annualized).
    train_end : str
        Last date of training window.

    Returns
    -------
    arch model result object.
    """
    # Align returns and RV on common dates
    common_idx = returns.index.intersection(rv.index)
    r = returns.loc[common_idx].loc[:train_end]
    x = rv.loc[common_idx].loc[:train_end]

    # arch package Realized GARCH
    # Use the log-linear specification (default in HHS 2012)
    model = arch_model(r, mean='Constant', vol='RealizedGARCH',
                       x=x, p=1, q=1)
    result = model.fit(disp='off', options={'maxiter': 500})
    return result
```

**Package:** `arch>=7.0.0` (`pip install arch`)

**Estimation details:**
- Maximum likelihood estimation (quasi-MLE)
- Expanding window; re-estimate every trading day (or every 22 days for speed; the conditional variance path is updated daily regardless)
- Minimum training window: 500 observations
- Optimizer: L-BFGS-B (default in `arch` package); if convergence fails, fall back to Nelder-Mead

**Known pitfalls:**
- **Date alignment is critical.** Returns and RV must be on the same dates. Yahoo Finance may have different holiday calendars than the RV dataset. Use inner join on dates.
- **Return scaling.** The `arch` package expects percentage returns (i.e., multiply log returns by 100). The RV is in decimal (variance of decimal returns). Make sure the units are consistent: if returns are in %, then RV should be in %^2 (multiply RV by 10,000). Alternatively, keep both in decimals. Check the `arch` documentation for the specific `RealizedGARCH` class.
- **Convergence failures.** The log-linear Realized GARCH can fail to converge for some stocks/windows. Implement a try/except block and log failures. Fall back to HAR forecast for that window if Realized GARCH fails.
- **Forecasting.** The one-step-ahead forecast from Realized GARCH is `h_{t+1}` from the GARCH equation. For multi-step-ahead, iterate the GARCH equation forward (no closed-form for h>1).
- **arch package version.** Verify that `arch.univariate.RealizedGARCH` exists in the installed version. If not available, implement the log-linear Realized GARCH manually via MLE using `scipy.optimize.minimize`. This is a fallback only.

**Manual fallback implementation (if arch does not support RealizedGARCH):**

```python
from scipy.optimize import minimize

def realized_garch_loglik(params, returns, rv):
    """Negative log-likelihood for log-linear Realized GARCH(1,1).

    params = [mu, omega, beta, gamma, xi, phi, tau1, tau2, sigma_u2]
    """
    mu, omega, beta, gamma, xi, phi, tau1, tau2, log_sigma_u2 = params
    sigma_u2 = np.exp(log_sigma_u2)
    T = len(returns)
    r = returns.values
    x = np.log(rv.values)  # log realized measure

    log_h = np.zeros(T)
    log_h[0] = omega / (1 - beta)  # unconditional

    nll = 0.0
    for t in range(1, T):
        log_h[t] = omega + beta * log_h[t-1] + gamma * x[t-1]
        h_t = np.exp(log_h[t])
        z_t = (r[t] - mu) / np.sqrt(h_t)

        # Return density: N(mu, h_t)
        nll += 0.5 * (np.log(2*np.pi) + log_h[t] + z_t**2)

        # Measurement density: log(x_t) | h_t ~ N(xi + phi*log(h_t) + tau(z_t), sigma_u2)
        tau_z = tau1 * z_t + tau2 * (z_t**2 - 1)
        mean_x = xi + phi * log_h[t] + tau_z
        nll += 0.5 * (np.log(2*np.pi) + np.log(sigma_u2) + (x[t] - mean_x)**2 / sigma_u2)

    return nll

# Optimize:
# x0 = [0.0, 0.01, 0.6, 0.3, -0.1, 1.0, -0.1, 0.1, np.log(0.5)]
# result = minimize(realized_garch_loglik, x0, args=(returns, rv),
#                   method='L-BFGS-B', options={'maxiter': 1000})
```

---

#### Model 6: ARFIMA (Andersen, Bollerslev, Diebold & Labys, 2003)

**Specification:**

```
(1 - L)^d * (1 - phi_1*L - ... - phi_p*L^p) * (log(RV_t) - mu) =
    (1 + theta_1*L + ... + theta_q*L^q) * eps_t
```

Standard choice: ARFIMA(1,d,1) on log(RV), where `0 < d < 0.5` captures long memory.

**Required data columns:** `RV` (transformed to log(RV))

**Python implementation:**

```python
from statsmodels.tsa.arima.model import ARIMA
# statsmodels does NOT natively support fractional d in ARIMA.
# Use the `arfima` package or implement via approximate MLE.

# Option A: Use the `statsmodels` ARFIMA via `fracDiff` (not directly available)
# Option B: Use the R `forecast` package via rpy2
# Option C: Use approximate fractional differencing + ARMA

# RECOMMENDED: Option C — Approximate approach via Haslett-Raftery algorithm
# or use the `fracdiff` package

# pip install fracdiff  (for fractional differencing)
import fracdiff

def estimate_arfima(log_rv: pd.Series, train_end: str,
                    p: int = 1, q: int = 1):
    """Estimate ARFIMA(p,d,q) on log(RV).

    Uses two-step approach:
    1. Estimate d via GPH (Geweke-Porter-Hudak) semiparametric estimator
    2. Fractionally difference the series
    3. Fit ARMA(p,q) on the fractionally differenced series
    """
    train = log_rv.loc[:train_end].dropna()

    # Step 1: Estimate d via GPH or Whittle
    d_hat = estimate_d_gph(train)
    d_hat = np.clip(d_hat, 0.01, 0.49)  # ensure stationarity

    # Step 2: Fractionally difference
    fd = fracdiff.Fracdiff(d=d_hat)
    fd_series = fd.fit_transform(train.values.reshape(-1, 1)).flatten()

    # Step 3: Fit ARMA(p,q) on differenced series
    from statsmodels.tsa.arima.model import ARIMA
    arma_model = ARIMA(fd_series, order=(p, 0, q))
    arma_result = arma_model.fit()

    return d_hat, arma_result, fd


def estimate_d_gph(series: pd.Series, bandwidth: float = 0.5) -> float:
    """GPH (log-periodogram) estimator of long-memory parameter d.

    bandwidth: fraction of frequencies to use (0.5 = sqrt(T) frequencies).
    """
    x = series.values - series.mean()
    T = len(x)
    m = int(T ** bandwidth)  # number of frequencies

    # Periodogram at Fourier frequencies
    freqs = np.arange(1, m+1) * 2 * np.pi / T
    fft_vals = np.fft.fft(x)
    periodogram = (np.abs(fft_vals[1:m+1])**2) / (2 * np.pi * T)

    # GPH regression: log(I(w_j)) = c - 2d * log(2*sin(w_j/2)) + error
    y = np.log(periodogram)
    X = np.column_stack([np.ones(m), -2 * np.log(2 * np.sin(freqs / 2))])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta[1]  # d estimate


def forecast_arfima(d_hat, arma_result, log_rv_history, h=1):
    """Produce h-step-ahead forecast from ARFIMA model.

    Uses the truncated infinite AR representation for forecasting.
    """
    # Truncated AR(inf) coefficients from ARFIMA
    # pi_j coefficients: pi_j = (j - 1 - d) / j * pi_{j-1}, pi_0 = 1
    max_lag = min(500, len(log_rv_history))
    pi = np.zeros(max_lag)
    pi[0] = 1.0
    for j in range(1, max_lag):
        pi[j] = (j - 1 - d_hat) / j * pi[j-1]

    # AR(inf) forecast
    mu = log_rv_history.mean()
    x = (log_rv_history.values[-max_lag:] - mu)[::-1]
    forecast = mu + np.dot(pi[1:len(x)+1], x[:len(pi)-1])
    return forecast
```

**Package:** `fracdiff>=0.9.0`, `statsmodels>=0.14.0`

**Alternative package:** `arch` has `HARFIMA` in development, or use `rpy2` to call R's `forecast::arfima()`.

**Estimation details:**
- Transform: work with `log(RV_t)` (approximately Gaussian, see ABD 2003)
- Typical values: `d ~ 0.35-0.45` for realized volatility (strong long memory)
- Re-estimate d every 22 days (d is slow-moving); re-estimate ARMA coefficients every day
- Minimum training window: 500 observations (GPH estimator needs sufficient frequencies)
- Forecasts in log space; exponentiate with bias correction: `E[RV_{t+h}] = exp(forecast + 0.5 * sigma_eps^2)`

**Known pitfalls:**
- `fracdiff` package may not handle pandas Series directly; convert to numpy first.
- GPH estimator is noisy for small samples. Local Whittle estimator is more efficient but harder to implement. GPH with bandwidth `m = T^{0.5}` to `T^{0.8}` is standard.
- Multi-step-ahead ARFIMA forecasting requires the truncated AR(infinity) representation. The recursive formula `pi_j = (j-1-d)/j * pi_{j-1}` converges slowly for `d` near 0.5; truncate at 500 lags.
- The bias correction when exponentiating log forecasts is important: `exp(mu_hat + 0.5*sigma2_hat)`. Without it, forecasts are systematically downward biased.
- ARFIMA can be numerically unstable for borderline stationary series (d near 0.5). Clip d to [0.01, 0.49].

---

### Tier 2: Nice to Have

---

#### Model 7: HAR-CJ (Andersen, Bollerslev & Diebold, 2007)

**Specification:**

```
RV_t = beta_0 + beta_C1 * C_{t-1} + beta_C2 * C^{(w)}_{t-1} + beta_C3 * C^{(m)}_{t-1}
       + beta_J1 * J_{t-1} + beta_J2 * J^{(w)}_{t-1} + beta_J3 * J^{(m)}_{t-1} + eps_t
```

where `C_t = BPV_t` (continuous component) and `J_t = max(RV_t - BPV_t, 0)` (jump component), each at daily/weekly/monthly frequency.

**Required data:** `RV`, `BPV`

**Implementation:** Identical to HAR framework but with 6 regressors (3 continuous + 3 jump). `statsmodels` OLS + HAC.

**Priority:** Implement if time permits; HAR-J captures the key jump effect with fewer parameters.

---

#### Model 8: SHAR (Patton & Sheppard, 2015)

**Specification:**

This is a specific parameterization of HAR-RS. In fact, HAR-RS (Model 3 above) IS the SHAR model. The terms are used interchangeably in the literature. Our Model 3 implementation already covers this.

If a separate "SHAR" is desired, it refers to the version where `RV_t` is decomposed as `RS+_t + RS-_t` on the LHS as well, which is algebraically equivalent to HAR-RS for the purpose of forecasting.

**Priority:** Already covered by Model 3 (HAR-RS).

---

#### Model 9: Realized EGARCH (Hansen & Huang, 2016)

**Specification:**

```
r_t = mu + sqrt(h_t) * z_t
log(h_t) = omega + beta * log(h_{t-1}) + tau_1(z_{t-1}) + tau_2(z_{t-1}) * (log(x_{t-1}) - xi - delta * log(h_{t-1}))
```

This extends Realized GARCH by adding an explicit leverage function and allowing the realized measure's innovation to enter the volatility equation.

**Required data:** Returns + RV (same as Realized GARCH)

**Implementation:** `arch` package may support this; otherwise custom MLE. The added complexity relative to Realized GARCH is modest, but convergence is harder.

**Priority:** Implement as robustness check. If Realized GARCH convergence is already problematic, skip.

---

#### Model 10: HEAVY (Shephard & Sheppard, 2010)

**Specification:**

```
RV_t = omega_r + alpha_r * RV_{t-1} + beta_r * mu_{t-1}    (RV equation)
mu_t  = omega_m + alpha_m * r_{t-1}^2 + beta_m * mu_{t-1}   (return variance equation)
```

A bivariate system where RV depends on its own lag and the conditional variance of returns.

**Required data:** Returns + RV

**Implementation:** Custom MLE via `scipy.optimize`. Not available in standard packages.

**Priority:** Low. Only include if Realized GARCH and Realized EGARCH produce interesting results. HEAVY adds interpretive complexity without clearly established superiority over Realized GARCH for point forecasting.

---

#### Model 11: Log-HAR

**Specification:**

```
log(RV_t) = beta_0 + beta_1 * log(RV_{t-1}) + beta_2 * log(RV^{(w)}_{t-1})
            + beta_3 * log(RV^{(m)}_{t-1}) + eps_t
```

All variables in logs. Avoids negative forecasts. Back-transform with bias correction.

**Required data:** `RV` only

**Implementation:** Same as HAR but apply `np.log()` to all variables. Add bias correction `exp(forecast + 0.5*sigma_hat^2)` when converting back to levels.

**Priority:** Include as a robustness check for HAR. Many practitioners prefer log-HAR because (a) log(RV) is closer to Gaussian and (b) it prevents negative forecasts.

---

## 2B. Foundation Model Candidates

### Model 1: Chronos-2 (Amazon, October 2025)

**Architecture:** Encoder-only transformer with group attention. Direct quantile output head. Supports univariate, multivariate, and exogenous covariates.

**HuggingFace model IDs:**
- `amazon/chronos-2-base` (120M params) -- primary
- `amazon/chronos-bolt-small` (9M) -- fast variant
- `amazon/chronos-bolt-base` (48M) -- medium variant

**Installation:**

```bash
pip install chronos-forecasting torch transformers
# Requires: torch >= 2.1, transformers >= 4.35
# GPU memory: ~2GB for chronos-bolt-small, ~4GB for chronos-2-base
# CPU inference possible but slow (~10x slower than GPU for bolt, ~50x for full)
```

**GPU/memory requirements:**
| Variant | Params | GPU VRAM (inference) | GPU VRAM (fine-tuning, batch=8) | CPU feasible? |
|---------|--------|---------------------|--------------------------------|---------------|
| chronos-bolt-small | 9M | ~1 GB | ~2 GB | Yes |
| chronos-bolt-base | 48M | ~2 GB | ~4 GB | Slow |
| chronos-2-base | 120M | ~4 GB | ~8 GB | Very slow |

**Input format:** 1D numpy array or torch tensor. Context window up to 2048 tokens (512 for bolt variants). Each observation = 1 token.

**Zero-shot code skeleton:**

```python
import torch
from chronos import ChronosPipeline

# Load model (downloads from HuggingFace on first run)
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-base",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float32,
)

def chronos_forecast(pipeline, context: np.ndarray,
                     prediction_length: int = 1,
                     num_samples: int = 100) -> dict:
    """Generate probabilistic forecast from Chronos.

    Parameters
    ----------
    context : np.ndarray
        Historical RV values (1D array, most recent last).
        Use up to 512 values for bolt, 2048 for chronos-2.
    prediction_length : int
        Number of steps ahead to forecast (1, 5, or 22).
    num_samples : int
        Number of sample paths for probabilistic forecast.

    Returns
    -------
    dict with keys: 'mean', 'median', 'q10', 'q25', 'q75', 'q90', 'samples'
    """
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

    # Generate forecast samples
    forecast_samples = pipeline.predict(
        context=context_tensor,
        prediction_length=prediction_length,
        num_samples=num_samples,
    )
    # forecast_samples shape: (1, num_samples, prediction_length)
    samples = forecast_samples[0].numpy()  # (num_samples, prediction_length)

    return {
        'mean': samples.mean(axis=0),
        'median': np.median(samples, axis=0),
        'q10': np.percentile(samples, 10, axis=0),
        'q25': np.percentile(samples, 25, axis=0),
        'q75': np.percentile(samples, 75, axis=0),
        'q90': np.percentile(samples, 90, axis=0),
        'samples': samples,
    }
```

**Point forecast extraction:** Use the **median** of the sample distribution as the point forecast. The mean is sensitive to outlier samples. For QLIKE loss comparison, the median is also preferred.

**Fine-tuning procedure:**

```python
from chronos import ChronosPipeline
from chronos.training import ChronosTrainer, TrainingConfig

# Fine-tuning is supported via the chronos-forecasting library
# Uses standard HuggingFace Trainer under the hood

config = TrainingConfig(
    output_dir="./chronos_finetuned",
    learning_rate=1e-5,         # small LR for fine-tuning
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_ratio=0.1,
    weight_decay=0.01,
    save_strategy="epoch",
    seed=42,
)

# Prepare training data: list of 1D arrays (one per asset or time series)
train_series = [rv_asset.values for rv_asset in rv_train_dict.values()]

# trainer = ChronosTrainer(
#     model_name="amazon/chronos-bolt-base",
#     config=config,
#     train_series=train_series,
# )
# trainer.train()
# trainer.save_model("./chronos_finetuned_rv")

# NOTE: Exact API may differ. Check chronos-forecasting docs at:
# https://github.com/amazon-science/chronos-forecasting
# The key principle: fine-tune on ALL assets' training data jointly,
# then evaluate on each asset's test period individually.
```

**Known issues with financial data:**
- Chronos was primarily trained on non-financial time series (energy, retail, web traffic). Financial volatility has much heavier tails and stronger long-memory than typical training data.
- RV has a natural lower bound of zero but Chronos can generate negative samples. Post-process by clipping: `samples = np.maximum(samples, 0)`.
- For h=5 and h=22, Chronos can generate multi-step paths directly (set `prediction_length=5` or `22`). Compare the direct multi-step output vs. using only the first step.
- Context length matters: longer context generally helps for long-memory processes. Use the maximum available context (up to 512 for bolt, 2048 for chronos-2).

---

### Model 2: TimesFM 2.5 (Google, late 2025)

**Architecture:** Decoder-only transformer with patched input (patch size = 32 for daily data). Autoregressive generation. Outputs mean + 10 quantiles (5th, 10th, 15th, 20th, 25th, 50th, 75th, 85th, 90th, 95th).

**HuggingFace model ID:** `google/timesfm-2.0-200m-pytorch`
(Note: v2.5 is a checkpoint update of v2.0; same model ID, different checkpoint. Check for `timesfm-2.5` variant.)

**Installation:**

```bash
pip install timesfm
# Alternatively: pip install timesfm[torch]
# Requires: torch >= 2.1, jax (optional, for JAX backend)
# GPU memory: ~2-3 GB for inference, ~6 GB for fine-tuning
```

**GPU/memory requirements:**
| Setting | VRAM | Notes |
|---------|------|-------|
| Inference (batch=1) | ~2 GB | Single series |
| Inference (batch=30) | ~3 GB | All 30 stocks |
| Fine-tuning (batch=8) | ~6 GB | LoRA recommended |

**Input format:** TimesFM expects a 2D array of shape `(batch, context_length)` with a frequency indicator.

**Zero-shot code skeleton:**

```python
import timesfm

# Initialize model
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        per_core_batch_size=32,
        horizon_len=22,        # max forecast horizon
        backend="gpu",         # or "cpu"
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.0-200m-pytorch",
    ),
)

def timesfm_forecast(model, context: np.ndarray,
                     horizon: int = 1,
                     freq: int = 0) -> dict:
    """Generate forecast from TimesFM.

    Parameters
    ----------
    context : np.ndarray
        1D array of historical RV values. TimesFM uses up to 512 context.
    horizon : int
        Forecast horizon (1, 5, or 22).
    freq : int
        Frequency indicator. 0 = high frequency (daily/hourly).
        TimesFM infers frequency; for daily financial data, use 0.

    Returns
    -------
    dict with 'mean' and quantile forecasts.
    """
    # TimesFM expects 2D input: (batch_size, context_length)
    context_2d = context.reshape(1, -1)

    # forecast returns: point_forecast (batch, horizon), quantile_forecast
    point_forecast, quantile_forecast = model.forecast(
        inputs=context_2d,
        freq=[freq],
    )

    # point_forecast shape: (1, horizon)
    # quantile_forecast shape: (1, horizon, n_quantiles)
    return {
        'mean': point_forecast[0, :horizon],
        'quantiles': quantile_forecast[0, :horizon, :] if quantile_forecast is not None else None,
        # quantile indices: [0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95]
    }
```

**Point forecast extraction:** TimesFM returns an explicit mean forecast. Use this directly. For consistency with other models, also extract the median (50th percentile) from the quantile output.

**Fine-tuning procedure:**

TimesFM supports in-context fine-tuning (ICF) as described in Das et al. (2024, ICML 2025):

```python
# TimesFM fine-tuning via in-context learning (no weight updates)
# The model conditions on exemplar (context, target) pairs
# This is "few-shot" learning, not traditional fine-tuning

# For actual weight fine-tuning, use LoRA:
# See: https://github.com/google-research/timesfm
# The recommended approach for financial data is:
# 1. Prepare training data as list of (context, target) tuples
# 2. Use HuggingFace PEFT (LoRA) on the underlying transformer
# 3. Fine-tune for 3-5 epochs with LR=1e-5, rank=8

# Goel et al. (2025, arxiv:2505.11163) found that fine-tuning
# TimesFM on RV data significantly improved performance.
# Their approach: train on 80% of dates, validate on 10%, test on 10%.
```

**Known issues with financial data:**
- Goel et al. (2025) showed TimesFM 1.0 underperforms HAR zero-shot but beats it after fine-tuning on RV. Expect similar behavior with v2.5.
- TimesFM can produce NaN forecasts for very short context lengths (<32 observations). Always provide at least 64 observations of context.
- The frequency indicator (`freq`) affects internal patch handling. For daily data, `freq=0` (high-frequency) is appropriate. Do NOT use calendar-based frequency indicators meant for weekly/monthly data.
- TimesFM's patching mechanism (patch size = 32) means it "sees" data in chunks of 32 days. For volatility with 5-day and 22-day cycles, this naturally aligns with the weekly and monthly patterns in HAR.

---

### Model 3: Moirai 2.0 (Salesforce, November 2025)

**Architecture:** Decoder-only transformer with quantile output heads and multi-token prediction. Trained on 295 billion observations (largest training dataset of any TSFM).

**HuggingFace model IDs:**
- `Salesforce/moirai-2.0-R-small` (11.4M params)
- `Salesforce/moirai-2.0-R-base` (87.1M params)
- `Salesforce/moirai-2.0-R-large` (305M params)

**Installation:**

```bash
pip install uni2ts gluonts
# uni2ts is Salesforce's unified time series library
# GluonTS provides the data loading interface
# Requires: torch >= 2.1, einops, huggingface_hub
```

**GPU/memory requirements:**
| Variant | Params | VRAM (inference) | VRAM (fine-tune) |
|---------|--------|-----------------|-----------------|
| small | 11.4M | ~1 GB | ~2 GB |
| base | 87.1M | ~2 GB | ~5 GB |
| large | 305M | ~4 GB | ~10 GB |

**Input format:** Moirai uses the GluonTS `PandasDataset` or `ListDataset` format.

**Zero-shot code skeleton:**

```python
import torch
from gluonts.dataset.pandas import PandasDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

def moirai_forecast(rv_series: pd.Series,
                    prediction_length: int = 1,
                    model_size: str = "base",
                    num_samples: int = 100) -> dict:
    """Generate forecast from Moirai 2.0.

    Parameters
    ----------
    rv_series : pd.Series
        Historical RV with DatetimeIndex (business day frequency).
    prediction_length : int
        Forecast horizon (1, 5, or 22).
    model_size : str
        One of 'small', 'base', 'large'.
    num_samples : int
        Number of sample paths.

    Returns
    -------
    dict with 'mean', 'median', 'samples'.
    """
    model_id = f"Salesforce/moirai-2.0-R-{model_size}"

    # Load model
    module = MoiraiModule.from_pretrained(model_id)
    model = MoiraiForecast(
        module=module,
        prediction_length=prediction_length,
        context_length=512,         # use up to 512 days of context
        patch_size="auto",          # let model choose
        num_samples=num_samples,
        target_dim=1,               # univariate
        feat_dynamic_real_dim=0,    # no covariates
        past_feat_dynamic_real_dim=0,
    )

    # Prepare data in GluonTS format
    # Ensure business day frequency
    rv_bday = rv_series.copy()
    rv_bday.index = pd.DatetimeIndex(rv_bday.index, freq='B')

    dataset = PandasDataset(
        dataframes=rv_bday.to_frame(name='target'),
        target='target',
        freq='B',
    )

    # Generate forecasts
    predictor = model.create_predictor(batch_size=1)
    forecasts = list(predictor.predict(dataset))
    fc = forecasts[0]

    samples = fc.samples  # shape: (num_samples, prediction_length)
    return {
        'mean': samples.mean(axis=0),
        'median': np.median(samples, axis=0),
        'q10': np.percentile(samples, 10, axis=0),
        'q25': np.percentile(samples, 25, axis=0),
        'q75': np.percentile(samples, 75, axis=0),
        'q90': np.percentile(samples, 90, axis=0),
        'samples': samples,
    }
```

**Point forecast extraction:** Use the median of the sample distribution. Moirai 2.0 also supports direct quantile output; if using quantile mode, extract the 50th percentile directly.

**Fine-tuning procedure:**

```python
# Moirai 2.0 fine-tuning via uni2ts library
# See: https://github.com/SalesforceAIResearch/uni2ts

# Key steps:
# 1. Prepare a GluonTS ListDataset with all training series
# 2. Use the uni2ts fine-tuning script with LoRA or full fine-tuning
# 3. Recommended: LoRA with rank=16, alpha=32, LR=1e-4, epochs=10

# from uni2ts.training import MoiraiFineTuner
# finetuner = MoiraiFineTuner(
#     model_id="Salesforce/moirai-2.0-R-base",
#     learning_rate=1e-4,
#     num_epochs=10,
#     batch_size=32,
#     lora_rank=16,
# )
# finetuner.train(train_dataset)
# finetuner.save("./moirai_finetuned_rv")
```

**Known issues with financial data:**
- Moirai expects `freq='B'` (business day) for daily financial data. If the DatetimeIndex has gaps (holidays), Moirai handles them via its "any-variate" framework, but verify that NaN handling is correct.
- The GluonTS `PandasDataset` requires the index to have a proper `freq` attribute. If your dates are irregular, set `freq='B'` and fill missing dates with NaN (Moirai is trained to handle NaN inputs).
- Moirai's multi-token prediction mode may produce smoother forecasts than single-token autoregressive mode. For h=1, both modes should give similar results; for h=22, multi-token may be better.
- Large model (305M) may be slow on CPU. Benchmark inference time per forecast step; if >1 second per step, use the base model (87M) for the main results and large as robustness.

---

### Model 4: Chronos-Bolt (Amazon, October 2025)

**Architecture:** Encoder-decoder (T5-based) with direct quantile regression head. Distilled from the original Chronos model for 250x faster inference. Shares the `chronos-forecasting` codebase with Chronos-2.

**HuggingFace model IDs:**
- `amazon/chronos-bolt-tiny` (9M params)
- `amazon/chronos-bolt-mini` (21M params)
- `amazon/chronos-bolt-small` (48M params)
- `amazon/chronos-bolt-base` (205M params)

**Installation:** Same as Chronos-2 (`pip install chronos-forecasting torch transformers`)

**GPU/memory requirements:** See Chronos-2 table above. Bolt variants are significantly faster and smaller.

**Zero-shot code skeleton:**

```python
# Chronos-Bolt uses the SAME ChronosPipeline interface as Chronos-2
# The only difference is the model ID

pipeline_bolt = ChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-base",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float32,
)

# Use the same chronos_forecast() function from Model 1 above
# with pipeline=pipeline_bolt
```

**Key difference from Chronos-2:** Bolt uses direct quantile regression rather than the encoder-only architecture of Chronos-2. This means:
- Bolt outputs specific quantile levels directly (not samples from a distribution)
- To get sample paths, Bolt uses quantile interpolation
- Bolt has a maximum context window of 512 tokens (vs. 2048 for Chronos-2)
- Bolt is 250x faster at inference than original Chronos (and ~10-50x faster than Chronos-2)

**Fine-tuning:** Same procedure as Chronos-2 (shared codebase).

**Known issues:**
- The 250x speedup makes Bolt attractive for the expanding-window evaluation (500+ refits). Even if Chronos-2 gives slightly better accuracy, the computational cost of running Chronos-2 on 500 expanding windows x 30 stocks is substantial. Bolt may be the practical choice for the main results.
- Bolt's context window (512) is sufficient for capturing the monthly (22-day) pattern but may miss longer cycles. For ARFIMA-like long memory (d~0.4, effective memory >200 days), 512 tokens should be adequate.

---

### TSFM Evaluation Workflow

For each TSFM, the evaluation proceeds as:

```
For each stock i in {1, ..., 30}:
    For each forecast date t in OOS period:
        context = RV_{i, t-L:t-1}  where L = min(context_window, available_history)
        forecast = TSFM.predict(context, horizon=h)
        store forecast[i, t, h]
```

**Critical implementation details for all TSFMs:**

1. **Input scaling/normalization.** Some TSFMs (especially Chronos) apply internal normalization. Do NOT pre-normalize the input. Pass raw RV values. The model handles scaling internally.

2. **Log vs. level.** TSFMs are trained on diverse data at various scales. Passing `log(RV)` may confuse the model if it expects level data. Test both: (a) raw RV input, (b) log(RV) input with exp() post-processing. Report which works better.

3. **Batch inference.** For computational efficiency, batch all 30 stocks into a single forward pass when the model supports batched input. Chronos and TimesFM support this natively. Moirai via GluonTS also supports batching.

4. **Random seed.** Set `torch.manual_seed(42)` and `np.random.seed(42)` before each forecast call that involves sampling. Report the seed. For reproducibility, consider generating 1000 samples and storing them, then computing statistics from stored samples.

5. **Context length selection.** Use the maximum context available (up to the model's limit). For a forecast at date t with training starting at date 0, provide context from `max(0, t - context_limit)` to `t-1`.

6. **Negative forecast handling.** RV cannot be negative. Any TSFM sample < 0 should be clipped to a small positive number (e.g., `1e-8`) before computing loss functions, especially QLIKE which involves `log(forecast)`.

---

## 2C. Forecast Evaluation Framework

### 1. Train/Test Split

**Design:** Expanding (recursive) window with fixed out-of-sample (OOS) period.

```
Total data:  |-------- Training (expanding) --------|--- OOS (fixed) ---|
                                                     ^                  ^
                                                  OOS start          OOS end

OOS period: last 2 years = ~504 trading days (2023-01-03 to 2024-12-31)
Initial training window: everything before OOS start (~4,842 days for full-sample stocks)
Minimum training window: 500 days (required for monthly HAR lag + stable estimation)
```

**Implementation:**

```python
OOS_START = '2023-01-03'  # first trading day of 2023
OOS_END = '2024-12-31'    # last trading day of 2024

def get_train_test_dates(rv_series: pd.Series,
                         oos_start: str = OOS_START) -> tuple:
    """Split dates into expanding training and fixed OOS.

    Returns
    -------
    tuple of (all_dates, oos_dates)
    """
    all_dates = rv_series.dropna().index
    oos_dates = all_dates[all_dates >= oos_start]
    return all_dates, oos_dates
```

**Re-estimation frequency:**
- HAR / HAR-J / HAR-RS / HARQ: re-estimate coefficients every trading day (OLS is fast, <1ms per fit)
- ARFIMA: re-estimate d every 22 days, ARMA coefficients every day
- Realized GARCH: re-estimate every 22 days (MLE is slower, ~1-5s per fit)
- TSFMs (zero-shot): no estimation; just update the context window each day
- TSFMs (fine-tuned): fine-tune once on the initial training set; optionally re-fine-tune quarterly

**Short-history stocks:** DOW starts ~2019 (1,266 obs). With 504 OOS days, only ~762 training days remain. This is above the 500-day minimum, so include DOW. For TRV, CRM, V: similar calculation shows sufficient training data.

---

### 2. Forecast Horizons

**Horizons:** h = 1, 5, 22 trading days

**Multi-step approach:**

| Horizon | Econometric models | TSFMs |
|---------|-------------------|-------|
| h = 1 | One-step-ahead forecast from each model | `prediction_length=1` |
| h = 5 | Direct projection: regress `RV^{(5)}_{t+5} = (1/5)*sum_{i=1}^{5} RV_{t+i}` on HAR regressors at time t | `prediction_length=5`, take mean of 5-day path, OR take day-5 forecast directly |
| h = 22 | Direct projection: regress `RV^{(22)}_{t+22}` on HAR regressors at time t | `prediction_length=22`, take mean of 22-day path, OR take day-22 forecast directly |

**Critical distinction for multi-step TSFM evaluation:**

There are two ways to construct the h-day-ahead RV forecast from TSFMs:
1. **Direct h-step forecast:** Ask the model for the value at position h in its output. This corresponds to a "day h" point forecast.
2. **Path average:** Ask the model for a path of length h and average. This corresponds to the h-day average RV forecast.

For comparability with HAR direct projections, use **path average** (option 2), since the HAR h-step specification targets `RV^{(h)}_{t+h} = (1/h) * sum RV_{t+i}`.

```python
def multi_step_forecast_tsfm(samples: np.ndarray, h: int) -> float:
    """Compute h-day average RV forecast from TSFM sample paths.

    Parameters
    ----------
    samples : np.ndarray, shape (num_samples, prediction_length)
        Sample paths from TSFM. prediction_length >= h.
    h : int
        Forecast horizon.

    Returns
    -------
    float: point forecast (median of sample-path averages).
    """
    # Average each sample path over h days, then take median across samples
    path_averages = samples[:, :h].mean(axis=1)  # (num_samples,)
    return np.median(path_averages)
```

---

### 3. Loss Functions

#### MSE (Mean Squared Error)

```
MSE = (1/T) * sum_{t=1}^{T} (RV_t - hat{RV}_t)^2
```

Standard, penalizes large errors quadratically. Sensitive to outliers (e.g., COVID-19 period).

#### MAE (Mean Absolute Error)

```
MAE = (1/T) * sum_{t=1}^{T} |RV_t - hat{RV}_t|
```

More robust to outliers than MSE.

#### QLIKE (Quasi-Likelihood Loss)

```
QLIKE = (1/T) * sum_{t=1}^{T} [RV_t / hat{RV}_t - log(RV_t / hat{RV}_t) - 1]
```

Equivalently: `QLIKE = (1/T) * sum [RV_t / hat{RV}_t - log(RV_t) + log(hat{RV}_t) - 1]`

QLIKE is the **primary loss function** for realized volatility forecasting (Patton, 2011). It is:
- Robust to noise in the volatility proxy (consistent ranking under imperfect proxies)
- The natural loss function for the quasi-Gaussian likelihood of volatility
- Standard in the RV forecasting literature (HAR, HARQ, etc.)

```python
def qlike_loss(rv_actual: np.ndarray, rv_forecast: np.ndarray) -> float:
    """Compute QLIKE loss (Patton 2011).

    Both inputs must be strictly positive.
    """
    assert np.all(rv_actual > 0) and np.all(rv_forecast > 0)
    ratio = rv_actual / rv_forecast
    return np.mean(ratio - np.log(ratio) - 1)
```

#### R^2_OOS (Out-of-Sample R-squared)

```
R^2_OOS = 1 - sum(RV_t - hat{RV}_t)^2 / sum(RV_t - bar{RV})^2
```

where `bar{RV}` is the expanding-window historical mean (the "no-change" benchmark). A positive R^2_OOS means the model beats the historical mean; negative means it is worse than the mean.

```python
def r2_oos(rv_actual: np.ndarray, rv_forecast: np.ndarray,
           rv_mean_expanding: np.ndarray) -> float:
    """Out-of-sample R-squared (Campbell & Thompson, 2008)."""
    ss_res = np.sum((rv_actual - rv_forecast)**2)
    ss_tot = np.sum((rv_actual - rv_mean_expanding)**2)
    return 1 - ss_res / ss_tot
```

**Implementation:**

```python
def compute_all_losses(rv_actual: pd.Series,
                       rv_forecast: pd.Series) -> dict:
    """Compute all loss functions for a single model-stock-horizon.

    Parameters
    ----------
    rv_actual : pd.Series
        Realized values in OOS period.
    rv_forecast : pd.Series
        Model forecasts, aligned with rv_actual.

    Returns
    -------
    dict with 'MSE', 'MAE', 'QLIKE', 'R2_OOS'.
    """
    actual = rv_actual.values
    forecast = rv_forecast.values

    # Clip forecasts to avoid division by zero in QLIKE
    forecast_clipped = np.maximum(forecast, 1e-10)

    # Expanding-window mean for R2_OOS
    # (computed separately from the training window)
    cumulative_mean = rv_actual.expanding().mean().shift(1).dropna()

    mse = np.mean((actual - forecast)**2)
    mae = np.mean(np.abs(actual - forecast))
    ql = qlike_loss(actual, forecast_clipped)

    # R2_OOS against historical mean benchmark
    ss_res = np.sum((actual - forecast)**2)
    # Use the full training sample mean as the benchmark
    # (not expanding within OOS; this is the standard in finance)
    rv_mean = actual.mean()  # placeholder; replace with proper expanding mean
    ss_tot = np.sum((actual - rv_mean)**2)
    r2oos = 1 - ss_res / ss_tot

    return {
        'MSE': mse,
        'MAE': mae,
        'QLIKE': ql,
        'R2_OOS': r2oos,
    }
```

---

### 4. Statistical Tests

#### Diebold-Mariano Test (Diebold & Mariano, 1995)

Tests whether two forecast models have equal predictive accuracy.

**Null hypothesis:** `E[d_t] = 0` where `d_t = L(e_{1,t}) - L(e_{2,t})` and `L(.)` is the loss function.

```python
from scipy import stats as sp_stats

def diebold_mariano_test(loss1: np.ndarray, loss2: np.ndarray,
                         h: int = 1, method: str = 'HLN') -> dict:
    """Diebold-Mariano test for equal predictive accuracy.

    Parameters
    ----------
    loss1 : np.ndarray
        Loss series for model 1 (e.g., squared errors).
    loss2 : np.ndarray
        Loss series for model 2.
    h : int
        Forecast horizon (for HAC correction).
    method : str
        'DM' for original DM test, 'HLN' for Harvey-Leybourne-Newbold
        (1997) finite-sample correction. Use 'HLN' (default).

    Returns
    -------
    dict with 'dm_stat', 'p_value', 'mean_diff'.
    """
    d = loss1 - loss2
    T = len(d)
    d_bar = d.mean()

    # HAC variance estimator (Newey-West with h-1 lags)
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1]
        gamma_sum += 2 * gamma_k

    var_d = (gamma_0 + gamma_sum) / T

    if var_d <= 0:
        # Fallback: use simple variance
        var_d = gamma_0 / T

    dm_stat = d_bar / np.sqrt(var_d)

    # Harvey-Leybourne-Newbold correction
    if method == 'HLN':
        correction = np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
        dm_stat = dm_stat * correction

    p_value = 2 * sp_stats.norm.sf(np.abs(dm_stat))  # two-sided

    return {
        'dm_stat': dm_stat,
        'p_value': p_value,
        'mean_diff': d_bar,
        'model1_better': d_bar < 0,
    }
```

**Usage pattern:**

```python
# Compare each model against HAR (the standard benchmark)
for model_name in all_models:
    if model_name == 'HAR':
        continue
    result = diebold_mariano_test(
        loss1=losses[model_name],
        loss2=losses['HAR'],
        h=forecast_horizon,
    )
    print(f"{model_name} vs HAR: DM={result['dm_stat']:.3f}, p={result['p_value']:.4f}")
```

**Known pitfalls:**
- DM test assumes loss differentials are covariance stationary. For RV forecasts during high-volatility regimes (e.g., COVID), this assumption may be violated. Report results for full OOS and for subperiods.
- For h>1 direct forecasts, use h-1 Newey-West lags. For h=1, use 0 lags (or a small number like 1-2 for robustness).
- The original DM test can be oversized in small samples. The HLN correction (Harvey, Leybourne & Newbold, 1997) is standard and should be used by default.
- Use the QLIKE-based loss differential as the primary test, since QLIKE is robust to the choice of volatility proxy (Patton, 2011).

---

#### Model Confidence Set (Hansen, Lunde & Nason, 2011)

The MCS identifies the set of models that contains the best model with a given confidence level (e.g., 90%). It avoids the multiple-testing problem inherent in pairwise DM tests.

**Algorithm:**

1. Start with the full set of models M.
2. Test if all models in M have equal predictive ability (equivalence test using T_max or T_R statistic).
3. If rejected, eliminate the worst model.
4. Repeat until the equivalence test is not rejected.
5. The surviving set is the MCS at confidence level (1 - alpha).

```python
def model_confidence_set(losses: dict, alpha: float = 0.10,
                         n_bootstrap: int = 10000,
                         block_length: int = 22,
                         seed: int = 42) -> dict:
    """Compute the Model Confidence Set (Hansen, Lunde & Nason, 2011).

    Parameters
    ----------
    losses : dict
        {model_name: np.ndarray of losses}, all arrays same length.
    alpha : float
        Significance level (0.10 = 90% MCS).
    n_bootstrap : int
        Number of bootstrap replications for p-value computation.
    block_length : int
        Block length for stationary bootstrap (22 = monthly for daily data).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with 'mcs_models' (list of surviving models),
              'p_values' (dict of MCS p-values for each model),
              'elimination_order' (list of eliminated models in order).
    """
    rng = np.random.RandomState(seed)
    model_names = list(losses.keys())
    T = len(next(iter(losses.values())))
    loss_matrix = np.column_stack([losses[m] for m in model_names])
    # loss_matrix shape: (T, n_models)

    surviving = list(range(len(model_names)))
    elimination_order = []
    p_values = {}

    while len(surviving) > 1:
        # Compute pairwise loss differentials for surviving models
        n = len(surviving)
        d_bar = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                d_bar[i, j] = (loss_matrix[:, surviving[i]] -
                               loss_matrix[:, surviving[j]]).mean()

        # T_R statistic: range of standardized loss differentials
        # Simplified: use the max of d_bar_i. (mean loss of model i minus mean of all)
        d_bar_i = np.array([
            np.mean([d_bar[i, j] for j in range(n) if j != i])
            for i in range(n)
        ])

        # Bootstrap the T_R statistic
        t_R_observed = np.max(d_bar_i) / np.std(d_bar_i) if np.std(d_bar_i) > 0 else 0

        # Stationary bootstrap
        boot_stats = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            # Generate bootstrap indices (stationary bootstrap)
            boot_idx = _stationary_bootstrap_indices(T, block_length, rng)
            boot_losses = loss_matrix[boot_idx][:, surviving]

            boot_d_bar_i = np.zeros(n)
            for i in range(n):
                diffs = [boot_losses[:, i] - boot_losses[:, j]
                         for j in range(n) if j != i]
                boot_d_bar_i[i] = np.mean([d.mean() for d in diffs])

            boot_d_bar_i_centered = boot_d_bar_i - d_bar_i
            std = np.std(boot_d_bar_i_centered)
            boot_stats[b] = np.max(boot_d_bar_i_centered) / std if std > 0 else 0

        p_val = np.mean(boot_stats >= t_R_observed)

        if p_val <= alpha:
            # Reject: eliminate worst model
            worst_idx = np.argmax(d_bar_i)
            worst_model = surviving[worst_idx]
            elimination_order.append(model_names[worst_model])
            p_values[model_names[worst_model]] = p_val
            surviving.remove(worst_model)
        else:
            # Cannot reject: all remaining models are in the MCS
            for idx in surviving:
                p_values[model_names[idx]] = p_val
            break

    if len(surviving) == 1:
        p_values[model_names[surviving[0]]] = 1.0

    mcs_models = [model_names[idx] for idx in surviving]

    return {
        'mcs_models': mcs_models,
        'p_values': p_values,
        'elimination_order': elimination_order,
    }


def _stationary_bootstrap_indices(T: int, block_length: int,
                                  rng: np.random.RandomState) -> np.ndarray:
    """Generate indices for the stationary bootstrap (Politis & Romano, 1994).

    Each index is drawn from a geometric distribution with mean block_length.
    """
    prob = 1.0 / block_length
    indices = np.zeros(T, dtype=int)
    indices[0] = rng.randint(0, T)
    for t in range(1, T):
        if rng.random() < prob:
            indices[t] = rng.randint(0, T)
        else:
            indices[t] = (indices[t-1] + 1) % T
    return indices
```

**Known pitfalls:**
- The MCS bootstrap is computationally expensive: O(n_bootstrap x n_models^2 x T). With 10 models, 10,000 bootstraps, and T=500, this takes ~30 seconds. Acceptable.
- Block length should match the serial dependence in loss differentials. For daily RV with h=1, `block_length=22` (monthly) is standard. For h=22, use `block_length=44` or `66`.
- The MCS can be sensitive to the choice of alpha. Report results for alpha = 0.10 (standard) and alpha = 0.25 (broader set) as in HLN (2011).
- Consider using the `arch` package's `MCS` class if available (`from arch.bootstrap import MCS`). This provides a tested implementation.

**arch package MCS (if available):**

```python
from arch.bootstrap import MCS

losses_df = pd.DataFrame(losses)
mcs = MCS(losses_df, size=0.10, method='max', reps=10000, block_size=22, seed=42)
mcs.compute()
print(mcs.included)   # models in the MCS
print(mcs.pvalues)    # MCS p-values
```

---

### 5. Table and Figure Format

#### Main results table: Model x Horizon x Loss

One table per loss function, showing all models x horizons. Plus an aggregate table.

**Table format (per loss function, e.g., QLIKE):**

```
Table X: Out-of-Sample QLIKE Loss (2023-2024)
Cross-sectional average across 30 DJIA stocks

                          h = 1           h = 5           h = 22
Model                  QLIKE   Rank    QLIKE   Rank    QLIKE   Rank
----------------------------------------------------------------------
Panel A: Econometric Benchmarks
HAR                    0.XXXX   (X)    0.XXXX   (X)    0.XXXX   (X)
HAR-J                  0.XXXX   (X)    0.XXXX   (X)    0.XXXX   (X)
HAR-RS                 0.XXXX   (X)    0.XXXX   (X)    0.XXXX   (X)
HARQ                   0.XXXX   (X)    0.XXXX   (X)    0.XXXX   (X)
Realized GARCH         0.XXXX   (X)    0.XXXX   (X)    0.XXXX   (X)
ARFIMA                 0.XXXX   (X)    0.XXXX   (X)    0.XXXX   (X)

Panel B: TSFMs (Zero-Shot)
Chronos-2              0.XXXX   (X)    0.XXXX   (X)    0.XXXX   (X)
Chronos-Bolt           0.XXXX   (X)    0.XXXX   (X)    0.XXXX   (X)
TimesFM 2.5            0.XXXX   (X)    0.XXXX   (X)    0.XXXX   (X)
Moirai 2.0             0.XXXX   (X)    0.XXXX   (X)    0.XXXX   (X)

Panel C: TSFMs (Fine-Tuned)
Chronos-2-FT           0.XXXX   (X)    0.XXXX   (X)    0.XXXX   (X)
Chronos-Bolt-FT        0.XXXX   (X)    0.XXXX   (X)    0.XXXX   (X)
TimesFM 2.5-FT         0.XXXX   (X)    0.XXXX   (X)    0.XXXX   (X)
Moirai 2.0-FT          0.XXXX   (X)    0.XXXX   (X)    0.XXXX   (X)

Notes: Bold = best in column. * = in MCS at 10%. Superscript a/b/c = DM test
rejects equal accuracy vs HAR at 1%/5%/10%.
```

#### DM test table:

```
Table Y: Diebold-Mariano Test p-values vs. HAR Benchmark (QLIKE loss)

                          h = 1           h = 5           h = 22
Model                  DM stat  p-val   DM stat  p-val   DM stat  p-val
------------------------------------------------------------------------
HAR-J                  -X.XX    0.XXX   ...
HAR-RS                 -X.XX    0.XXX   ...
...
Chronos-2 (ZS)         X.XX     0.XXX   ...
Chronos-2 (FT)        -X.XX     0.XXX   ...
...

Notes: Negative DM stat = model beats HAR. Two-sided test with HLN correction.
```

#### MCS results table:

```
Table Z: Model Confidence Set (90% confidence)

                          h = 1           h = 5           h = 22
Model                  p-MCS  In MCS?  p-MCS  In MCS?  p-MCS  In MCS?
----------------------------------------------------------------------
HAR                    0.XXX    Yes     ...
...
Chronos-2 (FT)         0.XXX    Yes     ...
```

#### Figures:

1. **Forecast comparison plot:** Time series of actual RV vs. forecasts from top 3-4 models for 2-3 representative stocks. Zoom into high-volatility and low-volatility subperiods.
2. **Cumulative loss differential plot:** Cumulative sum of `L(HAR) - L(Model)` over OOS period for each model. Positive values = model beats HAR up to that point. Shows when each model gains/loses advantage.
3. **MCS inclusion heatmap:** Models (rows) x Loss/Horizon combinations (columns), shaded by MCS p-value.
4. **Bar chart:** Average rank across all stocks for each model x horizon, grouped by loss function.

---

## Appendix: Package Versions and Environment

```
# requirements.txt (exact versions for reproducibility)
numpy>=1.24.0,<2.0
pandas>=2.0.0
statsmodels>=0.14.0
arch>=7.0.0
scipy>=1.11.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
fracdiff>=0.9.0
yfinance>=0.2.30
torch>=2.1.0
transformers>=4.35.0
chronos-forecasting>=1.4.0
timesfm>=1.3.0
uni2ts>=1.0.0
gluonts>=0.14.0
tqdm>=4.65.0
```

**Hardware assumption:**
- GPU: NVIDIA with at least 8 GB VRAM (e.g., RTX 3070/4070 or A100)
- If no GPU: Chronos-Bolt-small and TimesFM are feasible on CPU (slow); Moirai-large and Chronos-2 require GPU
- CPU fallback: use smallest model variants and extend runtime budget

**Estimated compute time (per stock, per horizon, full OOS ~504 days):**

| Model | Re-estimation | Time per OOS step | Total (1 stock, 1 horizon) |
|-------|---------------|-------------------|---------------------------|
| HAR / HAR-J / HAR-RS / HARQ | Every day | ~1 ms | ~0.5 seconds |
| ARFIMA | d every 22 days | ~50 ms | ~25 seconds |
| Realized GARCH | Every 22 days | ~2 s (MLE) | ~50 seconds |
| Chronos-Bolt-small (GPU) | None | ~50 ms | ~25 seconds |
| Chronos-2-base (GPU) | None | ~200 ms | ~100 seconds |
| TimesFM (GPU) | None | ~100 ms | ~50 seconds |
| Moirai-base (GPU) | None | ~150 ms | ~75 seconds |

**Total for 30 stocks x 3 horizons x ~14 models:**
- Econometric models: ~30 x 3 x 6 x ~30s avg = ~2.7 hours
- TSFMs zero-shot: ~30 x 3 x 4 x ~60s avg = ~6 hours
- TSFMs fine-tuned: fine-tuning ~2h per model, then ~6 hours inference = ~14 hours
- **Grand total: ~24 hours on a single GPU machine (parallelizable)**

---

## Appendix: Execution Order

```
Phase 0: Data preparation
    0.1  Load and clean RV data (replace zeros with NaN)
    0.2  Download daily returns from Yahoo Finance for all 30 tickers
    0.3  Align dates between RV and returns datasets
    0.4  Construct all derived features: J_t, semivariances, RQ interactions
    0.5  Save processed panel to data/processed/panel_rv.parquet

Phase 1: Econometric baselines (Tier 1)
    1.1  HAR: expanding window, h=1,5,22
    1.2  HAR-J: same
    1.3  HAR-RS: same
    1.4  HARQ: same
    1.5  ARFIMA: expanding window with periodic d re-estimation
    1.6  Realized GARCH: expanding window
    Save all forecasts to data/processed/forecasts_econometric.parquet

Phase 2: TSFM zero-shot evaluation
    2.1  Chronos-2: zero-shot, h=1,5,22, all stocks
    2.2  Chronos-Bolt: zero-shot, h=1,5,22, all stocks
    2.3  TimesFM 2.5: zero-shot, h=1,5,22, all stocks
    2.4  Moirai 2.0: zero-shot, h=1,5,22, all stocks
    Save to data/processed/forecasts_tsfm_zeroshot.parquet

Phase 3: TSFM fine-tuned evaluation
    3.1  Fine-tune each TSFM on training data (all stocks pooled)
    3.2  Generate forecasts with fine-tuned models
    Save to data/processed/forecasts_tsfm_finetuned.parquet

Phase 4: Evaluation
    4.1  Compute MSE, MAE, QLIKE, R2_OOS for all model-stock-horizon combos
    4.2  Run pairwise DM tests vs HAR benchmark
    4.3  Run MCS at alpha=0.10 and alpha=0.25
    4.4  Generate all tables and figures
    Save results to data/processed/evaluation_results.parquet

Phase 5: Robustness checks
    5.1  1-min vs 5-min RV as forecast target
    5.2  Log-HAR variant
    5.3  Subsample analysis (pre-COVID, COVID, post-COVID)
    5.4  Asset-level analysis (which stocks do TSFMs help most?)
    5.5  Context length sensitivity for TSFMs
    5.6  Tier 2 econometric models (if time permits)
```
