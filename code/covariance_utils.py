"""
covariance_utils.py — Utilities for realized covariance matrix operations.

Provides:
    - vech / ivech: half-vectorization and inverse
    - ensure_psd: project to nearest PSD matrix (Higham 2002)
    - cov_to_drd / drd_to_cov: D-R-D decomposition (Sigma = D R D)
    - build_gmv_weights: global minimum variance portfolio weights
"""

import numpy as np
from typing import Tuple


def vech(matrix: np.ndarray) -> np.ndarray:
    """Extract the lower-triangular elements (including diagonal) of a symmetric matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Symmetric (N x N) matrix.

    Returns
    -------
    np.ndarray
        Vector of length N*(N+1)/2.
    """
    n = matrix.shape[0]
    idx = np.tril_indices(n)
    return matrix[idx]


def ivech(vector: np.ndarray, n: int) -> np.ndarray:
    """Reconstruct a symmetric matrix from its half-vectorization.

    Parameters
    ----------
    vector : np.ndarray
        Vector of length N*(N+1)/2.
    n : int
        Matrix dimension.

    Returns
    -------
    np.ndarray
        Symmetric (N x N) matrix.
    """
    mat = np.zeros((n, n))
    idx = np.tril_indices(n)
    mat[idx] = vector
    mat = mat + mat.T - np.diag(np.diag(mat))
    return mat


def ensure_psd(matrix: np.ndarray, min_eigenvalue: float = 1e-10) -> np.ndarray:
    """Project a symmetric matrix to the nearest positive semi-definite matrix.

    Uses eigenvalue clipping (simpler than full Higham 2002 but sufficient
    when the matrix is close to PSD already, which is our case).

    Parameters
    ----------
    matrix : np.ndarray
        Symmetric (N x N) matrix.
    min_eigenvalue : float
        Floor for eigenvalues.

    Returns
    -------
    np.ndarray
        Nearest PSD matrix.
    """
    matrix = (matrix + matrix.T) / 2  # enforce symmetry
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, min_eigenvalue)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def ensure_correlation(matrix: np.ndarray) -> np.ndarray:
    """Project to valid correlation matrix (PSD with unit diagonal).

    Parameters
    ----------
    matrix : np.ndarray
        Approximate correlation matrix.

    Returns
    -------
    np.ndarray
        Valid correlation matrix.
    """
    matrix = ensure_psd(matrix)
    d = np.sqrt(np.diag(matrix))
    d[d == 0] = 1.0
    corr = matrix / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


def cov_to_drd(cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose covariance: Sigma = D R D.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix (N x N).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        D: vector of standard deviations (length N).
        R: correlation matrix (N x N).
    """
    d = np.sqrt(np.maximum(np.diag(cov), 0))
    d_safe = np.where(d > 0, d, 1.0)
    D_inv = np.diag(1.0 / d_safe)
    R = D_inv @ cov @ D_inv
    np.fill_diagonal(R, 1.0)
    return d, R


def drd_to_cov(d: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Recombine Sigma = D R D from standard deviations and correlations.

    Parameters
    ----------
    d : np.ndarray
        Standard deviations (length N).
    R : np.ndarray
        Correlation matrix (N x N).

    Returns
    -------
    np.ndarray
        Covariance matrix (N x N).
    """
    D = np.diag(d)
    return D @ R @ D


def build_gmv_weights(cov_matrix: np.ndarray) -> np.ndarray:
    """Compute global minimum variance portfolio weights.

    w = Sigma^{-1} 1 / (1' Sigma^{-1} 1)

    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix (N x N), must be PSD.

    Returns
    -------
    np.ndarray
        Portfolio weights (length N), summing to 1.
    """
    n = cov_matrix.shape[0]
    ones = np.ones(n)
    try:
        inv_sigma = np.linalg.solve(cov_matrix, ones)
    except np.linalg.LinAlgError:
        # Fallback: regularize
        cov_reg = cov_matrix + 1e-8 * np.eye(n)
        inv_sigma = np.linalg.solve(cov_reg, ones)
    w = inv_sigma / inv_sigma.sum()
    return w


def get_pair_list(assets: list) -> list:
    """Generate list of (asset1, asset2) pairs for upper triangular + diagonal.

    Parameters
    ----------
    assets : list
        Sorted list of asset names.

    Returns
    -------
    list of tuples
        All (i, j) pairs where i <= j.
    """
    pairs = []
    for i, a1 in enumerate(assets):
        for j, a2 in enumerate(assets):
            if j >= i:
                pairs.append((a1, a2))
    return pairs
