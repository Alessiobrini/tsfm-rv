"""Generate a PDF checklist of what cluster jobs still need to run."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib import colors
from reportlab.lib.units import inch
import os

out = os.path.join(os.path.dirname(__file__), '..', 'paper', 'cluster_run_checklist.pdf')
doc = SimpleDocTemplate(out, pagesize=letter, topMargin=0.6*inch, bottomMargin=0.6*inch)
styles = getSampleStyleSheet()

title_style = ParagraphStyle('Title2', parent=styles['Title'], fontSize=16, spaceAfter=6)
h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=13, spaceAfter=4, spaceBefore=10)
h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=11, spaceAfter=3, spaceBefore=8)
body = ParagraphStyle('Body2', parent=styles['Normal'], fontSize=9, spaceAfter=3)
code_style = ParagraphStyle('Code', parent=styles['Normal'], fontSize=8, fontName='Courier', spaceAfter=3, leftIndent=20)
check = ParagraphStyle('Check', parent=styles['Normal'], fontSize=9, spaceAfter=2, leftIndent=15)
small = ParagraphStyle('Small', parent=styles['Normal'], fontSize=8, textColor=colors.grey)

TS = TableStyle([
    ('FONTSIZE', (0,0), (-1,-1), 8),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
])

story = []

story.append(Paragraph('Cluster Run Checklist', title_style))
story.append(Paragraph('Human x AI Finance &mdash; Generated 2026-03-13. Print and check off as jobs complete.', small))
story.append(Spacer(1, 8))
story.append(HRFlowable(width='100%', thickness=1, color=colors.black))
story.append(Spacer(1, 6))

# ---- COMPLETED ----
story.append(Paragraph('COMPLETED (no action needed)', h1))

done_data = [
    ['Component', 'Models', 'Count', 'Status'],
    ['VOLARE univariate RV', 'HAR/J/RS, HARQ, Log-HAR, ARFIMA,\nChr-Bolt-S/B, Moirai-2.0-S, Lag-Llama', '150 each\n(50x3h)', 'DONE'],
    ['CAPIRe univariate RV', 'Same 8 models (+ Kronos stale)', '87 each\n(29x3h)', 'DONE'],
    ['Covariance (forex/futures/stocks)', 'Element-HAR, HAR-DRD,\nChr-Bolt-S, Moirai-2.0-S', '3 horizons\neach', 'DONE'],
    ['Context sensitivity (ctx128/256)', 'Chr-Bolt-S/B, Moirai-2.0-S,\nLag-Llama (all 50 tickers)', '600+600\nfiles', 'DONE'],
    ['VOLARE metrics/evaluation', 'DM, MCS, MZ, GR, subsample,\nwindow-512, bootstrap CI', 'All', 'DONE'],
    ['Paper tables (26 .tex files)', 'All current models', '26 files', 'DONE'],
]
ts_done = TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.Color(0.2, 0.5, 0.2)),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.Color(0.95,0.97,0.95), colors.white]),
])
ts_done.add(*TS.getCommands()[0])
ts_done.add(*TS.getCommands()[1])
ts_done.add(*TS.getCommands()[2])
ts_done.add(*TS.getCommands()[3])
t = Table(done_data, colWidths=[1.8*inch, 2.5*inch, 0.9*inch, 0.6*inch])
t.setStyle(ts_done)
story.append(t)
story.append(Spacer(1, 12))

# ---- PHASE 1 ----
story.append(Paragraph('PHASE 1: New TSFM Univariate Forecasts (GPU)', h1))
story.append(Paragraph(
    '4 new models need RV forecasts on all 50 VOLARE tickers x 3 horizons = 150 files each. '
    'Total expected: 600 new CSV files.', body))

p1_data = [
    ['#', 'Job', 'SLURM Script', 'Models', 'Expected Output'],
    ['1a', 'New TSFMs - stocks\n(40 tickers, array job)', 'cluster/run_new_tsfms_rv.slurm\n(stocks variant)', 'TimesFM-2.5\nToto\nSundial\nMoirai-MoE-S', '40x3x4 = 480 CSVs'],
    ['1b', 'New TSFMs - FX+futures\n(10 tickers, single job)', 'cluster/run_new_tsfms_rv.slurm\n(small variant)', 'Same 4 models', '10x3x4 = 120 CSVs'],
]
ts1 = TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.Color(0.8, 0.2, 0.2)),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
])
ts1.add(*TS.getCommands()[0])
ts1.add(*TS.getCommands()[1])
ts1.add(*TS.getCommands()[2])
ts1.add(*TS.getCommands()[3])
t = Table(p1_data, colWidths=[0.3*inch, 1.5*inch, 1.7*inch, 1.2*inch, 1.5*inch])
t.setStyle(ts1)
story.append(t)
story.append(Spacer(1, 4))
story.append(Paragraph(
    'Verify: ls results/volare/forecasts/ | grep -iE "toto|sundial|moirai_moe|timesfm" | wc -l  # expect 600',
    code_style))
story.append(Spacer(1, 8))

# ---- PHASE 2 ----
story.append(Paragraph('PHASE 2: Evaluation Pipeline (after Phase 1)', h1))
story.append(Paragraph('Run locally or on cluster after all 600 new forecast files arrive.', body))

p2_data = [
    ['#', 'Task', 'Command', 'Output'],
    ['2a', 'Re-run VOLARE evaluation', 'python code/run_evaluation_volare.py', 'DM, MCS, aggregate metrics\nwith all 8 TSFMs'],
    ['2b', 'Re-run advanced eval', 'python code/run_advanced_evaluation.py', 'MZ regressions + GR\nfor new models'],
    ['2c', 'Regenerate all tables', 'python code/process_results.py', 'All 26 .tex tables updated'],
    ['2d', 'Regenerate figures', 'python code/generate_figures.py\n+ gen_fig_qlike_boxplot.py\n+ gen_fig_persistence_drivers.py', 'MCS heatmap, QLIKE boxplot,\npersistence scatter'],
    ['2e', 'Bootstrap CIs', 'python code/compute_bootstrap_ci.py', 'CIs for new models'],
    ['2f', 'Subsample metrics', 'python code/compute_subsample_metrics.py', 'Pre/post-COVID for new models'],
]
ts2 = TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.Color(0.9, 0.5, 0.0)),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
])
ts2.add(*TS.getCommands()[0])
ts2.add(*TS.getCommands()[1])
ts2.add(*TS.getCommands()[2])
ts2.add(*TS.getCommands()[3])
t = Table(p2_data, colWidths=[0.3*inch, 1.5*inch, 2.3*inch, 2.1*inch])
t.setStyle(ts2)
story.append(t)
story.append(Spacer(1, 8))

# ---- PHASE 3 ----
story.append(Paragraph('PHASE 3: Covariance - Element-wise + DRD-TSFM (GPU)', h1))
story.append(Paragraph(
    'Run new TSFMs for element-wise covariance AND DRD-TSFM hybrid. '
    'Independent of Phase 1 univariate results.', body))

p3_data = [
    ['#', 'Job', 'SLURM Script', 'Models', 'Notes'],
    ['3a', 'Element-wise cov\n(new TSFMs)', 'Need new script or extend\nrun_cov_foundation.py', 'Toto, Sundial,\nMoirai-MoE-S,\nTimesFM-2.5', 'forex/futures/stocks\nx 3 horizons'],
    ['3b', 'DRD-TSFM hybrid cov\n(ALL TSFMs)', 'cluster/run_cov_drd_tsfm.slurm', 'Chr-Bolt-S/B,\nMoirai-2.0-S,\nToto, Sundial,\nMoirai-MoE-S', '48h wall time.\nStocks = 820 pairs'],
    ['3c', 'Re-run portfolio eval', 'python code/run_portfolio_eval.py', 'All cov models', 'After 3a+3b complete'],
]
ts3 = TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.Color(0.6, 0.2, 0.6)),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
])
ts3.add(*TS.getCommands()[0])
ts3.add(*TS.getCommands()[1])
ts3.add(*TS.getCommands()[2])
ts3.add(*TS.getCommands()[3])
t = Table(p3_data, colWidths=[0.3*inch, 1.3*inch, 1.7*inch, 1.2*inch, 1.7*inch])
t.setStyle(ts3)
story.append(t)
story.append(Spacer(1, 8))

# ---- PHASE 4 ----
story.append(Paragraph('PHASE 4: Context Sensitivity for New TSFMs (GPU)', h1))
story.append(Paragraph(
    'Run ctx=128 and ctx=256 for the 4 new TSFMs. Can run in parallel with Phase 3.', body))

p4_data = [
    ['#', 'Job', 'Details'],
    ['4a', 'ctx sensitivity - stocks (40 tickers, array)', '4 new models x 40 tickers x 3h x 2 ctx = 960 files'],
    ['4b', 'ctx sensitivity - FX+futures (10 tickers)', '4 new models x 10 tickers x 3h x 2 ctx = 240 files'],
    ['4c', 'Regenerate ctx table', 'python code/compute_context_sensitivity.py'],
]
ts4 = TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.Color(0.2, 0.4, 0.7)),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
])
ts4.add(*TS.getCommands()[0])
ts4.add(*TS.getCommands()[1])
ts4.add(*TS.getCommands()[2])
ts4.add(*TS.getCommands()[3])
t = Table(p4_data, colWidths=[0.3*inch, 2.0*inch, 3.9*inch])
t.setStyle(ts4)
story.append(t)
story.append(Spacer(1, 8))

# ---- PHASE 5 ----
story.append(Paragraph('PHASE 5: Paper Updates (after all results)', h1))
p5_items = [
    'Update model count throughout paper ("three TSFMs" to "seven TSFMs")',
    'Add model description paragraphs for TimesFM-2.5, Toto, Sundial, Moirai-MoE in Section 3',
    'Update TSFM summary table (tab:tsfm_summary) with 4 new rows',
    'Update computational cost table with new model timings',
    'Revise results discussion in Sections 5-6 with new model findings',
    'Add DRD-TSFM results to Section 6 (covariance forecasting)',
    'Update conclusion with expanded model coverage',
]
for item in p5_items:
    story.append(Paragraph(f'<bullet>&bull;</bullet> {item}', check))

story.append(Spacer(1, 12))
story.append(HRFlowable(width='100%', thickness=1, color=colors.black))
story.append(Spacer(1, 4))

# ---- KNOWN ISSUES ----
story.append(Paragraph('KNOWN ISSUES', h2))
issues = [
    '<b>TimesFM version mismatch:</b> code/run_foundation.py lists "timesfm-2.0" but factory expects "timesfm-2.5". Fix before running CAPIRe. VOLARE script is correct.',
    '<b>Kronos stale results:</b> 444 forecast files + 294 ctx files exist but are excluded from paper. Do NOT delete.',
    '<b>Moirai cov at h=5:</b> ensure_psd floor=1e-10 causes weight explosions. Monitor DRD-TSFM Moirai results.',
    '<b>Stocks covariance:</b> 820 pairs per model. Longest job. Consider splitting if 48h wall time is insufficient.',
]
for item in issues:
    story.append(Paragraph(f'<bullet>&bull;</bullet> {item}', check))

story.append(Spacer(1, 12))

# ---- DEPENDENCY ORDER ----
story.append(Paragraph('DEPENDENCY ORDER', h2))
story.append(Paragraph('Phase 1 (univariate forecasts)  -->  Phase 2 (evaluation + tables)', code_style))
story.append(Paragraph('Phase 3 (covariance, independent)  -->  Phase 2 step 2c (tables)', code_style))
story.append(Paragraph('Phase 4 (ctx sensitivity, after Phase 1)  -->  Phase 2 step 2c (tables)', code_style))
story.append(Paragraph('Phase 5 (paper edits, after everything)', code_style))

doc.build(story)
print(f'PDF saved to: {os.path.abspath(out)}')
print(f'Size: {os.path.getsize(out)} bytes')
