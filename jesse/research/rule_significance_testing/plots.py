"""
Plotting utility for rule_significance_test() results.
"""

import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe in scripts and notebooks
from matplotlib import pyplot as plt


def plot_significance_test(result: dict, charts_folder: str = None) -> None:
    """
    Visualise the sampling distribution produced by rule_significance_test().

    The histogram shows where the observed mean sits relative to the
    distribution of simulated means under H0.  The shaded region represents
    the fraction of simulations that equalled or exceeded the observed mean
    (i.e. the p-value).

    Parameters
    ----------
    result        : dict returned by rule_significance_test()
    charts_folder : directory to save the PNG; defaults to ./charts/
    """
    sim_means = result['simulated_means']
    observed_mean = result['observed_mean']
    p_value = result['p_value']

    annualized = result['annualized_return']
    n_obs = result['n_observations']
    n_sim = result['n_simulations']

    fig, ax = plt.subplots(figsize=(10, 5))

    # ---- Histogram of simulated means ----
    n_bins = min(50, max(20, n_sim // 10))
    counts, bin_edges, patches = ax.hist(
        sim_means, bins=n_bins,
        color='steelblue', edgecolor='white', linewidth=0.4, alpha=0.85,
        label='Simulated means (H₀)',
    )

    # ---- Shade the region that equals or exceeds the observed mean ----
    # This shaded area IS the p-value visually.
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        if left_edge >= observed_mean:
            patch.set_facecolor('tomato')
            patch.set_alpha(0.9)

    # ---- Vertical line at the observed mean ----
    ax.axvline(
        observed_mean, color='darkred', linewidth=1.8, linestyle='--',
        label=f'Observed mean = {observed_mean:.6f}',
    )

    # ---- Annotations ----
    method_label = "Bootstrap Significance Test"
    significance = ''
    if p_value < 0.01:
        significance = '  ★★  highly significant (p < 0.01)'
    elif p_value < 0.05:
        significance = '  ★  significant (p < 0.05)'
    else:
        significance = '  not significant (p ≥ 0.05)'

    info_text = (
        f'p-value = {p_value:.4f}{significance}\n'
        f'Annualised return = {annualized * 100:.4f} %\n'
        f'Observations = {n_obs} bars   |   Simulations = {n_sim}'
    )
    ax.text(
        0.02, 0.97, info_text,
        transform=ax.transAxes, verticalalignment='top',
        fontsize=9, family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8),
    )

    ax.set_title('Rule Significance Test — Bootstrap', fontsize=12, fontweight='bold')
    ax.set_xlabel('Mean bar-level log return')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=9)
    plt.tight_layout()

    # ---- Save ----
    if charts_folder is None:
        charts_folder = os.path.abspath('charts')
    os.makedirs(charts_folder, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'rule_significance_bootstrap_{timestamp}.png'
    path = os.path.join(charts_folder, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved significance test chart to: {path}')
