"""Regenerate the article figures into figures/.  Usage: python -m src.make_figures"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from .simulate import simulate, naive_gap, adjusted, fuzzy_rd, TRUE_EFFECT, CUTOFF

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
BLUE, RED = '#4a86c8', '#d94141'


def fig1(df, path='figures/fig1_estimates.png'):
    adj, orc, r = adjusted(df), adjusted(df, oracle=True), fuzzy_rd(df, bw=10)
    labels = ['Naive\n(adopters vs\nnon-adopters)', 'Regression\n(seats, tenure)',
              'Fuzzy RD / 2SLS\n(seat threshold)', 'Oracle\n(with engagement)', 'True effect\n(simulation)']
    vals = [100 * naive_gap(df), 100 * adj.params['adopted'], 100 * r['late'],
            100 * orc.params['adopted'], 100 * TRUE_EFFECT]
    cols = [RED, RED, BLUE, '#444444', '#222222']
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(labels, vals, color=cols, width=0.6)
    ax.errorbar([1, 2], [vals[1], vals[2]], yerr=[196 * adj.bse['adopted'], 196 * r['se']],
                fmt='none', ecolor='black', capsize=5, lw=1.2)
    for i, v in enumerate(vals):
        ax.text(i, v + (8.5 if i == 2 else 0.8), f'{v:+.1f} pp', ha='center', fontweight='bold')
    ax.axhline(100 * TRUE_EFFECT, ls='--', color='grey', lw=1)
    ax.set_ylabel('Estimated retention lift from adoption (pp)'); ax.set_ylim(0, 20)
    ax.set_title('Same data, four estimates of the AI assistant effect', fontweight='bold')
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def fig2(df, path='figures/fig2_rd.png'):
    w = df[(df.seats >= 13) & (df.seats <= 37)]
    g = w.groupby('seats').agg(adopt=('adopted', 'mean'), ret=('retained', 'mean'), n=('seats', 'size')).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, col, ycol, title, yl in [
        (axes[0], 'adopt', 'adopted', 'First stage: adoption jumps at 25 seats', 'Adoption rate'),
        (axes[1], 'ret', 'retained', 'Reduced form: retention jumps at 25 seats', '6-month retention rate')]:
        ax.scatter(g.seats, g[col], s=g.n / 12, color=BLUE, alpha=.8)
        for mask, xs in [(w.seats < CUTOFF, np.array([13, 24.999])), (w.seats >= CUTOFF, np.array([25, 37]))]:
            m = smf.ols(f'{ycol} ~ seats', data=w[mask]).fit()
            ax.plot(xs, m.params['Intercept'] + m.params['seats'] * xs, color=RED, lw=2)
        ax.axvline(24.5, ls='--', color='grey')
        ax.set_xlabel('Seats on the account'); ax.set_ylabel(yl)
        ax.set_title(title, fontweight='bold', fontsize=11)
    axes[0].set_ylim(-0.03, 0.5); axes[1].set_ylim(0.50, 0.62)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


if __name__ == '__main__':
    df = simulate()
    fig1(df); fig2(df)
    print('wrote figures/fig1_estimates.png and figures/fig2_rd.png')
