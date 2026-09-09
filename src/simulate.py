"""
Simulation and estimators for "Your AI Adoption Lift Is a Selection Effect".

Usage:
    python -m src.simulate            # prints every number cited in the article
    from src.simulate import simulate, fuzzy_rd
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

TRUE_EFFECT = 0.04   # +4 pp 6-month retention from adopting the AI assistant
CUTOFF = 25          # assistant only available to accounts with >= 25 seats


def simulate(n=40_000, seed=2026, true_effect=TRUE_EFFECT, cutoff=CUTOFF):
    """Synthetic B2B accounts with an opt-in AI feature gated at `cutoff` seats.

    `engagement` is latent: it drives both adoption and retention and is the
    confounder the analyst never observes.
    """
    rng = np.random.default_rng(seed)
    engagement = np.clip(rng.normal(0, 1, n), -2.5, 2.5)
    seats = np.clip(
        np.round(np.exp(rng.normal(3.2, 0.6, n) + 0.10 * engagement)), 3, 300
    ).astype(int)
    tenure = rng.uniform(6, 36, n)
    eligible = (seats >= cutoff).astype(int)

    p_adopt = 1 / (1 + np.exp(-(-0.6 + 1.4 * engagement)))
    adopted = ((eligible == 1) & (rng.uniform(size=n) < p_adopt)).astype(int)

    p_retain = (0.55
                + 0.10 * engagement
                + 0.03 * (np.log(seats) - np.log(cutoff))
                + true_effect * adopted)
    retained = (rng.uniform(size=n) < p_retain).astype(int)

    return pd.DataFrame(dict(seats=seats, tenure=tenure, eligible=eligible,
                             adopted=adopted, retained=retained,
                             engagement=engagement))


def naive_gap(df):
    m = df.groupby('adopted')['retained'].mean()
    return m[1] - m[0]


def adjusted(df, oracle=False):
    """OLS on eligible accounts. `oracle=True` adds the unobserved confounder."""
    elig = df[df.eligible == 1]
    f = 'retained ~ adopted + np.log(seats) + tenure' + (' + engagement' if oracle else '')
    return smf.ols(f, data=elig).fit(cov_type='HC3')


def fuzzy_rd(df, cutoff=CUTOFF, bw=10):
    """Fuzzy RD at the seat threshold, estimated by 2SLS.

    Eligibility instruments adoption. Local linear trend on each side.
    Returns first stage, reduced form, and the 2SLS adoption effect with a
    heteroskedasticity-robust standard error.
    """
    w = df[(df.seats >= cutoff - bw) & (df.seats < cutoff + bw)].copy()
    w['x'] = w.seats - cutoff
    w['above'] = (w.x >= 0).astype(int)
    w['above_x'] = w.above * w.x

    first = smf.ols('adopted ~ above * x', data=w).fit(cov_type='HC1')
    reduced = smf.ols('retained ~ above * x', data=w).fit(cov_type='HC1')
    model = IV2SLS.from_formula(
        'retained ~ 1 + x + above_x + [adopted ~ above]', data=w
    ).fit(cov_type='robust')

    return dict(n=len(w),
                first_stage=first.params['above'],
                reduced_form=reduced.params['above'],
                reduced_form_ci=tuple(reduced.conf_int().loc['above']),
                late=model.params['adopted'],
                se=model.std_errors['adopted'],
                ci=tuple(model.conf_int().loc['adopted']),
                window=w, model=model)


def placebo_jumps(df, cutoffs=(15, 35, 45, 55), bw=10):
    """Reduced-form jump at fake cutoffs. Windows must not cross the real cutoff."""
    out = []
    for pc in cutoffs:
        wp = df[(df.seats >= pc - bw) & (df.seats < pc + bw)].copy()
        wp['x'] = wp.seats - pc
        wp['above'] = (wp.x >= 0).astype(int)
        m = smf.ols('retained ~ above * x', data=wp).fit(cov_type='HC1')
        out.append(dict(cutoff=pc, n=len(wp), jump=m.params['above'], se=m.bse['above']))
    return pd.DataFrame(out)


def covariate_smoothness(df, covariates=('tenure', 'engagement'), cutoff=CUTOFF, bw=10):
    w = df[(df.seats >= cutoff - bw) & (df.seats < cutoff + bw)].copy()
    w['x'] = w.seats - cutoff
    w['above'] = (w.x >= 0).astype(int)
    out = []
    for c in covariates:
        m = smf.ols(f'{c} ~ above * x', data=w).fit(cov_type='HC1')
        out.append(dict(covariate=c, jump=m.params['above'], se=m.bse['above']))
    return pd.DataFrame(out)


if __name__ == '__main__':
    df = simulate()
    print(f'Naive adopter gap:                 {naive_gap(df):+.3f}')
    print(f'Naive gap, eligible only:          {naive_gap(df[df.eligible == 1]):+.3f}')
    a = adjusted(df)
    print(f'Regression-adjusted (observables): {a.params["adopted"]:+.3f}  (±{1.96 * a.bse["adopted"]:.3f})')
    print(f'Oracle (with engagement):          {adjusted(df, oracle=True).params["adopted"]:+.3f}')
    print()
    for bw in (5, 10, 15, 20):
        r = fuzzy_rd(df, bw=bw)
        print(f'RD bw=±{bw:2d}  n={r["n"]:6,}  first={r["first_stage"]:.3f}  '
              f'reduced={r["reduced_form"]:+.4f}  2SLS={r["late"]:+.3f}  '
              f'se={r["se"]:.3f}  CI=[{r["ci"][0]:+.3f}, {r["ci"][1]:+.3f}]')
    r = fuzzy_rd(df, bw=10)
    print(f'\nReduced form at ±10: {r["reduced_form"]:+.4f}  CI=[{r["reduced_form_ci"][0]:+.3f}, {r["reduced_form_ci"][1]:+.3f}]')
    w = r['window']
    m_cl = IV2SLS.from_formula('retained ~ 1 + x + above_x + [adopted ~ above]', data=w).fit(
        cov_type='clustered', clusters=w.seats)
    print(f'2SLS SE, robust: {r["se"]:.3f}   clustered by seat: {m_cl.std_errors["adopted"]:.3f}')
    print('\nPlacebo cutoffs:')
    print(placebo_jumps(df).round(4).to_string(index=False))
    print('\nCovariate smoothness at cutoff:')
    print(covariate_smoothness(df).round(3).to_string(index=False))
    print('\nDensity around cutoff:')
    print(df.seats.value_counts().sort_index().loc[20:30].to_dict())
