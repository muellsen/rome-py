"""Audit saved failed path points; no replacement of historical benchmark outputs."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import tempfile
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, LinearConstraint, linprog
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'concomitant-huber-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from concomitant_huber.path_diagnostics import trace_warm_point
from concomitant_huber.solver import _diagnostics, _objective
from pipeline import capture, save, verify

BASE = ROOT/'analysis/outputs/rome-paper-validation'
OUT = ROOT/'analysis/outputs/incomplete-path-audit'
CONFIG = dict(rho=.5, tol=1e-6, continuation_sweeps=4000,
              variants={'MM1': [1, True, 2000], 'MM64': [64, True, 2000],
                        'fixed-scale': [8, False, 2000]},
              dual='SLSQP on n-dimensional perspective dual; feasible rescaling certifies lower bound',
              timing=False, scope='first failed point of each of seven saved paths; no claim about unattempted tails')


def score(z, rho):
    return np.sign(z)*np.minimum(abs(z), rho)


def dual_bound(X, y, lam, rho, a):
    """max y'v/n: |v|<=rho, |X'v/n|<=lam, ||v||²<=2na."""
    n = len(y)
    matrix = X.T/n
    result = minimize(lambda v: -y@v/n, np.zeros(n), jac=lambda v: -y/n,
                      bounds=[(-rho, rho)]*n,
                      constraints=[LinearConstraint(matrix, -lam, lam),
                                   {'type': 'ineq', 'fun': lambda v: 2*n*a-v@v,
                                    'jac': lambda v: -2*v}],
                      method='SLSQP', options={'maxiter': 2000, 'ftol': 1e-12})
    v = result.x
    # A feasible lower bound remains valid even if SLSQP does not converge.
    factor = max(1., max(abs(v))/rho, max(abs(matrix@v))/lam,
                 np.linalg.norm(v)/np.sqrt(2*n*a))
    v = v/factor*(1-1e-12)
    return v, dict(success=bool(result.success), message=result.message,
                   iterations=int(result.nit), value=float(y@v/n),
                   scale_constraint_slack=float(a-v@v/(2*n)),
                   cap_violation=float(max(0., max(abs(v))-rho)),
                   coefficient_constraint_violation=float(max(0., max(abs(matrix@v))-lam)))


def boundary_lp(X, y, lam, rho):
    """Independent closed sigma=0 lasso-LAD problem via HiGHS."""
    n, p = X.shape
    # beta=beta+ - beta- and residual=r+ - r-.
    matrix = np.column_stack([X, -X, np.eye(n), -np.eye(n)])
    cost = np.r_[np.full(2*p, lam), np.full(2*n, rho/n)]
    fit = linprog(cost, A_eq=matrix, b_eq=y, bounds=(0, None), method='highs')
    if not fit.success:
        raise RuntimeError(fit.message)
    dual = fit.eqlin.marginals
    factor = max(1., max(abs(dual))*n/rho, max(abs(X.T@dual))/lam)
    dual = dual/factor*(1-1e-12)
    return fit.x[:p]-fit.x[p:2*p], float(y@dual), float(fit.fun)


def geometry(X, y, beta, sigma, rho, a):
    active = abs(beta)>1e-12
    r = y-X@beta
    central = abs(r)<rho*sigma
    Z = np.column_stack([X[:, active], r/sigma])[central]
    H = Z.T@Z/(len(y)*sigma)
    eigen = np.linalg.eigvalsh(H)
    threshold = max(eigen[-1] if len(eigen) else 0, 1.)*1e-12
    positive = eigen[eigen>threshold]
    rms = np.sqrt(np.mean(X**2, axis=0))
    contraction = {}
    if len(eigen) and sigma > 0:
        # Local linearization: one MM scale step, then exact cyclic coefficients.
        for batch in (1, 8, 64):
            scale_map = np.eye(len(H))
            scale_map[-1] -= sigma/(2*a)*H[-1]
            cycle = np.linalg.matrix_power(scale_map, batch)
            for j in range(len(H)-1):
                if H[j, j] > 0:
                    update = np.eye(len(H))
                    update[j] -= H[j]/H[j, j]
                    cycle = update@cycle
            contraction[str(batch)] = float(max(abs(np.linalg.eigvals(cycle))))
    return dict(local_sweep_spectral_radius=contraction,
                active=int(active.sum()), central=int(central.sum()),
                joint_local_dimension=int(H.shape[0]),
                joint_hessian_rank=int(len(positive)),
                joint_positive_condition=float(positive[-1]/positive[0]) if len(positive) else None,
                column_rms_ratio=float(max(rms)/min(rms)))


def main():
    OUT.mkdir(exist_ok=True)
    verify(json.loads((BASE/'provenance.json').read_text()),
           source_paths=['src/concomitant_huber/solver.py', 'src/concomitant_huber/path.py',
                         'analysis/rome_paper_experiment.py', 'analysis/configs/rome.json'])
    summary = json.loads((BASE/'summary.json').read_text())
    a = summary['a']
    failures = [r for r in summary['results'] if r['method']=='concomitant'
                and r['status'] not in ('converged', 'not_attempted')]
    assert len(failures)==7
    saved = np.load(BASE/'paths.npz')
    rows, arrays, inputs = [], {}, [BASE/'summary.json', BASE/'paths.npz', BASE/'provenance.json']
    fig, axes = plt.subplots(7, 3, figsize=(10, 16), constrained_layout=True)
    for index, failure in enumerate(failures):
        tag = f"{failure['n']}_{failure['p']}_{failure['design']}_{failure['seed']}"
        case = BASE/tag
        X = np.asfortranarray(np.loadtxt(case/'X.tsv'))
        y = np.loadtxt(case/'y.tsv')
        inputs.extend([case/'X.tsv', case/'y.tsv'])
        k = failure['path_index']-1
        beta = saved[tag+'_joint_beta'][k].copy()
        sigma = float(saved[tag+'_joint_sigma'][k])
        lam = float(saved[tag+'_penalties'][k])
        dual, certificate = dual_bound(X, y, lam, .5, a)
        boundary_beta, boundary_value, boundary_primal = boundary_lp(X, y, lam, .5)
        initial_obj = float(_objective(y-X@beta, beta, sigma, .5, a, lam))
        record = dict(case=tag, path_index=k+1, penalty=lam, original_scale=sigma,
                      initial_objective=initial_obj, dual=certificate,
                      boundary_objective_lower_bound=boundary_value, boundary_primal=boundary_primal,
                      boundary_minus_initial=boundary_value-initial_obj,
                      geometry=geometry(X, y, beta, sigma, .5, a), variants={})
        arrays[tag+'_dual']=dual
        arrays[tag+'_boundary_beta']=boundary_beta
        variants = {'MM8': [8, True, 4000], **CONFIG['variants']}
        for label, (batch, estimate, budget) in variants.items():
            trace = trace_warm_point(X, y, beta, sigma, lam, rho=.5, scale_penalty=a,
                                     tol=1e-6, max_iter=budget,
                                     scale_mm_steps=batch, estimate_scale=estimate)
            states = np.unique(np.r_[0, np.arange(1, len(trace['sigma']), 10), len(trace['sigma'])-1])
            residuals = np.array([_diagnostics(X, y-X@trace['beta'][j], trace['beta'][j],
                                               trace['sigma'][j], .5, a, lam, False)
                                  for j in states])
            gaps = trace['objective']-certificate['value']
            arrays[tag+'_'+label+'_beta'] = trace['beta']
            arrays[tag+'_'+label+'_sigma'] = trace['sigma']
            arrays[tag+'_'+label+'_objective'] = trace['objective']
            arrays[tag+'_'+label+'_sampled_sweeps'] = states
            arrays[tag+'_'+label+'_kkt'] = residuals
            final = trace['beta'][-1]
            final_sigma = float(trace['sigma'][-1])
            vb, vs = _diagnostics(X, y-X@final, final, final_sigma, .5, a, lam, False)
            record['variants'][label] = dict(converged=trace['converged'],
                sweeps=len(trace['sigma'])-1, scale=final_sigma, coefficient_kkt=float(vb),
                scale_kkt=float(vs), objective=float(trace['objective'][-1]),
                certified_gap=float(gaps[-1]), initial_certified_gap=float(gaps[0]),
                residual_recompute_difference=float(abs(vb-trace['coefficient_kkt'])),
                maximum_objective_increase=float(max(np.diff(trace['objective']))),
                final_geometry=geometry(X, y, final, final_sigma, .5, a))
            for ax, values in zip(axes[index], [trace['sigma'], np.maximum(gaps, 1e-16)]):
                ax.plot(values, label=label)
            axes[index, 2].semilogy(states, np.maximum(residuals[:, 0], 1e-16), label=label)
        axes[index, 0].set_title(f"{failure['design']}, p={failure['p']}, point {k+1}", fontsize=9)
        axes[index, 0].set_ylabel('Scale')
        axes[index, 1].set(yscale='log', ylabel='Primal minus dual bound')
        axes[index, 2].set_ylabel('Coefficient KKT')
        axes[index, 2].axhline(1e-6, color='k', ls=':', lw=.7)
        for ax in axes[index]:
            ax.set_xlabel('Additional sweeps')
        rows.append(record)
        save(dict(configuration=CONFIG, a=a, results=rows), OUT/'progress.json')
        print(tag, k+1, 'boundary minus initial', round(record['boundary_minus_initial'], 7),
              {m:(v['converged'], v['sweeps'], round(v['coefficient_kkt'], 9))
               for m,v in record['variants'].items()}, flush=True)
    axes[0, 2].legend(fontsize=7)
    fig.savefig(OUT/'diagnostics.pdf')
    fig.savefig(OUT/'diagnostics.png', dpi=150)
    plt.close(fig)
    save(dict(configuration=CONFIG, a=a, results=rows), OUT/'summary.json')
    np.savez_compressed(OUT/'traces.npz', **arrays)
    lines = ['| Design | p | Failed point | Scale | Initial gap bound | MM8 extra sweeps | MM8 final coefficient KKT |',
             '|:--|--:|--:|--:|--:|--:|--:|']
    for r in rows:
        v = r['variants']['MM8']
        lines.append(f"| {r['case'].split('_')[2]} | {r['case'].split('_')[1]} | {r['path_index']} | {r['original_scale']:.3g} | {v['initial_certified_gap']:.2e} | {v['sweeps']} | {v['coefficient_kkt']:.2e} |")
    (OUT/'results.qmd').write_text('\n'.join(lines)+'\n')
    outputs = [OUT/n for n in ('summary.json', 'traces.npz', 'results.qmd', 'diagnostics.pdf', 'diagnostics.png')]
    save(capture(outputs, inputs=inputs, configuration=CONFIG, seed=71), OUT/'provenance.json')


if __name__=='__main__':
    if sys.argv[1:]==['--record-report']:
        manifest = OUT/'provenance.json'
        verify(json.loads(manifest.read_text()), source_paths=[
            'src/concomitant_huber/solver.py', 'src/concomitant_huber/path.py',
            'src/concomitant_huber/path_diagnostics.py', 'analysis/audit_incomplete_paths.py'])
        document = ROOT/'_build/analysis/incomplete-path-audit.html'
        source = ROOT/'analysis/incomplete-path-audit.qmd'
        if not document.exists() or document.stat().st_mtime < source.stat().st_mtime:
            raise RuntimeError('Render the current audit report first')
        save(capture([document], inputs=[manifest, source],
             configuration={'stage': 'incomplete-path-audit-report'},
             command=['quarto', 'render', 'analysis/incomplete-path-audit.qmd', '--to', 'html']),
             OUT/'report.provenance.json')
        print('Audit report provenance recorded.')
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('Usage: audit_incomplete_paths.py [--record-report]')
