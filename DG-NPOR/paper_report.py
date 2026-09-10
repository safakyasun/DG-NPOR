"""Publication figures and concise LaTeX from four aligned frozen scores."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from frozen_inputs import save_json

METHODS = ('DG-NPOR', 'ParticleNet (JetSet)', 'GN2v01', 'DL1dv01')
COLORS = {'DG-NPOR': '#0072B2', 'ParticleNet (JetSet)': '#D55E00', 'GN2v01': '#009E73', 'DL1dv01': '#8B65A8'}
LINES = {'DG-NPOR': '-', 'ParticleNet (JetSet)': '--', 'GN2v01': '-.', 'DL1dv01': ':'}


def style():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
        'lines.linewidth': 1.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.facecolor': 'white', 'figure.facecolor': 'white'})


def save_plot(fig, path):
    fig.savefig(Path(path).with_suffix('.pdf'), bbox_inches='tight', pad_inches=.03)
    fig.savefig(Path(path).with_suffix('.png'), dpi=500, bbox_inches='tight', pad_inches=.03)
    plt.close(fig)


def latex_escape(text):
    return str(text).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def number(value, digits=1):
    if not np.isfinite(value):
        return r'\infty'
    return ('%.*f' % (digits, value))


def rejection_latex(row, token):
    value, low, high = (float(row[token + key]) for key in ('', '_lower68', '_upper68'))
    if not np.isfinite(value):
        return r'$\infty\;[>%s]$' % number(low)
    plus = number(high - value) if np.isfinite(high) else r'\infty'
    return r'$%s^{+%s}_{-%s}$' % (number(value), plus, number(value - low))


def make_table(summary, wp):
    rows = []
    for method in METHODS:
        metric = summary.set_index('method').loc[method]
        row = {'method': method, 'auc': metric.auc, 'auc_lower95': metric.auc_lower_95, 'auc_upper95': metric.auc_upper_95,
               'comparison': 'frozen local model' if method in METHODS[:2] else 'pretrained same-jet reference'}
        for eff in (.6, .7, .77, .85):
            w = wp[(wp.method == method) & np.isclose(wp.target_b_efficiency, eff)].iloc[0]
            token = 'R%d' % round(100 * eff)
            row.update({token: w.light_rejection, token + '_lower68': w.light_rejection_lower_68,
                        token + '_upper68': w.light_rejection_upper_68,
                        token + '_mistags': int(w.light_mistagged_jets), token + '_light_jets': int(w.total_light_jets),
                        token + '_achieved_b_efficiency': w.achieved_b_efficiency,
                        token + '_zero_observed_mistags': bool(w.zero_observed_light_mistags_censored)})
        rows.append(row)
    return pd.DataFrame(rows)


def table_latex(table):
    lines = [r'\begin{tabular}{lccc}', r'\toprule',
             r'Method & AUC [95\% CI] & $R_{\mathrm{light}}$ at $\epsilon_b=0.70$ & $R_{\mathrm{light}}$ at $\epsilon_b=0.77$ \\', r'\midrule']
    for i, row in enumerate(table.to_dict('records')):
        if i == 2:
            lines.append(r'\midrule')
        auc = r'$%.4f\;[%.4f,\,%.4f]$' % (row['auc'], row['auc_lower95'], row['auc_upper95'])
        lines.append(' & '.join([latex_escape(row['method']), auc, rejection_latex(row, 'R70'), rejection_latex(row, 'R77')]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '']
    return '\n'.join(lines)


def write_text(path, value):
    Path(path).write_text(value, encoding='utf-8')


def make_curves(y, scores, output):
    from jetset_litcomp.metrics import roc_points
    style()
    fig, ax = plt.subplots(figsize=(3.45, 2.65), layout='constrained')
    rocfig, rocax = plt.subplots(figsize=(3.45, 2.65), layout='constrained')
    for method in METHODS:
        curve = roc_points(y, scores[method])
        curve.to_csv(output / 'statistics' / ('ROC_' + method.split()[0].replace('-', '_') + '.csv.gz'), index=False)
        # Infinite rejection from zero observed background is not drawn as a finite measurement.
        keep = np.isfinite(curve.light_rejection) & (curve.b_efficiency >= .45)
        visible = curve.loc[keep]
        if len(visible):
            ax.plot(visible.b_efficiency, visible.light_rejection, color=COLORS[method], linestyle=LINES[method], label=method)
        rocax.plot(curve.light_efficiency, curve.b_efficiency, color=COLORS[method], linestyle=LINES[method], label=method)
    ax.set(xlim=(.5, .95), yscale='log', xlabel=r'$b$-jet efficiency $\epsilon_b$', ylabel=r'Light-jet rejection $1/\epsilon_{\mathrm{light}}$')
    ax.grid(axis='y', which='major', color='#dddddd', linewidth=.5)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), ncol=2, frameon=False,
              handlelength=2.0, columnspacing=.9, fontsize=7.5, borderaxespad=0)
    rocax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel=r'Light-jet efficiency $\epsilon_{\mathrm{light}}$', ylabel=r'$b$-jet efficiency $\epsilon_b$')
    rocax.plot([0, 1], [0, 1], color='#aaaaaa', linewidth=.7, zorder=0)
    rocax.legend(loc='lower right', frameon=False, handlelength=2.6)
    save_plot(fig, output / 'figures/FIGURE_1_REJECTION')
    save_plot(rocfig, output / 'figures/FIGURE_S1_ROC')


def orbital_figure(y, state, selected, output):
    style()
    width = state.shape[1]
    ncols = min(3, width)
    nrows = int(np.ceil(width / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 2.05 * nrows), squeeze=False, layout='constrained')
    for j, ax in enumerate(axes.ravel()):
        if j >= width:
            ax.set_visible(False)
            continue
        for label, color, name, line in [(0, '#0072B2', 'Light jets', '-'), (1, '#D55E00', r'$b$ jets', '--')]:
            values = state[y == label, j]
            ax.hist(values, bins=np.linspace(-1.0001, 1.0001, 51), density=True, histtype='step',
                    color=color, linestyle=line, linewidth=1.5, label=name)
        ax.set(xlim=(-1.03, 1.03), xlabel=r'Normalized coordinate $z_%d$' % (j + 1), title=r'(%s) Orbital $\phi_{%d}$' % (chr(97 + j), selected[j]))
        if j % ncols == 0:
            ax.set_ylabel('Class-normalized density')
        if j == 0:
            ax.legend(frameon=False, fontsize=8)
    save_plot(fig, output / 'figures/FIGURE_2_ORBITALS')


def kinematic_table(predictions, scores, wp):
    rows = []
    y = predictions.y_true_light0_b1.to_numpy(int)
    definitions = [('pt_GeV', 20, 40), ('pt_GeV', 40, 70), ('pt_GeV', 70, 120), ('pt_GeV', 120, 250), ('pt_GeV', 250, np.inf),
                   ('abs_eta', 0, .6), ('abs_eta', .6, 1.2), ('abs_eta', 1.2, 1.8), ('abs_eta', 1.8, 2.5)]
    for variable, low, high in definitions:
        values = predictions.pt_GeV.to_numpy() if variable == 'pt_GeV' else np.abs(predictions.eta.to_numpy())
        mask = (values >= low) & (values < high)
        labels = y[mask]
        for method in METHODS:
            p = scores[method][mask]
            threshold = wp[(wp.method == method) & np.isclose(wp.target_b_efficiency, .7)].iloc[0].score_threshold
            n_b, n_l = int((labels == 1).sum()), int((labels == 0).sum())
            b_eff = float(np.mean(p[labels == 1] >= threshold)) if n_b else None
            l_eff = float(np.mean(p[labels == 0] >= threshold)) if n_l else None
            rows.append({'method': method, 'variable': variable, 'lower': low, 'upper': high, 'b_jets': n_b, 'light_jets': n_l,
                         'auc': roc_auc_score(labels, p) if n_b and n_l else None,
                         'b_efficiency_at_global70_threshold': b_eff, 'light_efficiency_at_global70_threshold': l_eff})
    return pd.DataFrame(rows)


def write_paper_results(predictions, state, metadata, output, repetitions=2000):
    from jetset_litcomp.metrics import classification_summary, event_cluster_auc_bootstrap, working_point
    output = Path(output)
    for folder in ('paper', 'statistics', 'figures', 'overleaf'):
        (output / folder).mkdir(parents=True, exist_ok=True)
    paper = output / 'paper'
    y = predictions.y_true_light0_b1.to_numpy(int)
    events = predictions.event_number.to_numpy()
    scores = {name: predictions[name].to_numpy(float) for name in METHODS}
    if state.shape != (len(y), metadata['K_DG']) or not np.isfinite(state).all() or np.any(np.linalg.norm(state, axis=1) > 1 + 1e-6):
        raise ValueError('Invalid selected-state figure input.')
    print('Paired whole-event AUC bootstrap: %d repetitions' % repetitions, flush=True)
    boot = event_cluster_auc_bootstrap(y, scores, events, repetitions=repetitions, random_state=2407, reference_method='DG-NPOR')
    boot.summary.to_csv(output / 'statistics/AUC_EVENT_BOOTSTRAP.csv', index=False)
    # Interval estimates are retained; the old bootstrap tail fraction is not presented as a hypothesis-test p-value.
    paired = boot.paired_differences.drop(columns=['paired_bootstrap_two_sided_p'], errors='ignore')
    paired.to_csv(output / 'statistics/PAIRED_AUC_DIFFERENCES.csv', index=False)
    boot.replicate_values.to_csv(output / 'statistics/AUC_BOOTSTRAP_REPLICATES.csv.gz', index=False, compression='gzip')
    metrics, working = [], []
    for name, p in scores.items():
        metrics.append({'method': name, **classification_summary(y, p, probability=name in METHODS[:2])})
        for efficiency in (.6, .7, .77, .85):
            working.append({'method': name, **working_point(y, p, efficiency, confidence=.68)})
    wp = pd.DataFrame(working)
    wp.to_csv(output / 'statistics/WORKING_POINTS_60_70_77_85.csv', index=False)
    pd.DataFrame(metrics).to_csv(output / 'statistics/ALL_METRICS.csv', index=False)
    table = make_table(boot.summary, wp)
    table.to_csv(paper / 'TABLE_1_COMPARISON.csv', index=False)
    write_text(paper / 'TABLE_1_COMPARISON.tex', table_latex(table))
    kinematic_table(predictions, scores, wp).to_csv(output / 'statistics/KINEMATIC_SLICES.csv', index=False)
    make_curves(y, scores, output)
    orbital_figure(y, state, metadata['selected_orbitals_one_based'], output)
    selected = np.asarray(metadata['selected_orbitals_one_based'])
    class_summary = []
    for j in range(state.shape[1]):
        for label in (0, 1):
            a = state[y == label, j]
            class_summary.append({'orbital': int(selected[j]), 'coordinate': j + 1, 'class_light0_b1': label, 'jets': len(a),
                                  'mean': np.mean(a), 'std': np.std(a), 'q16': np.quantile(a, .16), 'median': np.median(a), 'q84': np.quantile(a, .84)})
    pd.DataFrame(class_summary).to_csv(output / 'statistics/ORBITAL_DISTRIBUTIONS.csv', index=False)
    count_rows = [{'stage': 'DG encoder training', 'jets': metadata['encoder_train_jets']},
                  {'stage': 'DG metric training', 'jets': metadata['metric_train_jets']},
                  {'stage': 'DG landmarks', 'jets': metadata['landmark_jets']},
                  {'stage': 'DG final PSD training', 'jets': metadata['PSD_train_jets']},
                  {'stage': 'ParticleNet training', 'jets': metadata['particle_net']['frozen_roles']['train']},
                  {'stage': 'Common final test', 'jets': len(y)}]
    pd.DataFrame(count_rows).to_csv(output / 'statistics/TRAINING_SAMPLE_COUNTS.csv', index=False)
    t = table.set_index('method')
    dg, pn, gn, dl = [t.loc[name] for name in METHODS]
    delta = paired.set_index('method').loc['ParticleNet (JetSet)']
    k, d = metadata['K_DG'], metadata['encoder_dimension']
    def in_text_rejection(value):
        return ('$%s$' % number(value)) if np.isfinite(value) else 'no observed light-jet mistags'
    results = (
        'The frozen models are evaluated on %s jets (%s $b$ and %s light jets) from %s held-out events in the ATLAS JetSet $t\\bar{t}$ simulation. '
        'Table~\\ref{tab:jetset_comparison} reports a common evaluation population for all four methods. '
        'DG-NPOR achieves an AUC of %.4f and a light-jet rejection of %s at $\\epsilon_b=0.70$. '
        'The local ParticleNet implementation obtains an AUC of %.4f and a rejection of %s. '
        'The paired AUC difference, ParticleNet minus DG-NPOR, is %.4f, with a 95\\%% event-bootstrap interval of [%.4f, %.4f].\n\n'
        % (f'{len(y):,}', f'{int(y.sum()):,}', f'{int((y == 0).sum()):,}', f'{len(np.unique(events)):,}',
           dg.auc, in_text_rejection(dg.R70), pn.auc, in_text_rejection(pn.R70),
           delta.auc_difference_method_minus_reference, delta.difference_lower_95, delta.difference_upper_95)
    )
    if not np.isfinite(dg.R70) or not np.isfinite(pn.R70):
        # Avoid calling an unobserved background tail a measured infinite performance.
        results = results.replace('a light-jet rejection of no observed light-jet mistags', 'zero observed light-jet mistags').replace('a rejection of no observed light-jet mistags', 'zero observed light-jet mistags')
    results += (
        'The automatically selected DG-NPOR state contains %d coordinates, obtained from the %d-dimensional encoder representation. '
        'The candidate-bank and derivative-selection stages yield $K_{\\mathrm{PF}}=%d$ and $K_{\\mathrm{DG}}=%d$, respectively. '
        'This describes the final representation width; it does not establish a corresponding reduction in total model parameters or runtime. '
        'Figure~\\ref{fig:jetset_rejection} shows the rejection curves. '
        'The stored GN2v01 and DL1dv01 discriminants provide same-jet pretrained references, with AUCs of %.4f and %.4f. '
        'Their inputs and training budgets are not matched to the locally trained models.\n'
        % (k, d, metadata['K_PF'], k, gn.auc, dl.auc)
    )
    write_text(paper / 'RESULTS_TEXT.tex', results)
    future = ('Future work will examine the stability of automatic orbital selection across training samples and improve the transfer of the learned class separation to the spectral representation. '
              'Further studies will compare representations under controlled training budgets and quantify variability across independently trained models.\n')
    write_text(paper / 'FUTURE_WORK.tex', future)
    macros = {'PorK': str(k), 'PorCandidateK': str(metadata['K_PF']), 'PorEncoderDimension': str(d),
              'TestJets': str(len(y)), 'TestEvents': str(len(np.unique(events))),
              'PorAUC': '%.4f' % dg.auc, 'ParticleNetAUC': '%.4f' % pn.auc,
              'PorRSeventy': r'\ensuremath{%s}' % number(dg.R70), 'ParticleNetRSeventy': r'\ensuremath{%s}' % number(pn.R70)}
    write_text(paper / 'PAPER_NUMBERS.tex', '\n'.join(r'\newcommand{\%s}{%s}' % (name, value) for name, value in macros.items()) + '\n')
    save_json(paper / 'PAPER_NUMBERS.json', {'summary': table.to_dict('records'), 'K_DG': k, 'encoder_dimension': d})
    caption = ('Common frozen-test comparison. AUC intervals use %d whole-event bootstrap resamples (95\\%%). '
        'Rejection errors are conditional 68\\%% Clopper--Pearson count intervals; empirical $b$-score quantiles define the operating points. '
        'ParticleNet denotes the existing JetSet adaptation. GN2v01 and DL1dv01 are pretrained references with different input/training budgets. '
        'Infinite rejection denotes zero observed mistags; a finite lower bound is shown in brackets.' % repetitions)
    table_include = ('\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{5pt}\n'
                     '\\input{TABLE_1_COMPARISON}\n\\caption{' + caption + '}\n\\label{tab:jetset_comparison}\n\\end{table*}\n')
    figure_include = ('\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\columnwidth]{FIGURE_1_REJECTION.pdf}\n'
                      '\\caption{Light-jet rejection as a function of $b$-jet efficiency on the common frozen test sample. All models and selections are fixed. Points with zero observed light-jet acceptance are omitted from the curves.}\n'
                      '\\label{fig:jetset_rejection}\n\\end{figure}\n')
    orbital_include = ('\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{FIGURE_2_ORBITALS.pdf}\n'
                       '\\caption{Class-normalized distributions of the selected orbital coordinates. Orbital functions, their ordering and signs are fixed by the trained reference graph; these are descriptive coordinates, not individual physical observables.}\n'
                       '\\label{fig:jetset_orbitals}\n\\end{figure*}\n')
    write_text(paper / 'RESULTS_MAIN.tex', '\\section{Results}\n\\input{RESULTS_TEXT}\n' + table_include + figure_include)
    write_text(paper / 'OPTIONAL_ORBITAL_FIGURE.tex', orbital_include)
    note = (
        '# Comparison and manuscript notes\n\n'
        '- The dataset is [Monte Carlo ttbar simulation](https://opendata.cern.ch/record/93940), not recorded collision events.\n'
        '- ParticleNet is the existing local JetSet adaptation. Its EdgeConv uses max aggregation and includes self-neighbors; batch normalization acts before invalid-edge masking. The [published ParticleNet](https://arxiv.org/html/1902.08570v3) uses mean aggregation. Describe the local implementation explicitly. No architecture change or retraining is claimed for reused results.\n'
        '- Both local models use the same nominal 19 reconstructed track variables, 4 reconstructed context variables and track limit, with separately fitted preprocessing. Their training allocations differ; see TRAINING_SAMPLE_COUNTS.csv.\n'
        '- GN2v01/DL1dv01 are stored same-jet reference discriminants. Their NLL and 0.5-threshold accuracy are not computed. Cite the JetSet record and [ATLAS GN2 paper](https://arxiv.org/abs/2505.19689) alongside the relevant tagger references already in your bibliography.\n'
        '- K is read from the frozen model, not forced to three. Candidate-bank selection and derivative selection are distinct stages.\n'
        '- The final test is event-disjoint from development roles. The original metric map uses a jet-level internal split; its event overlap is %d in this run. Do not state that every internal early-stop split is event-disjoint.\n'
        '- AUC intervals are conditional on this fitted run. Rejection intervals condition on jet counts and the resolved threshold, and do not include retraining variability or full event-cluster uncertainty.\n'
        '- Model-selection choices are not optimized using these output tables. Re-exporting fixed scores does not make a previously inspected test newly untouched.\n'
        '- Bootstrap median/percentile results are not extrapolated beyond observed score support. Zero-mistag rejection is censored, not a precisely measured infinite value.\n'
        '- No future-work changes, synthetic audit scores, alternative readouts, or WP-development scores enter the main table.\n'
        % metadata['metric_inner_overlap_events']
    )
    write_text(paper / 'COMPARISON_NOTES.md', note)
    overleaf = output / 'overleaf'
    for path in paper.glob('*.tex'):
        shutil.copy2(path, overleaf / path.name)
    for path in (output / 'figures').glob('*.pdf'):
        shutil.copy2(path, overleaf / path.name)
    preview = (r'''\documentclass[10pt]{article}
\usepackage[a4paper,margin=0.65in]{geometry}
\usepackage{amsmath,graphicx,booktabs}
\usepackage[T1]{fontenc}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\begin{document}
{\large\bfseries DG-NPOR: Results material}\par
\input{RESULTS_PREVIEW_TEXT}
\begin{center}
{\small\setlength{\tabcolsep}{5pt}\input{TABLE_1_COMPARISON}}
\end{center}
{\footnotesize ''' + caption + r'''}\par
\begin{center}
\includegraphics[width=0.47\textwidth]{FIGURE_1_REJECTION.pdf}\hfill
\includegraphics[width=0.47\textwidth]{FIGURE_S1_ROC.pdf}\par
\includegraphics[width=0.94\textwidth]{FIGURE_2_ORBITALS.pdf}
\end{center}
\end{document}
''')
    # The standalone preview intentionally has no floating labels. Production fragments retain labels.
    preview_text = results.replace(r'Table~\ref{tab:jetset_comparison}', 'The table below').replace(r'Figure~\ref{fig:jetset_rejection}', 'The first plot below')
    write_text(overleaf / 'RESULTS_PREVIEW_TEXT.tex', preview_text)
    write_text(overleaf / 'RESULTS_PREVIEW.tex', preview)
    if shutil.which('pdflatex'):
        result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'RESULTS_PREVIEW.tex'], cwd=overleaf, capture_output=True, text=True)
        if result.returncode:
            write_text(overleaf / 'LATEX_BUILD_ERROR.txt', result.stdout[-12000:] + result.stderr)
            raise RuntimeError('LaTeX preview compilation failed; see LATEX_BUILD_ERROR.txt.')
        shutil.copy2(overleaf / 'RESULTS_PREVIEW.pdf', paper / 'RESULTS_PREVIEW.pdf')
    else:
        write_text(overleaf / 'COMPILE_ON_OVERLEAF.txt', 'Upload this folder and compile RESULTS_PREVIEW.tex. For the manuscript, input RESULTS_MAIN.tex and optionally OPTIONAL_ORBITAL_FIGURE.tex.\n')
    with __import__('zipfile').ZipFile(output / 'OVERLEAF_RESULTS.zip', 'w', __import__('zipfile').ZIP_DEFLATED) as archive:
        for path in sorted(overleaf.iterdir()):
            if path.suffix in ('.tex', '.pdf', '.txt'):
                archive.write(path, path.name)
    print(table[['method', 'auc', 'R70', 'R70_mistags', 'R77', 'R85']].to_string(index=False), flush=True)
    return table
