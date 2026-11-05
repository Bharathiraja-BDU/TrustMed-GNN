"""TrustMed-GNN Phase 3: exploratory, calibration-selected abstention.

Input: Phase 2B predictions (NOT an independent test set). No model loading.
Thresholds are selected using calibration rows only and evaluated on the
previously separated evaluation rows. Reuses the Phase 2B split; results remain
exploratory because the original validation set influenced model development.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.5)
METHODS = {'entropy': 'entropy_score', 'mc_sd': 'mc_sd', 'fusion': 'fusion_score'}


def load(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'seed', 'disease_id', 'phenotype_id', 'partition', 'label',
                    'calibrated_probability', *METHODS.values()}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing columns: {sorted(missing)}')
        rows = list(reader)
    if not rows:
        raise ValueError('Empty predictions file')
    return rows


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows, indices):
    if len(indices) == 0:
        return {'retained': 0, 'errors': 0, 'risk': None, 'accuracy': None}
    y = np.array([int(rows[i]['label']) for i in indices])
    p = np.array([float(rows[i]['calibrated_probability']) for i in indices])
    errors = int(np.sum((p >= 0.5).astype(int) != y))
    return {'retained': len(indices), 'errors': errors,
            'risk': round(errors / len(indices), 8),
            'accuracy': round(1 - errors / len(indices), 8)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--target-coverage', type=float, default=0.8,
                    help='Calibration-selected retention fraction, default 0.8')
    args = ap.parse_args()
    if not 0 < args.target_coverage <= 1:
        ap.error('target-coverage must be in (0,1]')
    source = Path(args.predictions).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = load(source)
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row['seed']].append(row)
    summary, curves, decisions = [], [], []
    for seed in sorted(by_seed, key=int):
        records = by_seed[seed]
        keys = [(r['disease_id'], r['phenotype_id']) for r in records]
        if len(set(keys)) != len(keys):
            raise ValueError(f'Duplicate pairs for seed {seed}')
        for r in records:
            if r['partition'] not in ('calibration', 'evaluation'):
                raise ValueError('Unexpected partition')
            if int(r['label']) not in (0, 1):
                raise ValueError('Labels must be binary')
            if not 0 <= float(r['calibrated_probability']) <= 1:
                raise ValueError('Invalid probability')
            for col in METHODS.values():
                if not np.isfinite(float(r[col])) or float(r[col]) < 0:
                    raise ValueError(f'Invalid {col}')
        cal = [i for i, r in enumerate(records) if r['partition'] == 'calibration']
        ev = [i for i, r in enumerate(records) if r['partition'] == 'evaluation']
        if len(cal) < 20 or len(ev) < 20:
            raise ValueError(f'Insufficient partition size for seed {seed}')
        baseline = metrics(records, ev)
        for method, col in METHODS.items():
            # Deterministic order: score then pair ID. Cutoff chosen ONLY on calibration.
            ranked_cal = sorted(cal, key=lambda i: (float(records[i][col]), *keys[i]))
            for target in sorted(set((*COVERAGES, args.target_coverage)), reverse=True):
                n_cal = max(1, int(np.floor(target * len(cal))))
                # Use calibration score at the requested rank. Ties at threshold are
                # all retained, so achieved coverage may exceed target.
                threshold = float(records[ranked_cal[n_cal - 1]][col])
                selected_cal = [i for i in cal if float(records[i][col]) <= threshold]
                selected_ev = [i for i in ev if float(records[i][col]) <= threshold]
                mcal = metrics(records, selected_cal)
                mev = metrics(records, selected_ev)
                item = {'seed': seed, 'method': method, 'target_coverage': target,
                        'threshold_from_calibration': threshold,
                        'calibration_n': len(cal), 'evaluation_n': len(ev),
                        'calibration_coverage': round(len(selected_cal)/len(cal), 8),
                        'evaluation_coverage': round(len(selected_ev)/len(ev), 8),
                        'evaluation_abstention_rate': round(1-len(selected_ev)/len(ev), 8),
                        'calibration_risk': mcal['risk'],
                        'evaluation_risk': mev['risk'],
                        'evaluation_accuracy': mev['accuracy'],
                        'evaluation_retained': mev['retained'],
                        'evaluation_errors': mev['errors'],
                        'full_evaluation_risk': baseline['risk']}
                curves.append(item)
                if abs(target - args.target_coverage) < 1e-10:
                    summary.append(item)
                    for i in ev:
                        p = float(records[i]['calibrated_probability'])
                        decisions.append({'seed': seed, 'method': method,
                                          'disease_id': keys[i][0], 'phenotype_id': keys[i][1],
                                          'label': int(records[i]['label']),
                                          'probability': p, 'uncertainty': float(records[i][col]),
                                          'threshold': threshold,
                                          'decision': 'predict' if float(records[i][col]) <= threshold else 'abstain',
                                          'predicted_label': int(p >= 0.5) if float(records[i][col]) <= threshold else ''})
            print(f'seed={seed} method={method} target={args.target_coverage:.0%} '
                  f'eval_coverage={summary[-1]["evaluation_coverage"]:.1%} '
                  f'eval_risk={summary[-1]["evaluation_risk"]} '
                  f'full_risk={baseline["risk"]:.4f}')
    write_csv(out/'phase3_summary.csv', summary, list(summary[0]))
    write_csv(out/'phase3_risk_coverage.csv', curves, list(curves[0]))
    write_csv(out/'phase3_decisions.csv', decisions, list(decisions[0]))
    protocol = {
        'phase': 'TrustMed-GNN Phase 3', 'scope': 'exploratory validation only',
        'source': str(source), 'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'threshold_selection': 'calibration partition only; score quantile; ties retained',
        'evaluation': 'Phase 2B evaluation partition; no evaluation labels used for thresholds',
        'target_coverage': args.target_coverage, 'methods': METHODS,
        'limitations': ['Existing validation set already influenced development',
                        'Phase 2B entropy selected after reviewing evaluation results; method comparison exploratory',
                        'No independent test set', 'Unknown negatives not clinically verified',
                        'Training-time graph node order not independently verified',
                        'No clinical reliability or risk guarantee']}
    (out/'phase3_protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print(f'Outputs: {out}')

if __name__ == '__main__':
    main()
