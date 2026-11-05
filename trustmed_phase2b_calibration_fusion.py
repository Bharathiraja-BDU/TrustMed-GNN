"""TrustMed-GNN Phase 2B: exploratory calibration and uncertainty fusion.

Reads only Phase 1 validation predictions. No checkpoints, graphs or test labels loaded.
Uses a stable pair-level split shared across seeds, stratified by label.
Temperature scaling is applied to logit(MC mean probability), not to original
model logits; therefore it is post-hoc probability recalibration.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

EPS = 1e-8
COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.5]
WEIGHTS = [i / 10 for i in range(11)]


def read_rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'seed','disease_id','phenotype_id','label','mc_mean_probability','mc_sd'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing columns: {sorted(required - set(reader.fieldnames or []))}')
        rows = list(reader)
    if not rows:
        raise ValueError('Empty predictions file')
    return rows


def hash_key(key, salt):
    return int(hashlib.sha256((salt + '|' + '|'.join(key)).encode()).hexdigest()[:16], 16)


def split_pairs(rows, fraction, salt):
    labels = {}
    for r in rows:
        key = (r['disease_id'], r['phenotype_id'])
        label = int(r['label'])
        if label not in (0, 1):
            raise ValueError('Labels must be binary')
        if key in labels and labels[key] != label:
            raise ValueError(f'Conflicting labels for pair {key}')
        labels[key] = label
    by_label = defaultdict(list)
    for key, label in labels.items():
        by_label[label].append(key)
    calibration = set()
    for label, keys in by_label.items():
        keys.sort(key=lambda k: hash_key(k, salt))
        n = max(1, min(len(keys)-1, round(len(keys)*fraction)))
        if len(keys) < 2:
            raise ValueError(f'Not enough unique pairs for class {label}')
        calibration.update(keys[:n])
    return calibration, labels


def get_arrays(rows):
    p = np.clip(np.array([float(r['mc_mean_probability']) for r in rows]), EPS, 1-EPS)
    sd = np.array([float(r['mc_sd']) for r in rows])
    y = np.array([int(r['label']) for r in rows], dtype=int)
    if not np.isfinite(p).all() or not np.isfinite(sd).all() or (sd < 0).any():
        raise ValueError('Invalid probability or MC standard deviation')
    return p, sd, y


def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1/(1+np.exp(-z))


def temperature_probs(p, temperature):
    logit = np.log(p/(1-p))
    return sigmoid(logit/temperature)


def nll(p, y):
    p = np.clip(p, EPS, 1-EPS)
    return float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p)))


def fit_temperature(p, y):
    # One-dimensional bounded grid search; deterministic, SciPy-free.
    grid = np.exp(np.linspace(np.log(0.2), np.log(10), 601))
    losses = [nll(temperature_probs(p, t), y) for t in grid]
    return float(grid[int(np.argmin(losses))])


def entropy(p):
    p = np.clip(p, EPS, 1-EPS)
    return -(p*np.log(p)+(1-p)*np.log(1-p))


def percentile_score(x, reference):
    sorted_ref = np.sort(reference)
    return np.searchsorted(sorted_ref, x, side='right') / len(sorted_ref)


def aurc(score, p, y):
    errors = ((p >= 0.5).astype(int) != y).astype(float)
    order = np.argsort(score, kind='stable')
    cumulative = np.cumsum(errors[order])
    return float(np.mean(cumulative / np.arange(1, len(y)+1)))


def coverage_risks(score, p, y):
    errors = ((p >= 0.5).astype(int) != y).astype(int)
    order = np.argsort(score, kind='stable')
    result = []
    for c in COVERAGES:
        n = max(1, int(math.floor(len(y)*c)))
        result.append((c, n, float(np.mean(errors[order[:n]]))))
    return result


def ece(p, y, bins=15):
    result = 0.0
    for i in range(bins):
        lo, hi = i/bins, (i+1)/bins
        mask = (p >= lo) & ((p < hi) if i < bins-1 else (p <= hi))
        if mask.any():
            result += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return result


def save_csv(path, rows, columns):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--calibration-fraction', type=float, default=0.4)
    ap.add_argument('--split-salt', default='trustmed-phase2b-v1')
    args = ap.parse_args()
    if not 0.1 <= args.calibration_fraction <= 0.9:
        ap.error('calibration-fraction must be between 0.1 and 0.9')
    source = Path(args.predictions).resolve()
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(source)
    calibration_pairs, labels = split_pairs(rows, args.calibration_fraction, args.split_salt)
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[r['seed']].append(r)
    summary, risk_rows, predictions_out = [], [], []
    for seed in sorted(by_seed, key=lambda s: int(s)):
        seed_rows = by_seed[seed]
        keys = [(r['disease_id'], r['phenotype_id']) for r in seed_rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f'Duplicate pairs in seed {seed}')
        p, sd, y = get_arrays(seed_rows)
        cal = np.array([k in calibration_pairs for k in keys])
        ev = ~cal
        if cal.sum() < 20 or ev.sum() < 20 or len(np.unique(y[cal])) < 2 or len(np.unique(y[ev])) < 2:
            raise ValueError(f'Insufficient calibration/evaluation samples in seed {seed}')
        temp = fit_temperature(p[cal], y[cal])
        pc = temperature_probs(p, temp)
        ent = entropy(pc)
        # Normalize both uncertainty signals by calibration empirical percentiles.
        ent_rank = percentile_score(ent, ent[cal])
        sd_rank = percentile_score(sd, sd[cal])
        # Select fusion weight on calibration data only.
        candidate_aurcs = [aurc(w*ent_rank[cal]+(1-w)*sd_rank[cal], pc[cal], y[cal]) for w in WEIGHTS]
        best_idx = int(np.argmin(candidate_aurcs))
        w = WEIGHTS[best_idx]
        scores = {'entropy': ent, 'mc_sd': sd, 'fusion': w*ent_rank+(1-w)*sd_rank}
        for method, score in scores.items():
            a = aurc(score[ev], pc[ev], y[ev])
            summary.append({'seed': seed, 'method': method, 'calibration_n': int(cal.sum()), 'evaluation_n': int(ev.sum()), 'temperature': round(temp, 6), 'entropy_weight': round(w, 2), 'evaluation_aurc': round(a, 8), 'evaluation_nll': round(nll(pc[ev],y[ev]), 8), 'evaluation_brier': round(float(np.mean((pc[ev]-y[ev])**2)), 8), 'evaluation_ece': round(ece(pc[ev],y[ev]), 8), 'uncalibrated_nll': round(nll(p[ev],y[ev]), 8), 'uncalibrated_brier': round(float(np.mean((p[ev]-y[ev])**2)), 8), 'uncalibrated_ece': round(ece(p[ev],y[ev]), 8)})
            for cov, n, risk in coverage_risks(score[ev], pc[ev], y[ev]):
                risk_rows.append({'seed':seed,'method':method,'coverage':cov,'retained':n,'risk':round(risk,8)})
        for i, r in enumerate(seed_rows):
            predictions_out.append({'seed':seed,'disease_id':r['disease_id'],'phenotype_id':r['phenotype_id'],'partition':'calibration' if cal[i] else 'evaluation','label':int(y[i]),'original_probability':float(p[i]),'calibrated_probability':float(pc[i]),'mc_sd':float(sd[i]),'entropy_score':float(ent[i]),'fusion_score':float(scores['fusion'][i])})
        print(f'seed={seed} cal={cal.sum()} eval={ev.sum()} T={temp:.3f} fusion_entropy_weight={w:.1f} | eval AURC entropy={aurc(ent[ev],pc[ev],y[ev]):.4f}, MC_SD={aurc(sd[ev],pc[ev],y[ev]):.4f}, fusion={aurc(scores["fusion"][ev],pc[ev],y[ev]):.4f}')
    save_csv(output/'phase2b_summary.csv',summary,list(summary[0]))
    save_csv(output/'phase2b_risk_coverage.csv',risk_rows,list(risk_rows[0]))
    save_csv(output/'phase2b_predictions.csv',predictions_out,list(predictions_out[0]))
    protocol = {'phase':'TrustMed Phase 2B','scope':'exploratory validation only; not independent testing','input':str(source),'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'split_salt':args.split_salt,'calibration_fraction':args.calibration_fraction,'split_unit':'unique disease_id + phenotype_id, shared across seeds, stratified by label','seeds':sorted(by_seed,key=lambda s:int(s)),'temperature_scaling':'logit of MC mean probability; not original model logits','fusion_weights':WEIGHTS,'weight_selection':'minimum calibration AURC; tie selects lower entropy weight','evaluation':'held-out portion of previously reused validation data; exploratory only','limitations':['No independent test set','Validation already influenced model development','Graph training-time ordering not independently verified','MC SD represents pair-head dropout only','No clinical claims or calibrated guarantees']}
    (output/'phase2b_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print(f'Outputs: {output}')

if __name__ == '__main__':
    main()
