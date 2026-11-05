"""TrustMed-GNN Phase 2A: validation-only uncertainty comparisons.
Uses only Phase 1 saved predictions. No checkpoint, graph, or test data loaded.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def number(row, field):
    v = float(row[field])
    if not math.isfinite(v):
        raise ValueError(f'Non-finite {field}: {row[field]}')
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    by_seed = defaultdict(list)
    with args.predictions.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            p = number(row, 'mc_mean_probability')
            sd = number(row, 'mc_sd')
            entropy = number(row, 'predictive_entropy')
            label = int(row['label'])
            if not 0 <= p <= 1 or sd < 0 or label not in (0, 1):
                raise ValueError('Invalid prediction row')
            error = int((p >= 0.5) != bool(label))
            by_seed[row['seed']].append({'error': error, 'mc_sd': sd,
                'entropy': entropy, 'least_confidence': min(p, 1-p)})
    if not by_seed:
        raise ValueError('No prediction records found')
    coverage_levels = [1.0, .9, .8, .7, .5]
    methods = ['mc_sd', 'entropy', 'least_confidence']
    risks, summary = [], []
    for seed in sorted(by_seed, key=lambda x: int(x)):
        records = by_seed[seed]
        n = len(records)
        for method in methods:
            # Stable ordering for ties. Lower score = more confident.
            ordered = sorted(enumerate(records), key=lambda t: (t[1][method], t[0]))
            cumulative_errors = 0
            total_risk = 0.0
            for k, (_, record) in enumerate(ordered, start=1):
                cumulative_errors += record['error']
                total_risk += cumulative_errors / k
            aurc = total_risk / n  # discrete AURC across every retained sample
            summary.append({'seed': seed, 'method': method, 'samples': n,
                            'aurc': f'{aurc:.8f}',
                            'full_risk': f'{sum(x["error"] for x in records)/n:.8f}'})
            for coverage in coverage_levels:
                k = max(1, math.floor(n * coverage))
                errors = sum(ordered[i][1]['error'] for i in range(k))
                risks.append({'seed': seed, 'method': method, 'coverage': coverage,
                              'retained': k, 'risk': f'{errors/k:.8f}'})
    def save(filename, records):
        with (args.out / filename).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    save('risk_coverage_comparison.csv', risks)
    save('aurc_by_seed.csv', summary)
    print('VALIDATION ONLY — exploratory; no independent test evidence')
    print('Method              Mean AURC     SD across seeds')
    for method in methods:
        values = [float(r['aurc']) for r in summary if r['method'] == method]
        print(f'{method:20s}{mean(values):.6f}      {stdev(values) if len(values)>1 else 0:.6f}')
    print('\nMean risk by method and coverage:')
    print('Method              100%      90%      80%      70%      50%')
    for method in methods:
        vals = [mean(float(r['risk']) for r in risks if r['method'] == method and r['coverage'] == c)
                for c in coverage_levels]
        print(f'{method:20s}' + '  '.join(f'{v:.4f}' for v in vals))
    print('\nNote: entropy and least-confidence have identical rankings for binary probabilities,')
    print('up to numerical rounding/ties; this is expected, not independent evidence.')
    print('Saved:', args.out.resolve())

if __name__ == '__main__':
    main()
