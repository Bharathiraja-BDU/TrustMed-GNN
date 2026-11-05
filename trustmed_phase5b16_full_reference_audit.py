#!/usr/bin/env python3
"""TrustMed-GNN Phase 5B.16: audit-only full HPOA reference extraction.
Run from baseline_medalign, or pass --base PATH. No training or inference.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, default=Path.cwd())
    args = ap.parse_args()
    base = args.base.resolve()
    candidates_path = base / 'results/phase10h4_eligibility_audit/candidate_pairs_NOT_APPROVED.csv'
    hpo_path = base / 'data/external/hpo/phenotype.hpoa'
    out = base / 'results/phase5b15_full_reference_audit'
    for p in (candidates_path, hpo_path):
        if not p.is_file():
            raise SystemExit(f'Missing input: {p}')
    out.mkdir(parents=True, exist_ok=True)
    candidates = {}
    source_to_pairs = defaultdict(set)
    excluded = 0
    with candidates_path.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            key = (row['disease_index'], row['phenotype_index'])
            if row['passes_pair_exclusion_checks'].strip().lower() != 'true':
                excluded += 1
                continue
            if row['independent_test_approved'].strip().lower() == 'true':
                raise SystemExit('Unexpected approved candidate in NOT_APPROVED source')
            if key in candidates:
                raise SystemExit(f'Duplicate eligible index pair: {key}')
            candidates[key] = row
            source_to_pairs[(row['hpo_disease_id'], row['hpo_phenotype_id'])].add(key)
    if len(candidates) != 23784 or excluded != 1:
        raise SystemExit(f'Unexpected candidate counts: eligible={len(candidates)}, excluded={excluded}')
    meta = []
    annotations = []
    with hpo_path.open(newline='', encoding='utf-8-sig') as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if line.startswith('#'):
                meta.append(line.strip())
            else:
                f.seek(pos)
                break
        reader = csv.DictReader(f, delimiter='\t')
        required = {'database_id','hpo_id','qualifier','aspect','reference','evidence','biocuration'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise SystemExit(f'HPOA header missing columns: {required - set(reader.fieldnames or [])}')
        for row in reader:
            if row['qualifier'].strip() or row['aspect'].strip() != 'P':
                continue
            pair_keys = source_to_pairs.get((row['database_id'].strip(), row['hpo_id'].strip()))
            if not pair_keys:
                continue
            for pair in sorted(pair_keys):
                ref = row['reference'].strip()
                typ = 'PMID' if ref.startswith('PMID:') else ('OMIM' if ref.startswith('OMIM:') else ('EMPTY' if not ref else 'OTHER'))
                annotations.append({
                    'disease_index': pair[0], 'phenotype_index': pair[1],
                    'hpo_disease_id': row['database_id'].strip(),
                    'hpo_phenotype_id': row['hpo_id'].strip(),
                    'reference': ref, 'reference_type': typ,
                    'evidence': row['evidence'].strip(),
                    'biocuration': row['biocuration'].strip(),
                    'external_test_approved': 'False'
                })
    columns = ['disease_index','phenotype_index','hpo_disease_id','hpo_phenotype_id','reference','reference_type','evidence','biocuration','external_test_approved']
    with (out / 'candidate_annotation_references_NOT_APPROVED.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(annotations)
    pmid_to_pairs = defaultdict(set)
    matched_pairs = set()
    counts = Counter()
    for r in annotations:
        pair = (r['disease_index'], r['phenotype_index'])
        matched_pairs.add(pair)
        counts[r['reference_type']] += 1
        for pmid in re.findall(r'PMID:(\d+)', r['reference']):
            pmid_to_pairs[pmid].add(pair)
    with (out / 'unique_pmids_NOT_APPROVED.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['pmid','candidate_pair_count','publication_date','temporal_independence_verified'])
        for pmid, pairs in sorted(pmid_to_pairs.items(), key=lambda x: int(x[0])):
            w.writerow([pmid,len(pairs),'','False'])
    missing = sorted(set(candidates) - matched_pairs)
    with (out / 'unmatched_candidates_NOT_APPROVED.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['disease_index','phenotype_index','hpo_disease_id','hpo_phenotype_id'])
        for key in missing:
            r = candidates[key]
            w.writerow([*key,r['hpo_disease_id'],r['hpo_phenotype_id']])
    report = {
        'phase': '5B.16', 'status': 'AUDIT_ONLY_NOT_APPROVED',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'inputs_sha256': {'candidate_csv': digest(candidates_path), 'hpoa': digest(hpo_path)},
        'hpoa_metadata': meta, 'total_eligible_candidate_pairs': len(candidates),
        'excluded_candidate_rows': excluded, 'matched_candidate_pairs': len(matched_pairs),
        'unmatched_candidate_pairs': len(missing), 'joined_annotation_rows': len(annotations),
        'reference_type_counts': dict(counts), 'unique_pmids': len(pmid_to_pairs),
        'limitations': [
            'No publication dates retrieved or inferred.',
            'HPOA biocuration date does not establish original evidence date.',
            'No historical PrimeKG evidence source comparison performed.',
            'HPOA positive-only data cannot establish negative labels.',
            'External test is NOT_APPROVED; no model inference or training performed.'
        ], 'external_test_approved': False
    }
    (out / 'full_reference_audit_NOT_APPROVED.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('total_eligible_candidate_pairs','excluded_candidate_rows','matched_candidate_pairs','unmatched_candidate_pairs','joined_annotation_rows','reference_type_counts','unique_pmids','external_test_approved')}, indent=2))
    print(f'Outputs: {out}')

if __name__ == '__main__':
    main()
