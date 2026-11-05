#!/usr/bin/env python3
"""TrustMed-GNN Phase 5B.21: audit-only, resumable PubMed metadata collection."""
import csv
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'baseline_medalign'
INPUT = BASE / 'results/phase5b15_full_reference_audit/unique_pmids_NOT_APPROVED.csv'
OUT = BASE / 'results/phase5b21_pubmed_metadata'
CACHE = OUT / 'pubmed_metadata_cache_NOT_APPROVED.jsonl'
CSV_OUT = OUT / 'pubmed_metadata_NOT_APPROVED.csv'
SUMMARY = OUT / 'pubmed_metadata_audit_NOT_APPROVED.json'
URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'
BATCH = 100
WAIT = 0.5  # below 3 requests/second


def precision(pubdate):
    s = (pubdate or '').strip()
    if re.fullmatch(r'\d{4}', s):
        return 'year'
    if re.fullmatch(r'\d{4}\s+[A-Za-z]{3,9}', s):
        return 'month'
    if re.fullmatch(r'\d{4}\s+[A-Za-z]{3,9}\s+\d{1,2}', s):
        return 'day'
    return 'other' if s else 'missing'


def fetch(session, ids):
    for attempt in range(6):
        try:
            r = session.get(URL, params={'db': 'pubmed', 'id': ','.join(ids), 'retmode': 'json', 'tool': 'TrustMedGNN'}, timeout=45)
            r.raise_for_status()
            data = r.json()['result']
            result = {}
            for pmid in ids:
                record = data.get(pmid)
                if isinstance(record, dict) and 'error' not in record:
                    result[pmid] = {'pmid': pmid, 'retrieval_status': 'ok', 'pubdate': str(record.get('pubdate', '')), 'sortpubdate': str(record.get('sortpubdate', '')), 'title': str(record.get('title', ''))}
                else:
                    result[pmid] = {'pmid': pmid, 'retrieval_status': 'missing', 'pubdate': '', 'sortpubdate': '', 'title': ''}
            return result
        except (requests.RequestException, ValueError, KeyError) as exc:
            if attempt == 5:
                print(f'Batch failed after retries: {ids[0]}...{ids[-1]}: {exc}', flush=True)
                return {}
            time.sleep(min(2 ** attempt, 20))
    return {}


def main():
    if not INPUT.is_file():
        raise SystemExit(f'Missing input: {INPUT}')
    OUT.mkdir(parents=True, exist_ok=True)
    with INPUT.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1956:
        raise SystemExit(f'Unexpected PMID inventory length: {len(rows)}; expected 1956')
    ids = [r['pmid'].strip() for r in rows]
    if len(set(ids)) != len(ids) or not all(re.fullmatch(r'\d+', p) for p in ids):
        raise SystemExit('Invalid or duplicated PMIDs')
    cached = {}
    if CACHE.exists():
        with CACHE.open(encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item.get('pmid') in ids:
                        cached[item['pmid']] = item
                except json.JSONDecodeError:
                    continue
    pending = [p for p in ids if p not in cached or cached[p].get('retrieval_status') != 'ok']
    print(f'PMIDs={len(ids)}, cached={len(cached)}, to_request={len(pending)}', flush=True)
    with requests.Session() as session, CACHE.open('a', encoding='utf-8') as log:
        for offset in range(0, len(pending), BATCH):
            batch = pending[offset:offset + BATCH]
            got = fetch(session, batch)
            for pmid, item in got.items():
                cached[pmid] = item
                log.write(json.dumps(item, ensure_ascii=False) + '\n')
            log.flush()
            print(f'Processed {min(offset + BATCH, len(pending))}/{len(pending)} pending; received={len(got)}', flush=True)
            time.sleep(WAIT)
    fields = ['pmid', 'candidate_pair_count', 'retrieval_status', 'publication_date', 'date_precision', 'sortpubdate', 'title', 'temporal_independence_verified']
    output = []
    for row in rows:
        pmid = row['pmid'].strip()
        record = cached.get(pmid, {})
        date = record.get('pubdate', '')
        output.append({'pmid': pmid, 'candidate_pair_count': row['candidate_pair_count'], 'retrieval_status': record.get('retrieval_status', 'request_failed'), 'publication_date': date, 'date_precision': precision(date), 'sortpubdate': record.get('sortpubdate', ''), 'title': record.get('title', ''), 'temporal_independence_verified': 'False'})
    with CSV_OUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    status = Counter(r['retrieval_status'] for r in output)
    dates = Counter(r['date_precision'] for r in output)
    years = Counter(r['publication_date'][:4] for r in output if re.match(r'^\d{4}', r['publication_date']))
    summary = {'phase': '5B.21', 'status': 'AUDIT_ONLY_NOT_APPROVED', 'generated_utc': datetime.now(timezone.utc).isoformat(), 'total_pmids': len(ids), 'retrieval_status_counts': dict(status), 'date_precision_counts': dict(dates), 'publication_year_counts': dict(sorted(years.items())), 'temporal_independence_verified': False, 'external_test_approved': False, 'limitations': ['PubMed dates do not establish PrimeKG historical evidence cutoff.', 'Publication dates alone cannot establish source independence.', 'Non-PMID references require separate provenance investigation.', 'Date precision is preserved; sortpubdate must not be interpreted as exact when pubdate is partial.']}
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ['total_pmids', 'retrieval_status_counts', 'date_precision_counts', 'external_test_approved']}, indent=2))
    print(f'Outputs: {OUT}')


if __name__ == '__main__':
    main()
