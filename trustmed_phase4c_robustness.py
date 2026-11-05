#!/usr/bin/env python3
"""TrustMed-GNN Phase 4C: read-only, exploratory risk/coverage robustness audit."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

METHODS = ('entropy', 'mc_sd', 'fusion')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--risk-coverage', required=True)
    p.add_argument('--summary', required=True)
    p.add_argument('--out', required=True)
    a=p.parse_args()
    src=Path(a.risk_coverage); summary_path=Path(a.summary); out=Path(a.out)
    if not src.is_file() or not summary_path.is_file(): p.error('Input file missing')
    out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(src); s=pd.read_csv(summary_path)
    required={'seed','method','target_coverage','evaluation_coverage','evaluation_risk','evaluation_n','evaluation_retained','evaluation_errors','full_evaluation_risk'}
    missing=required-set(d.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    if d.duplicated(['seed','method','target_coverage']).any(): raise ValueError('Duplicate seed/method/coverage records')
    d=d.copy()
    d['coverage_gap']=d.evaluation_coverage-d.target_coverage
    d['relative_risk_reduction']=np.where(d.full_evaluation_risk>0,(d.full_evaluation_risk-d.evaluation_risk)/d.full_evaluation_risk,np.nan)
    full=d[np.isclose(d.target_coverage,1.0)].copy()
    if len(full)==0: raise ValueError('Missing 100% target rows')
    bad=full[(full.evaluation_retained!=full.evaluation_n) | (~np.isclose(full.evaluation_coverage,1.0))]
    # Correct reference rows for robustness reporting, without changing source CSV.
    if len(bad):
        mask=np.isclose(d.target_coverage,1.0)
        d.loc[mask,'evaluation_coverage']=1.0
        d.loc[mask,'evaluation_retained']=d.loc[mask,'evaluation_n']
        d.loc[mask,'evaluation_risk']=d.loc[mask,'full_evaluation_risk']
        d.loc[mask,'evaluation_errors']=np.rint(d.loc[mask,'evaluation_n']*d.loc[mask,'full_evaluation_risk']).astype(int)
        d.loc[mask,'coverage_gap']=0.0
        d.loc[mask,'relative_risk_reduction']=0.0
    d.to_csv(out/'phase4c_corrected_risk_coverage.csv',index=False)
    records=[]
    for (method,target),g in d.groupby(['method','target_coverage']):
        records.append({'method':method,'target_coverage':target,'n_seeds':len(g),
            'mean_actual_coverage':g.evaluation_coverage.mean(),
            'sd_actual_coverage':g.evaluation_coverage.std(ddof=1),
            'mean_coverage_gap':g.coverage_gap.mean(),
            'mean_risk':g.evaluation_risk.mean(),
            'sd_risk':g.evaluation_risk.std(ddof=1),
            'mean_full_risk':g.full_evaluation_risk.mean(),
            'mean_absolute_risk_reduction':(g.full_evaluation_risk-g.evaluation_risk).mean(),
            'mean_relative_risk_reduction':g.relative_risk_reduction.mean()})
    stats=pd.DataFrame(records).sort_values(['target_coverage','mean_risk','method'],ascending=[False,True,True])
    stats.to_csv(out/'phase4c_robustness_summary.csv',index=False)
    # Pairwise comparison is descriptive only, no significance testing on reused validation.
    comparisons=[]
    for target,g in d.groupby('target_coverage'):
        pivot=g.pivot(index='seed',columns='method',values='evaluation_risk')
        if 'entropy' not in pivot: continue
        for other in ('mc_sd','fusion'):
            if other not in pivot: continue
            diff=(pivot[other]-pivot['entropy']).dropna()
            comparisons.append({'target_coverage':target,'comparison':f'{other} minus entropy',
                'n_seeds':len(diff),'mean_risk_difference':diff.mean(),
                'entropy_lower_risk_seeds':int((diff>1e-12).sum()),
                'equal_risk_seeds':int((abs(diff)<=1e-12).sum()),
                'entropy_higher_risk_seeds':int((diff < -1e-12).sum())})
    pd.DataFrame(comparisons).to_csv(out/'phase4c_method_comparison.csv',index=False)
    protocol={'scope':'Exploratory validation; not independent test','source_sha256':{},
        'corrected_100pct_reference_rows':int(len(bad)),
        'caveats':['Same validation split reused during development',
                   'Sampled unknown negatives not clinically verified',
                   'Exact training-time graph identity not independently verified',
                   'Phase10H external candidates NOT APPROVED for independent test']}
    for path in (src,summary_path): protocol['source_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'phase4c_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print('Phase 4C complete (validation only).')
    print('100% coverage rows corrected:',len(bad))
    print(stats[['method','target_coverage','mean_actual_coverage','mean_risk','sd_risk']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('Outputs:',out.resolve())

if __name__=='__main__': main()
