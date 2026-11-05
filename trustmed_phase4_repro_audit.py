"""TrustMed-GNN Phase 4 read-only reproducibility and protocol audit.
Does NOT establish exact training-time graph identity or independent test approval.
"""
import argparse,csv,hashlib,json,collections,datetime
from pathlib import Path

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def csvrows(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
 a=argparse.ArgumentParser(description=__doc__)
 a.add_argument('--baseline',required=True)
 a.add_argument('--project',required=True)
 a.add_argument('--out',required=True)
 z=a.parse_args(); base=Path(z.baseline).resolve(); root=Path(z.project).resolve(); out=Path(z.out).resolve();out.mkdir(parents=True,exist_ok=True)
 checks=[]
 def record(name,status,detail):
  checks.append(dict(check=name,status=status,detail=str(detail)));print(f'{status:8} {name}: {detail}')
 proto=root/'results/phase1_mc_dropout_full/protocol.json'
 if proto.exists():
  p=json.loads(proto.read_text(encoding='utf-8')); record('validation_scope','PASS' if p.get('test_loaded') is False else 'FAIL',f"test_loaded={p.get('test_loaded')}")
  dims=p.get('graph_dimensions');record('reconstructed_graph_dimensions','PASS' if dims==[37481,180540,20] else 'WARN',dims)
  for rel,expected in p.get('checkpoint_sha256',{}).items():
   f=base/Path(rel.replace('\\','/'))
   record('checkpoint '+rel,'PASS' if f.is_file() and digest(f)==expected else 'FAIL', 'sha256 matched' if f.is_file() and digest(f)==expected else 'missing or hash mismatch')
 else:record('phase1_protocol','FAIL','missing '+str(proto))
 phase2=root/'results/phase2b_calibration_fusion/phase2b_predictions.csv'
 if phase2.exists():
  rows=csvrows(phase2);required={'seed','disease_id','phenotype_id','label','partition'}
  if rows and required.issubset(rows[0]):
   assignments=collections.defaultdict(set);labels=collections.defaultdict(set);dupes=collections.Counter()
   for r in rows:
    key=(r['disease_id'],r['phenotype_id']);assignments[key].add(r['partition']);labels[key].add(r['label']);dupes[(r['seed'],*key)]+=1
   mixed=sum(len(x)>1 for x in assignments.values());conflicts=sum(len(x)>1 for x in labels.values());duplicate=sum(v-1 for v in dupes.values() if v>1)
   record('pair_partition_consistency','PASS' if mixed==0 else 'FAIL',f'{mixed} pairs assigned to multiple partitions')
   record('pair_label_consistency','PASS' if conflicts==0 else 'FAIL',f'{conflicts} conflicting labels')
   record('duplicate_seed_pairs','PASS' if duplicate==0 else 'FAIL',f'{duplicate} duplicate records')
   record('phase2b_partitions','INFO',dict(collections.Counter(r['partition'] for r in rows)))
  else:record('phase2b_schema','FAIL','missing required columns')
 else:record('phase2b_predictions','WARN','missing; cannot check split consistency')
 phase3=root/'results/phase3_selective_prediction/phase3_risk_coverage.csv'
 if phase3.exists():
  rows=csvrows(phase3);record('phase3_risk_coverage','INFO',f'{len(rows)} records; inspect 100% target handling separately')
 else:record('phase3_risk_coverage','WARN','missing')
 for rel in ['manifests/frozen_data.json','manifests/medalign_models_manifest.csv','manifests/medalign_graph_manifest.csv']:
  f=root/rel;record('archive '+rel,'PASS' if f.is_file() else 'WARN','present' if f.is_file() else 'missing')
 record('exact_training_graph_identity','UNVERIFIED','matching dimensions do not prove node ordering, edge ordering, edge types or training-time sampler identity')
 record('independent_test_approval','NOT_APPROVED','Phase10H candidates are not approved as an independent test set')
 record('negative_label_validity','UNVERIFIED','unknown pairs are not clinically verified negatives')
 result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'read-only reproducibility audit; not independent test evidence','baseline':str(base),'project':str(root),'checks':checks}
 (out/'phase4_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 with (out/'phase4_checks.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['check','status','detail']);w.writeheader();w.writerows(checks)
 print('\nSaved:',out)
if __name__=='__main__':main()
