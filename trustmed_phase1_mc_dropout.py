#!/usr/bin/env python3
"""TrustMed-GNN phase 1: validation-only MC head-dropout uncertainty.
No training, no held-out test access, and no modification of frozen artifacts.
Run from TrustMed-GNN/baseline_medalign with --out pointing outside the baseline.
"""
import argparse, csv, hashlib, json, random
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from phase5e_label_blind_baseline import read_pairs, collect_label_blind_seeds, stream_edge_samples, make_graph
from phase6c_text_only_baseline import PairDecoder
from phase10e_text_first_residual import GraphResidual


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def metrics(y,p):
    p=np.clip(np.asarray(p),1e-7,1-1e-7)
    return dict(auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)),
                brier=float(brier_score_loss(y,p)),nll=float(log_loss(y,p,labels=[0,1])))


def ece(y,p,bins=10):
    ix=np.minimum((p*bins).astype(int),bins-1)
    return float(sum(np.mean(ix==b)*abs(np.mean(y[ix==b])-np.mean(p[ix==b]))
                     for b in range(bins) if np.any(ix==b)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True)
    ap.add_argument('--seeds',nargs='+',type=int,default=[11])
    ap.add_argument('--passes',type=int,default=10)
    ap.add_argument('--mode',choices=['constant','confidence','structural','combined'],default='constant')
    ap.add_argument('--pairs-dir',default='results/phase5e_label_blind')
    ap.add_argument('--text-checkpoints',default='results/phase10d_baseline_verification/strong_text_6c')
    ap.add_argument('--graph-checkpoints',default='results/phase10f_residual_ablation/constant')
    ap.add_argument('--csv',default='data/raw/kg.csv')
    ap.add_argument('--embeddings',default='data/processed/phase6b/embeddings.npy')
    ap.add_argument('--node-ids',default='data/processed/phase6b/node_ids.csv')
    ap.add_argument('--max-neighbors',type=int,default=4)
    ap.add_argument('--max-edges',type=int,default=30000)
    ap.add_argument('--chunksize',type=int,default=200000)
    ap.add_argument('--split-seed',type=int,default=2026)
    args=ap.parse_args()
    if args.passes<2 or not args.seeds or len(args.seeds)!=len(set(args.seeds)): ap.error('Need >=2 passes and unique seeds')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    source_root=Path.cwd().resolve()
    out=Path(args.out).resolve()
    if out==source_root or source_root in out.parents: ap.error('Output must be OUTSIDE the frozen baseline directory')
    if out.exists() and any(out.iterdir()): ap.error('Output directory must be new or empty')
    for s in ['train_positive','train_negative','val_positive','val_negative']:
        if not (Path(args.pairs_dir)/(s+'.csv')).is_file(): ap.error(f'Missing pair file: {s}')
    pairs={s:read_pairs(Path(args.pairs_dir)/(s+'.csv')) for s in ['train_positive','train_negative','val_positive','val_negative']}
    for s in pairs:
        for t in pairs:
            if s<t and set(pairs[s])&set(pairs[t]): raise ValueError(f'Overlapping pairs: {s}, {t}')
    forbidden=Path('data/processed/forbidden_positive_pairs.csv')
    if not forbidden.is_file(): raise FileNotFoundError('Required forbidden_positive_pairs.csv is missing')
    known=set(read_pairs(forbidden))
    for s in ('train_negative','val_negative'):
        if known&set(pairs[s]): raise ValueError(f'Known positive in {s}')
    embeddings=np.load(args.embeddings,mmap_mode='r')
    ids=pd.read_csv(args.node_ids,dtype=str,keep_default_na=False).node_id.tolist()
    if embeddings.ndim!=2 or len(ids)!=len(embeddings) or len(set(ids))!=len(ids): raise ValueError('Embedding/node-ID alignment failed')
    embedding_ix={v:i for i,v in enumerate(ids)}
    endpoints={x for group in pairs.values() for a,b in group for x in (a,b)}
    if endpoints-set(ids): raise ValueError('Missing endpoint text embedding')
    print('Reconstructing frozen graph (read-only)...',flush=True)
    seeds,_=collect_label_blind_seeds(args.csv,args.chunksize)
    seeds|={x for a,b in pairs['train_positive'] for x in (a,b)}
    _,targeted,_=stream_edge_samples(args.csv,seeds,args.max_edges,args.max_neighbors,args.split_seed,args.chunksize)
    kg=targeted
    nodes=sorted({x for a,b,_ in kg for x in (a,b)}|endpoints)
    node_ix={x:i for i,x in enumerate(nodes)}
    relations=sorted({r for _,_,r in kg})
    relnames=[f'kg::{r}::{direction}' for direction in ('forward','reverse') for r in relations]+['train_disease_phenotype','train_phenotype_disease']
    edge_index,edge_type=make_graph(kg,pairs['train_positive'],node_ix,{r:i for i,r in enumerate(relnames)})
    if (len(nodes),int(edge_index.shape[1]),len(relnames))!=(37481,180540,20):
        raise ValueError(f'Frozen graph shape mismatch: {(len(nodes),int(edge_index.shape[1]),len(relnames))}')
    degree=torch.log1p(torch.bincount(edge_index[1],minlength=len(nodes)).float())
    degree/=degree.max().clamp_min(1)
    exposure=torch.zeros(len(nodes))
    for a,b in pairs['train_positive']: exposure[node_ix[a]]=1;exposure[node_ix[b]]=1
    val=pairs['val_positive']+pairs['val_negative']
    y=np.array([1]*len(pairs['val_positive'])+[0]*len(pairs['val_negative']))
    va=torch.tensor([(node_ix[a],node_ix[b]) for a,b in val],dtype=torch.long)
    local_ids=sorted(endpoints);local_ix={x:i for i,x in enumerate(local_ids)}
    vectors=np.asarray(embeddings[[embedding_ix[x] for x in local_ids]],dtype=np.float32)
    norm=np.linalg.norm(vectors,axis=1,keepdims=True)
    if np.any(norm<=0): raise ValueError('Zero text embedding norm')
    vectors=torch.from_numpy(vectors/norm)
    a_ix=torch.tensor([local_ix[a] for a,b in val]);b_ix=torch.tensor([local_ix[b] for a,b in val])
    out.mkdir(parents=True,exist_ok=True)
    rows=[];summary=[];hashes={}
    for seed in args.seeds:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        text_path=Path(args.text_checkpoints)/f'best_seed{seed}.pt'
        graph_path=Path(args.graph_checkpoints)/f'best_seed{seed}.pt'
        for path in [text_path,graph_path]:
            if not path.is_file(): raise FileNotFoundError(path)
            hashes[str(path)]=digest(path)
        text_state=torch.load(text_path,map_location='cpu',weights_only=True)
        if int(text_state['embedding_dim'])!=vectors.shape[1]: raise ValueError('Text checkpoint dimension mismatch')
        text_model=PairDecoder(vectors.shape[1],int(text_state['hidden']),float(text_state['dropout']))
        text_model.load_state_dict(text_state['state_dict']);text_model.eval()
        with torch.inference_mode(): base=text_model(vectors[a_ix],vectors[b_ix]).detach()
        features=torch.stack([2*(base.sigmoid()-.5).abs(),.5*(degree[va[:,0]]+degree[va[:,1]]),exposure[va[:,0]]*exposure[va[:,1]]],dim=1)
        state=torch.load(graph_path,map_location='cpu',weights_only=True)
        if state.get('mode')!=args.mode: raise ValueError(f'Checkpoint mode mismatch: {state.get("mode")} vs {args.mode}')
        weights=state['state_dict'];hidden=int(weights['embedding.weight'].shape[1]);n_nodes=int(weights['embedding.weight'].shape[0])
        if n_nodes!=len(nodes): raise ValueError('Graph checkpoint node-count mismatch')
        drop=0.2  # Frozen Phase-10F protocol value; not stored in checkpoint.
        model=GraphResidual(len(nodes),len(relnames),hidden,drop,args.mode,float(state['correction_scale']))
        model.load_state_dict(weights,strict=True);model.eval()
        # Only activate the original pair-head dropout. R-GCN stays in eval mode.
        model.pair_head[2].train()
        draws=[]
        with torch.inference_mode():
            for i in range(args.passes):
                torch.manual_seed(seed*10000+i)
                residual,_=model(va,edge_index,edge_type,features)
                draws.append(torch.sigmoid(base+residual).numpy())
        draws=np.stack(draws)
        mean=draws.mean(axis=0);sd=draws.std(axis=0,ddof=1)
        uncertainty=-(mean*np.log(np.clip(mean,1e-7,1))+(1-mean)*np.log(np.clip(1-mean,1e-7,1)))
        base_p=base.sigmoid().numpy()
        m=metrics(y,mean)
        record={'seed':seed,'passes':args.passes,'mode':args.mode,'ece_10bin':ece(y,mean),'mean_mc_sd':float(sd.mean()),
                **{f'mc_{k}':v for k,v in m.items()},**{f'text_{k}':v for k,v in metrics(y,base_p).items()},
                'text_ece_10bin':ece(y,base_p)}
        summary.append(record)
        for j,(pair,label) in enumerate(zip(val,y)):
            rows.append({'seed':seed,'disease_id':pair[0],'phenotype_id':pair[1],'label':int(label),
                         'text_probability':float(base_p[j]),'mc_mean_probability':float(mean[j]),
                         'mc_sd':float(sd[j]),'predictive_entropy':float(uncertainty[j])})
        print(f'seed={seed}: MC AUROC={m["auroc"]:.4f}, AUPRC={m["auprc"]:.4f}, Brier={m["brier"]:.4f}, ECE={record["ece_10bin"]:.4f}',flush=True)
    pd.DataFrame(summary).to_csv(out/'validation_summary.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'validation_predictions.csv',index=False)
    metadata={'phase':'TrustMed phase 1','scope':'validation only; exploratory','test_loaded':False,
              'created_utc':datetime.now(timezone.utc).isoformat(),'passes':args.passes,'seeds':args.seeds,
              'mode':args.mode,'dropout':0.2,'graph_dimensions':[len(nodes),int(edge_index.shape[1]),len(relnames)],
              'checkpoint_sha256':hashes,
              'limitations':['Repeated validation use; not independent test evidence.',
                             'MC variation comes only from pair-head dropout, not graph structural uncertainty.',
                             'Unknown negatives are not clinically confirmed negatives.',
                             'Training-time graph node ordering not independently verified by a stored hash.',
                             'No temperature scaling or calibrated confidence guarantees.']}
    (out/'protocol.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Validation-only outputs:',out,flush=True)

if __name__=='__main__':main()
