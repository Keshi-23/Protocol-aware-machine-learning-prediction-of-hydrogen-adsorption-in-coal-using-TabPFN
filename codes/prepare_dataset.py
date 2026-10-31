"""Basis harmonization, provisional rank estimation and recalculated common support.
No four-component total normalization; reported moisture is never changed.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parent
P='Pressure (MPa)';T='Temp (K)';Y='H2 excess sorption (mmol/g)'
FEATURES=[P,T,'Moisture Content (wt%)','Ash Content (wt%)','Volatile Content (wt%)','Fixed carbon (wt%)']
COMP=FEATURES[2:];M,A,V,F=COMP
CLOSURE_TOL=.02  # wt percentage points; rounding tolerance only
MONO_MODE='flag'  # diagnostic only: no observations removed
USE_COMMON_SUPPORT=False # full dataset; eligibility is only a scoring mask
RANK_MAP={'Li':'Lignite','Sub-Bi':'Subbituminous','Bi':'Bituminous','Bi-1':'Bituminous','Bi-4':'Bituminous','An':'Anthracitic'}

def group_keys(d):
    key=d[COMP].round(2).astype(str).agg('|'.join,axis=1)
    d=d.copy();d['sample_id']=pd.factorize(key)[0]
    # Source-aware isotherms prevent pooling different authors' coincident compositions.
    d['isotherm_id']=pd.factorize(d['source_key'].astype(str)+'|'+key+'|'+d[T].astype(str))[0]
    return d

def harmonize(raw):
    d=raw.copy();d['original_excel_row']=np.arange(len(d))+2
    for c in COMP:d['reported_'+c]=d[c]
    d['reported_total']=d[COMP].sum(axis=1);d['reported_dry_total']=d[[A,V,F]].sum(axis=1)
    d['basis_status']='unresolved';d['basis_eligible']=False;d['fc_rounding_adjustment']=0.
    for i,r in d.iterrows():
        if not np.isfinite(r[COMP].to_numpy(float)).all() or min(r[COMP])<0 or r[M]>=100:continue
        if abs(r.reported_total-100)<=CLOSURE_TOL+1e-9:
            d.loc[i,'basis_status']='consistent_with_as_received';d.loc[i,'basis_eligible']=True
        elif abs(r.reported_dry_total-100)<=CLOSURE_TOL+1e-9:
            d.loc[i,[A,V,F]]=r[[A,V,F]].to_numpy(float)*(1-r[M]/100)
            d.loc[i,'basis_status']='dry_solids_to_as_received';d.loc[i,'basis_eligible']=True
        else:continue
        # Close only sub-tolerance rounding residual via FC, never alter moisture.
        residual=100-float(d.loc[i,COMP].sum())
        assert abs(residual)<=CLOSURE_TOL+1e-8
        d.loc[i,F]+=residual;d.loc[i,'fc_rounding_adjustment']=residual
    d['harmonized_total']=d[COMP].sum(axis=1)
    assert np.array_equal(d[M],d['reported_'+M])
    assert np.allclose(d.loc[d.basis_eligible,COMP].sum(axis=1),100)
    d['Sample_reported']=d.Sample
    d['rank_reported']=d.Sample.map(RANK_MAP)
    d['rank_assigned']=d.rank_reported
    d['rank_status']=np.where(d.rank_reported.notna(),'reported label; not independently verified','missing')
    # Preserve DOI metadata as supplied. No inference of publication years from labels.
    d['source_key']=d.Link.fillna(d.Author).astype(str).str.strip()
    return d

def estimate_ranks(d):
    classes=['Lignite','Subbituminous','Bituminous','Anthracitic'];codes={c:i+1 for i,c in enumerate(classes)}
    # One reference per unique composition, avoiding pressure-series pseudoreplication.
    known=d[d.basis_eligible & d.rank_reported.notna()]
    proto=[];conf=[]
    for vals,g in known.groupby(COMP):
        labels=sorted(g.rank_reported.unique())
        if len(labels)>1:
            conf.append(dict(zip(COMP,vals))|dict(labels='; '.join(labels),authors='; '.join(g.Author.unique())))
            continue
        proto.append(dict(zip(COMP,vals))|dict(rank=labels[0],authors='; '.join(g.Author.unique())))
    ref=pd.DataFrame(proto);sc=StandardScaler().fit(ref[COMP]);z=sc.transform(ref[COMP]);rows=[];neighbors=[]
    missing=d[d.rank_reported.isna() & d.basis_eligible]
    for vals,g in missing.groupby(['Author']+COMP,dropna=False):
        vec=np.asarray(vals[1:],float);dist=np.linalg.norm(z-sc.transform(pd.DataFrame([vec],columns=COMP))[0],axis=1)
        ii=np.argsort(dist)[:min(3,len(ref))];w=1/np.maximum(dist[ii],1e-8);w/=w.sum()
        ordinal=float(np.dot(w,ref.iloc[ii]['rank'].map(codes)))
        label=classes[int(np.clip(np.floor(ordinal+.5),1,4))-1]
        outside=bool(((vec<ref[COMP].min().to_numpy())|(vec>ref[COMP].max().to_numpy())).any())
        agreement=float(sum(w[j] for j,k in enumerate(ii) if ref.iloc[k]['rank']==label))
        rec=dict(author=vals[0],n_rows=len(g),estimated_rank=label,ordinal_score=ordinal,
                 nearest_distance=float(dist[ii[0]]),neighbor_weight_for_label=agreement,
                 outside_reference_feature_ranges=outside,status='provisional composition-based estimate; not ASTM rank')
        rec.update(dict(zip(COMP,vec)));rows.append(rec)
        d.loc[g.index,'rank_assigned']=label;d.loc[g.index,'rank_status']=rec['status']
        for j,k in enumerate(ii):neighbors.append(dict(target_author=vals[0],neighbor=j+1,distance=float(dist[k]),weight=float(w[j]),**ref.iloc[k].to_dict()))
    return d,pd.DataFrame(rows),pd.DataFrame(neighbors),pd.DataFrame(conf)

def prepare():
    raw=pd.read_excel(ROOT/'data/revised_input.xlsx');raw.columns=raw.columns.str.strip()
    raw=raw.rename(columns={'Authors':'Author','P (MPa)':P})
    assert all(c in raw for c in FEATURES+[Y,'Author','Link','Sample'])
    d=harmonize(raw);d,ranks,neighbors,conf=estimate_ranks(d)
    base=group_keys(d[d.basis_eligible].copy()).reset_index(drop=True)
    drop=[];mono=[]
    for iso,g in base.groupby('isotherm_id'):
        g=g.sort_values(P);delta=np.diff(g[Y]);peak=max(float(g[Y].max()),1e-12)
        bad=np.where((delta < -.02-1e-12)&(-delta/peak > .1+1e-12))[0]
        for j in bad:
            row=g.iloc[j+1];drop.append(g.index[j+1]);mono.append(dict(original_excel_row=int(row.original_excel_row),isotherm_id=iso,drop_mmol_g=float(-delta[j]),relative_drop=float(-delta[j]/peak)))
    if MONO_MODE=='drop_points':screened=base.drop(index=sorted(set(drop))).reset_index(drop=True)
    elif MONO_MODE=='flag':screened=base.copy()
    else:raise ValueError('MONO_MODE must be drop_points or flag')
    screened=group_keys(screened).reset_index(drop=True);screened['row_idx']=np.arange(len(screened))
    history=[]
    def lang(p,n,l):return n*p/(p+l)
    for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=42).split(screened),1):
        for iso in screened.iloc[te].isotherm_id.unique():
            train=screened.iloc[tr];train=train[train.isotherm_id.eq(iso)]
            test=screened.iloc[te];test=test[test.isotherm_id.eq(iso)]
            count=train[P].nunique();pred=np.full(len(test),np.nan)
            reason='fewer than 3 distinct training pressures; fit not attempted'
            if count>=3:
                try:
                    pars,_=curve_fit(lang,train[P],train[Y],p0=[max(train[Y].max(),.001),1],bounds=([0,1e-6],[np.inf,np.inf]),maxfev=20000)
                    pred=lang(test[P].to_numpy(),*pars);reason='Langmuir prediction available'
                except (RuntimeError,ValueError,FloatingPointError) as e:reason='solver failure: '+str(e)
            for (_,row),value in zip(test.iterrows(),pred):history.append(dict(original_excel_row=int(row.original_excel_row),row_idx=int(row.row_idx),fold=fold,n_training=len(train),unique_training_pressures=count,Langmuir=value,retained=bool(np.isfinite(value)),reason=reason))
    audit=pd.DataFrame(history).sort_values('row_idx');assert len(audit)==len(screened)
    selected=screened[screened.original_excel_row.isin(audit.loc[audit.retained,'original_excel_row'])].copy() if USE_COMMON_SUPPORT else screened.copy()
    selected=group_keys(selected).reset_index(drop=True);selected.insert(0,'row_model',np.arange(len(selected)))
    d['screen_flag']=d.original_excel_row.isin([m['original_excel_row'] for m in mono])
    d['selected_for_model']=d.original_excel_row.isin(selected.original_excel_row)
    assert len(selected)>20 and selected['original_excel_row'].is_unique
    for name,frame in [('all_rows_audit',d),('unresolved_basis',d[~d.basis_eligible]),('rank_estimates',ranks),('rank_neighbors',neighbors),('rank_conflicts',conf),('monotonicity_flags',pd.DataFrame(mono)),('langmuir_support_audit',audit.rename(columns={'retained':'Langmuir_available'})),('model_dataset',selected)]:frame.to_csv(ROOT/'data'/f'{name}.csv',index=False)
    selected[['Author']+FEATURES+[Y]].to_csv(ROOT/'data/model_input.csv',index=False)
    summary=dict(input_rows=len(d),basis_eligible=int(d.basis_eligible.sum()),dry_converted=int(d.basis_status.eq('dry_solids_to_as_received').sum()),unresolved=int((~d.basis_eligible).sum()),monotonicity_flagged=len(set(drop)),monotonicity_mode=MONO_MODE,screened_rows=len(screened),diagnostic_cv_langmuir_covered=int(audit.retained.sum()),model_rows=len(selected),use_common_support=USE_COMMON_SUPPORT,composition_groups=selected.sample_id.nunique(),isotherms=selected.isotherm_id.nunique(),sources=selected.source_key.nunique())
    (ROOT/'data/preparation_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    return selected,d
if __name__=='__main__':prepare()
