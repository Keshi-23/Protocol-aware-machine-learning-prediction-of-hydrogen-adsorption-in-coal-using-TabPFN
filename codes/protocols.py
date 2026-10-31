"""Matched splits, training-only Langmuir fits and plot-ready exports."""
from pathlib import Path
import json
import importlib.metadata
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from prepare_dataset import ROOT, FEATURES, P, T, Y
PROTOCOLS=['A_random_rows','B_holdout_isotherm','C_holdout_sample','D_holdout_source']

def grouped_split(groups, test_fraction=.2, seed=42):
    """Seeded subset-sum on GROUP SIZES only; nearest achievable 20% rows.
    Outcome and model errors never enter split construction. Groups stay intact.
    """
    groups=np.asarray(groups); unique,counts=np.unique(groups,return_counts=True)
    if len(unique)<2:raise ValueError('Need at least two groups')
    order=np.random.default_rng(seed).permutation(len(unique)); ways={0:()}
    for j in order:
        for total,chosen in list(ways.items()):
            new=total+int(counts[j])
            if new not in ways:ways[new]=chosen+(int(j),)
    target=int(np.ceil(len(groups)*test_fraction))
    n=min((k for k in ways if 0<k<len(groups)),key=lambda k:(abs(k-target),k))
    mask=np.isin(groups,unique[list(ways[n])]);return np.flatnonzero(~mask),np.flatnonzero(mask)

def make_splits(d,seed=42):
    n=len(d); idx=np.arange(n)
    groups=[None,d.isotherm_id.to_numpy(),d.sample_id.to_numpy(),d.source_key.astype(str).to_numpy()]
    for name,g in zip(PROTOCOLS,groups):
        tr,te=train_test_split(idx,test_size=.2,random_state=seed) if g is None else grouped_split(g,seed=seed)
        assert len(set(tr)&set(te))==0 and len(tr)+len(te)==n
        if g is not None:assert not set(g[tr])&set(g[te])
        yield name,np.sort(tr),np.sort(te),g

def lang(P,nL,PL):return nL*P/(P+PL)

def langmuir_fit(d,tr):
    """Fit per-isotherm parameters and a separately labelled pooled baseline."""
    fits={}; logs=[]
    for iso in d.isotherm_id.unique():
        g=d.iloc[tr];g=g[g.isotherm_id.eq(iso)]
        rec=dict(isotherm_id=int(iso),n_train=len(g),n_unique_pressure=g[P].nunique())
        if len(g)<3 or g[P].nunique()<3:
            rec['status']='unavailable: fewer than 3 distinct training pressures'
        else:
            try:
                pars,cov=curve_fit(lang,g[P].to_numpy(float),g[Y].to_numpy(float),
                    p0=[max(float(g[Y].max()),.001),1.],
                    bounds=([0,1e-6],[np.inf,np.inf]),maxfev=20000)
                fits[iso]=pars;rec.update(status='fitted',nL=pars[0],PL=pars[1],
                                         covariance_finite=bool(np.isfinite(cov).all()))
            except (RuntimeError,ValueError,FloatingPointError) as e:
                rec['status']='solver failure: '+str(e)
        logs.append(rec)
    g=d.iloc[tr]
    pooled=curve_fit(lang,g[P],g[Y],p0=[max(g[Y].max(),.001),1.],
        bounds=([0,1e-6],[np.inf,np.inf]),maxfev=20000)[0]
    logs.append(dict(isotherm_id='pooled',n_train=len(g),n_unique_pressure=g[P].nunique(),
                     status='fitted',nL=pooled[0],PL=pooled[1]))
    pred=np.array([lang(p,*fits[i]) if i in fits else np.nan for p,i in zip(d[P],d.isotherm_id)])
    return pred,lang(d[P].to_numpy(),*pooled),pd.DataFrame(logs)

def score(y,p):
    ok=np.isfinite(p);y=np.asarray(y)[ok];p=np.asarray(p)[ok]
    return dict(n=len(y),R2=r2_score(y,p) if len(y)>1 else np.nan,
        RMSE=float(np.sqrt(mean_squared_error(y,p))) if len(y) else np.nan,
        MAE=float(mean_absolute_error(y,p)) if len(y) else np.nan)

def diagnostics(d,tr,te,pred):
    """Leverage from training design; standardized residual = residual/train RMSE."""
    a=d[FEATURES].to_numpy(float);mu=a[tr].mean(0);sd=a[tr].std(0);sd[sd==0]=1
    z=np.column_stack([np.ones(len(a)),(a-mu)/sd]);zt=z[tr]
    h=np.einsum('ij,jk,ik->i',z,np.linalg.pinv(zt.T@zt),z)
    rank=np.linalg.matrix_rank(zt);hstar=3*rank/len(tr)
    rmse=score(d.iloc[tr][Y],pred[tr])['RMSE']
    resid=d[Y].to_numpy()-pred
    return h,hstar,resid/rmse if rmse>0 else np.full(len(d),np.nan)

def run(ns):
    out=Path(ns['OUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(ROOT/'data/model_dataset.csv');X=ns['X'];y=ns['y']
    assert len(d)==len(X) and np.allclose(d[Y],y)
    assert ns['feature_names']==FEATURES
    allpred=[]; mets=[]; fits=[]; splits=[]; cvpred=[]; cvmets=[]; settings=[]
    for name,tr,te,group in make_splits(d,ns['SEED']):
        print(name,'train',len(tr),'test',len(te))
        split=dict(protocol=name,n_train=len(tr),n_test=len(te),test_fraction=len(te)/len(d),
          isotherm_overlap=len(set(d.iloc[tr].isotherm_id)&set(d.iloc[te].isotherm_id)),
          composition_overlap=len(set(d.iloc[tr].sample_id)&set(d.iloc[te].sample_id)),
          source_label_overlap=len(set(d.iloc[tr].Author)&set(d.iloc[te].Author)),
          source_doi_overlap=len(set(d.iloc[tr].source_key)&set(d.iloc[te].source_key)))
        splits.append(split)
        pred={};per,pooled,log=langmuir_fit(d,tr);log.insert(0,'protocol',name);fits.append(log)
        pred['Langmuir_per_isotherm']=per;pred['Langmuir_pooled']=pooled
        for model in ['TabPFN','XGB']:
            m,sc=ns['fit_model'](model,X[tr],y[tr],groups_tr=None if group is None else group[tr],grouped=group is not None)
            pred[model]=ns['predict_model'](model,m,sc,X)
            settings.append(dict(protocol=name,model=model,parameters=m.get_params() if hasattr(m,'get_params') else {},scaling='MinMax fitted on training rows'))
        for subset,ii in [('train',tr),('test',te)]:
            frame=d.iloc[ii][['row_model','row_idx','original_excel_row','Author','source_key','Sample_reported','rank_reported','rank_assigned','rank_status','basis_status','sample_id','isotherm_id']+FEATURES+[Y]].copy()
            frame=frame.rename(columns={Y:'measured'});frame.insert(0,'subset',subset);frame.insert(0,'protocol',name)
            common=np.isfinite(per[ii])
            frame['Langmuir_eligible']=common
            for model,p in pred.items():
                frame[model]=p[ii]
                mets.append(dict(protocol=name,subset=subset,model=model,support='available',n_total=len(ii),**score(y[ii],p[ii])))
                mets.append(dict(protocol=name,subset=subset,model=model,support='common_per_isotherm',n_total=len(ii),**score(y[ii][common],p[ii][common])))
                if model in ['TabPFN','XGB']:
                    h,hs,r=diagnostics(d,tr,te,p);frame[model+'_leverage']=h[ii];frame[model+'_hstar']=hs;frame[model+'_std_residual']=r[ii]
            allpred.append(frame)
        if ns.get('RUN_1_CV',True):
            splitter=KFold(5,shuffle=True,random_state=ns['SEED']) if group is None else GroupKFold(min(5,len(np.unique(group))))
            for f,(ct,ce) in enumerate(splitter.split(X,y,group),1):
                fp=pd.DataFrame(dict(protocol=name,fold=f,row_idx=ce,actual=y[ce]))
                for model in ['TabPFN','XGB']:
                    p,_,_=ns['fit_predict'](model,X[ct],y[ct],X[ce],groups_tr=None if group is None else group[ct],grouped=group is not None)
                    fp[model]=p
                lp,pg,_=langmuir_fit(d,ct);fp['Langmuir_per_isotherm']=lp[ce];fp['Langmuir_pooled']=pg[ce]
                common=np.isfinite(lp[ce]);fp['Langmuir_eligible']=common
                fp['original_excel_row']=d.iloc[ce].original_excel_row.to_numpy()
                for model in ['TabPFN','XGB','Langmuir_per_isotherm','Langmuir_pooled']:
                    values=fp[model].to_numpy()
                    cvmets.append(dict(protocol=name,fold=f,model=model,support='available',n_total=len(ce),**score(y[ce],values)))
                    cvmets.append(dict(protocol=name,fold=f,model=model,support='common_per_isotherm',n_total=len(ce),**score(y[ce][common],values[common])))
                cvpred.append(fp)
    predictions=pd.concat(allpred,ignore_index=True);metrics=pd.DataFrame(mets)
    predictions.to_csv(out/'train_test_predictions.csv',index=False)
    metrics.to_csv(out/'train_test_metrics.csv',index=False,na_rep='N/A')
    pd.DataFrame(splits).to_csv(out/'split_manifest.csv',index=False)
    pd.concat(fits).to_csv(out/'langmuir_fit_audit.csv',index=False)
    pd.DataFrame(cvmets,columns=['protocol','fold','model','support','n_total','n','R2','RMSE','MAE']).to_csv(out/'run1_protocol_metrics.csv',index=False)
    if cvpred:pd.concat(cvpred).to_csv(out/'run1_protocol_predictions.csv',index=False)
    elif (out/'run1_protocol_predictions.csv').exists():(out/'run1_protocol_predictions.csv').unlink()
    with pd.ExcelWriter(out/'train_test_predictions.xlsx',engine='openpyxl') as xl:
        predictions.to_excel(xl,sheet_name='All_predictions',index=False)
        for name,g in predictions.groupby('protocol',sort=False):g.to_excel(xl,sheet_name=name[:31],index=False)
        metrics.to_excel(xl,sheet_name='Metrics',index=False,na_rep='N/A')
        predictions[predictions.Langmuir_eligible].to_excel(xl,sheet_name='Langmuir_matched_rows',index=False)
        pd.DataFrame(cvmets).to_excel(xl,sheet_name='CV_metrics',index=False,na_rep='N/A')
        pd.DataFrame(splits).to_excel(xl,sheet_name='Splits',index=False)
        pd.concat(fits).to_excel(xl,sheet_name='Langmuir_fits',index=False)
        if cvpred:pd.concat(cvpred).to_excel(xl,sheet_name='CV_test_predictions',index=False)
        for ws in xl.book.worksheets:
            ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
            for cells in ws.columns:ws.column_dimensions[cells[0].column_letter].width=min(45,max(14,len(str(cells[0].value))+2))
    env={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','scipy','tabpfn','xgboost','optuna']}
    (out/'environment_and_parameters.json').write_text(json.dumps(dict(packages=env,settings=settings,tabpfn_config=ns['TABPFN_CONFIG']),indent=2,default=str))
    from plot_manuscript import plot_all
    plot_all(out)
    return [pd.DataFrame(cvmets)],cvpred
