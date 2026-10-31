"""Validate the revised subset and split design without installing TabPFN or XGB."""
from pathlib import Path
import pandas as pd
import numpy as np
from prepare_dataset import prepare,ROOT,Y
from protocols import make_splits,langmuir_fit,score

def audit():
 d,_=prepare();rows=[];pred=[];logs=[]
 for p,tr,te,group in make_splits(d):
  per,pooled,log=langmuir_fit(d,tr);log.insert(0,'protocol',p);logs.append(log)
  rows.append(dict(protocol=p,n_train=len(tr),n_test=len(te),test_fraction=len(te)/len(d),
   per_isotherm_train_covered=int(np.isfinite(per[tr]).sum()),per_isotherm_test_covered=int(np.isfinite(per[te]).sum()),
   isotherm_overlap=len(set(d.iloc[tr].isotherm_id)&set(d.iloc[te].isotherm_id)),
   composition_overlap=len(set(d.iloc[tr].sample_id)&set(d.iloc[te].sample_id)),
   source_label_overlap=len(set(d.iloc[tr].Author)&set(d.iloc[te].Author)),
   source_doi_overlap=len(set(d.iloc[tr].source_key)&set(d.iloc[te].source_key))))
  for sub,ii in [('train',tr),('test',te)]:
   z=d.iloc[ii][['row_model','row_idx','original_excel_row','Author','sample_id','isotherm_id',Y]].copy().rename(columns={Y:'measured'})
   z.insert(0,'subset',sub);z.insert(0,'protocol',p);z['Langmuir_per_isotherm']=per[ii];z['Langmuir_pooled']=pooled[ii];z['Langmuir_eligible']=np.isfinite(per[ii]);pred.append(z)
 summary=pd.DataFrame(rows);summary.to_csv(ROOT/'data/split_design_audit.csv',index=False)
 pd.concat(pred).to_csv(ROOT/'data/langmuir_train_test_audit.csv',index=False)
 pd.concat(logs).to_csv(ROOT/'data/langmuir_fit_audit.csv',index=False)
 print(summary.to_string(index=False))
if __name__=='__main__':audit()
