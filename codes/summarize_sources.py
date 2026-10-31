"""Generate dataset summary and completed Table 4 from all retained observations.
Rank display is user-requested terminology, not a new scientific classification.
"""
from pathlib import Path
import pandas as pd
from prepare_dataset import ROOT, P, T
DISPLAY={'Lignite':'Lignite','Bituminous':'Bituminous','Subbituminous':'Semi-bituminous','Anthracitic':'Anthracite'}
NOTE=('Rank terminology: Semi-bituminous is the requested display label for records supplied as Sub-Bi (previously expanded to Subbituminous). This display mapping is not a verified reclassification. Original labels and rank provenance remain in model_dataset.csv. An asterisk marks provisional composition-based estimates (Hu: 33 rows; Li: 60 rows). Ranks are not model predictors. Author strings and DOI links are retained as supplied; publication-year discrepancies have not been resolved. Ranges are the compiled dataset ranges, not necessarily full experimental ranges in the source publications.')
def span(v):return f'{v.min():g}–{v.max():g}'
def markdown(d):
    return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in d.itertuples(index=False,name=None))
def summarize():
    d=pd.read_csv(ROOT/'data/model_dataset.csv');d['display_rank']=d.rank_assigned.map(DISPLAY)
    assert d.display_rank.notna().all() and d.display_rank.nunique()==4
    rows=[]
    for (rank,author),g in d.groupby(['display_rank','Author'],sort=False):
        estimated=g.rank_status.str.contains('provisional').any()
        rows.append({'Coal rank':rank+('*' if estimated else ''),'P range (MPa)':span(g[P]),'T range (K)':span(g[T]),'Authors':author})
    summary=pd.DataFrame(rows)
    inventory=[]
    for source,g in d.groupby('source_key',sort=False):
        ranks=[]
        for rank,z in g.groupby('display_rank',sort=False):ranks.append(rank+('*' if z.rank_status.str.contains('provisional').any() else ''))
        inventory.append({'Authors':'; '.join(g.Author.unique()),'Rows':len(g),'Coal rank':'; '.join(ranks),'P range (MPa)':span(g[P]),'T range (K)':span(g[T]),'Composition groups':g.sample_id.nunique(),'Isotherms':g.isotherm_id.nunique(),'Source DOI/link':source})
    inventory=pd.DataFrame(inventory)
    out=ROOT/'tables';out.mkdir(exist_ok=True)
    for title,df in [('Dataset_summary',summary),('Table_4_Source_inventory',inventory)]:
        df.to_csv(out/(title+'.csv'),index=False)
        (out/(title+'.md')).write_text(title.replace('_',' ')+ '\n\n'+markdown(df)+'\n\n'+NOTE+'\n')
    assert inventory.Rows.sum()==781
    return summary,inventory
if __name__=='__main__':summarize()
