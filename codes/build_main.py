"""Generate the updated main script, preserving RUN 2–6 from the attachment."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
s=(ROOT/'original_supplied_script.py').read_text()
s=s.replace('evaluation under three splitting protocols.','revised-data evaluation under four splitting protocols.')
s=s.replace('781 describes the number of pressure steps.','The selected observation count is reported in preparation_summary.json.')
s=s.replace('import os\n','import os\nfrom pathlib import Path\nfrom prepare_dataset import prepare\nprepare()\nPACKAGE_ROOT = Path(__file__).resolve().parent\n',1)
s=s.replace("FILE_PATH  = 'h2coal_dataset.xlsx'", "FILE_PATH  = str(PACKAGE_ROOT / 'data/model_input.csv')")
s=s.replace("OUT_DIR    = './results_protocols'", "OUT_DIR    = str(PACKAGE_ROOT / 'results_revised')\nRUN_1_CV = True  # five-fold supplement; grouped folds need not contain exactly 20% rows")
s=s.replace('pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)','pd.read_csv(FILE_PATH)')
s=s.replace('MODELS        = ["TabPFN", "XGB", "RF", "MLP"]','MODELS        = ["TabPFN", "XGB"]')
s=s.replace('"XGB": None, "RF": None','"XGB": "minmax", "RF": None')
s=s.replace('MC_PRESSURES    = [0.1, 1.0, 5.0, 10.0, 15.0, 18.0]','MC_PRESSURES    = [0.1, 1.0, 5.0, 10.0, 15.0, 20.0]')
s=s.replace('# ── feature matrix ─', '# ── feature matrix ─', 1)
anchor='# ── feature matrix '
pos=s.index(anchor)
s=s[:pos]+"""# Use audited source-aware isotherms in all downstream analyses.
_prepared=pd.read_csv(PACKAGE_ROOT / 'data/model_dataset.csv')
assert len(_prepared)==len(df_raw)
for _group in ['sample_id','isotherm_id']:
    df_raw[_group]=_prepared[_group].to_numpy()
n_samples=df_raw['sample_id'].nunique()
source_id=pd.factorize(_prepared['source_key'])[0]
df_raw['source_id']=source_id

"""+s[pos:]
a=s.index('def langmuir_random_split(');b=s.index('# ═════════════════════════════════════════════════════════════════\n#  RUN 2',a)
s=s[:a]+'''# RUN 1 is replaced by matched 80:20 evaluations and optional five-fold CV.
# A per-isotherm Langmuir fit and a pooled Langmuir fit are different baselines.
# Entirely unseen isotherms have no per-isotherm parameter estimates.
protocol_results, protocol_preds = [], []
if RUN_1_PROTOCOLS:
    from protocols import run
    protocol_results, protocol_preds = run(globals())


'''+s[b:]
s='''# Updated revised-data analysis. See README.md for selection and benchmark limits.
# Remaining RUN 2–6 calculations are retained from the supplied script.
'''+s
s+='''\n# Regenerate the additional manuscript panels after all original runs finish.
from plot_manuscript import plot_all
plot_all(Path(OUT_DIR))
'''
compile(s,'tabpfn_h2_coal_revised.py','exec')
(ROOT/'tabpfn_h2_coal_revised.py').write_text(s)
if __name__=='__main__':print('Updated main script generated.')
