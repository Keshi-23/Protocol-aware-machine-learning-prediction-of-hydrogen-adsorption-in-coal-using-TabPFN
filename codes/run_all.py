"""Run preparation, Excel audit, data plots and the complete modeling workflow."""
from pathlib import Path
import sys,subprocess
root=Path(__file__).resolve().parent
for name in ['audit_splits.py','summarize_sources.py','export_data_audit.py','plot_preparation.py','tabpfn_h2_coal_revised.py','plot_manuscript.py']:
    subprocess.run([sys.executable,str(root/name)],cwd=root,check=True)
