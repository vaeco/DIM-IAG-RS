import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print("repo root:", ROOT)

from dim_iag_rs.dim import DIM
from dim_iag_rs.dim_wrr import DIM_WRR
from dim_iag_rs.dim_wrr_iag import DIM_WRR_IAG, load_iag_checkpoint

print("Import OK: DIM, DIM_WRR, DIM_WRR_IAG")
