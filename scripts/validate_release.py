"""Check presence and strict-load a supplied single-gate checkpoint only."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ckpt',default='weights/dim_iag_rs_best.pth')
    a=p.parse_args()
    if not Path(a.ckpt).is_file():raise SystemExit('Missing original trained single-gate checkpoint; not a reproduced release.')
    from dim_iag_rs.dim_wrr_iag import DIM_WRR_IAG,load_iag_checkpoint
    load_iag_checkpoint(DIM_WRR_IAG(),a.ckpt,map_location='cpu')
    print('Checkpoint load completed; this is not downstream performance verification.')
if __name__=='__main__':main()
