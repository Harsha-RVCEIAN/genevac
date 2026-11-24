"""Inventory and prepare one trajectory dataset source."""
import argparse, json
from genevac.prediction.dataset_pipeline import inventory_datasets, load_eth_ucy, load_sdd_pickle, export_dataset

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root",default="data/raw"); p.add_argument("--output",default="data/processed/datasets")
    p.add_argument("--dataset",choices=("ETH","HOTEL","UNIV","ZARA1","ZARA2","SDD"),default="ETH")
    p.add_argument("--file"); p.add_argument("--scene",default="default"); p.add_argument("--inventory-only",action="store_true")
    p.add_argument("--trust-pickle",action="store_true")
    p.add_argument("--seconds-per-source-frame",type=float,default=.04)
    a=p.parse_args()
    if a.inventory_only:
        print(json.dumps(inventory_datasets(a.raw_root),indent=2)); return 0
    if not a.file: p.error("--file is required unless --inventory-only is used")
    tracks=load_sdd_pickle(a.file,trusted_pickle=a.trust_pickle) if a.dataset=="SDD" else load_eth_ucy(a.file,seconds_per_source_frame=a.seconds_per_source_frame)
    result=export_dataset(tracks,a.output,dataset=a.dataset,scene=a.scene,source=a.file)
    print(json.dumps(result,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
