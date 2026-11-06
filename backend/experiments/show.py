import json, sys
for line in open("experiments/results.jsonl"):
    r = json.loads(line)
    print(f"{r['tag']:24s} oosQ={r['oos20_qlike']:.4f} oosR={r['oos20_rmse']:.4f} fullQ={r['full_qlike']:.4f} ep={r['epochs']} extra={r.get('inner_val_qlike','')}")
