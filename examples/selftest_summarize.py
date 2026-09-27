#!/usr/bin/env python3
"""Synthetic analyzer self-test only. NOT a measured #47922 result."""
import json, subprocess, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent

def rec(arm, repeat, cap, final_hits):
    events=[]
    for i,label in enumerate(["seed","churn_01","seed_replay"]):
        hits=final_hits if label=="seed_replay" else 0
        events.append({
            "event_index":i,"label":label,"prompt_sha256":f"h-{label}",
            "control":{"output_token_control":"PASS"},
            "metrics":{"delta":{"prefix_cache_queries":896,"prefix_cache_hits":hits,"num_preemptions":0,"prompt_tokens":896},
                       "after":{"kv_cache_usage_perc":0.5 if arm.startswith("A") else 0.9,"num_requests_running":0,"num_requests_waiting":0},
                       "prefix_hit_ratio":hits/896},
        })
    return {"schema_version":"vpp.scheduler-kv/0.1","profile":"synthetic-selftest","source":{"collector":"synthetic-selftest","upstream_pr":47922},
            "arm":arm,"repeat":repeat,"runtime":{"model":"synthetic","kv_cache_memory_bytes":cap},"workload":{"sha256":"same-workload"},
            "environment":{"git_head":"synthetic-sha"},"events":events,"control":{"output_token_control":"PASS","required_prefix_metrics":"PASS"},"status":"PASS"}

with tempfile.TemporaryDirectory() as td:
    td=Path(td); evidence=td/"evidence"; out=td/"out"; evidence.mkdir()
    for r in range(3):
        (evidence/f"A_r{r}.json").write_text(json.dumps(rec("A_high_capacity",r,16*1024**3,880)))
        (evidence/f"B_r{r}.json").write_text(json.dumps(rec("B_low_capacity",r,2*1024**3,0)))
    cp=subprocess.run([sys.executable,str(HERE/"summarize.py"),str(evidence),"--out-dir",str(out),"--expected-repeats","3"],check=True,capture_output=True,text=True)
    s=json.loads((out/"SUMMARY.json").read_text())
    assert s["arms"]["A_high_capacity"]["repeat_stability"]["status"]=="PASS"
    assert s["arms"]["B_low_capacity"]["repeat_stability"]["status"]=="PASS"
    assert s["comparison"]["output_token_control"]=="PASS"
    assert s["comparison"]["first_observable_cross_arm_divergence"]["label"]=="seed_replay"
    print("synthetic summarize self-test: PASS")
    print(cp.stdout)
