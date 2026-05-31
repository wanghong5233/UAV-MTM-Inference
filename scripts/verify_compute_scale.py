"""Quick sanity check: print delay/rho breakdown for MTAN and Split under new compute scale."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.utils.config_loader import load_config
from src.env.uav_env import UAVEnv

CONFIGS = [
    ("MTAN",  "configs/experiments/main_gnn_ppo_tmc_stable.yaml"),
    ("Split", "configs/experiments/main_gnn_ppo_tmc_stable_split.yaml"),
]

for model_name, cfg_file in CONFIGS:
    cfg  = load_config(cfg_file)
    env  = UAVEnv(cfg)
    env.reset(seed=42)

    f         = env.uav_compute          # FLOPs/s per UAV
    total_F   = float(np.sum(env.model_graph.node_flops))
    lam       = env.arrival_rate
    chi       = env.chi
    f_mean    = float(np.mean(f))
    f_strong  = float(np.max(f))
    f_weak    = float(np.min(f))

    # rho if everything on one (mean) UAV, avg_pi ≈ 2/3
    avg_pi    = 2.0 / 3.0
    rho_mean  = lam * total_F * avg_pi / f_mean
    delay_local_mean   = (total_F / f_mean)   * (1.0 + chi * rho_mean)
    rho_strong = lam * total_F * avg_pi / f_strong
    delay_local_strong = (total_F / f_strong) * (1.0 + chi * rho_strong)

    max_feat  = float(np.max(env.model_graph.node_out_bytes))
    bw_bps    = 200e6   # 200 Mbps representative link
    trans_full = max_feat * 8.0 / bw_bps
    trans_4bit = (max_feat / 8.0) * 8.0 / bw_bps

    print(f"=== {model_name} ===")
    print(f"  Total FLOPs       : {total_F/1e9:7.1f} GFLOPs")
    print(f"  f_u (GFLOPs/s)    : weak={f_weak/1e9:.0f}  mean={f_mean/1e9:.0f}  strong={f_strong/1e9:.0f}")
    print(f"  rho (all-local, mean UAV): {rho_mean:.4f}")
    print(f"  proc_delay local  (mean  UAV): {delay_local_mean:.3f} s")
    print(f"  proc_delay local  (strong UAV): {delay_local_strong:.3f} s")
    print(f"  Max feature size  : {max_feat/1e6:.2f} MB")
    print(f"  Trans delay full  (@200 Mbps): {trans_full:.3f} s")
    print(f"  Trans delay 4-bit (@200 Mbps): {trans_4bit:.4f} s")
    print(f"  tau_ref in config : {cfg.get('reward',{}).get('tau_ref','?')}")
    print()
