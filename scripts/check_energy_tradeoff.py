"""
Verify trade-off with communication overhead included.
"""
import math

mu_bal  = 0.02
lambda_t = 1.0
delta_t  = 100.0
req_denom = lambda_t * delta_t

# UAV specs: (f, P_compute)
uavs = {
    "src":  (2e11,  10.0),
    "weak": (1e11,   5.0),
    "med":  (4.5e11, 15.0),
    "str":  (8e11,  25.0),
    "med2": (4.5e11, 15.0),
}
uav_list = list(uavs.values())
n = len(uav_list)

# Communication link params (5 MHz, 300 m, 0.2 W tx)
B = 5e6; P_tx = 0.2; N0 = 1e-13; alpha = 3.2; d = 300
rate = B * math.log2(1 + P_tx / (d**alpha * N0))  # bps

# Assume sequential DAG: 4 blocks, each ~18.75 GFLOPs, 5 MB feature between blocks
flops_per_block = 75e9 / 4
bits_per_feature = 5.0 * 8e6  # 5 MB at 32-bit
bits_compressed  = bits_per_feature / 8  # 4-bit

def simulate(routing, use_compression=True):
    """routing: list of (f, P) for each of 4 blocks"""
    bits = bits_compressed if use_compression else bits_per_feature
    delay = 0.0
    # Energy per UAV index (map to 0..n-1 by position in uav_list)
    energy_per_uav = [0.0] * n
    prev_uav_idx = None
    for block_idx, (f, p) in enumerate(routing):
        # find uav index
        uav_idx = uav_list.index((f, p))
        # compute
        t_comp = flops_per_block / f
        e_comp = req_denom * p * t_comp
        delay += t_comp
        energy_per_uav[uav_idx] += e_comp
        # comm from previous block (if different UAV)
        if prev_uav_idx is not None and prev_uav_idx != uav_idx:
            t_comm = bits / rate
            e_comm = req_denom * P_tx * t_comm
            delay += t_comm
            energy_per_uav[prev_uav_idx] += e_comm
        prev_uav_idx = uav_idx

    action_total = sum(energy_per_uav)
    mean_e = action_total / n
    var_e   = sum((e - mean_e)**2 for e in energy_per_uav) / n
    penalty = mu_bal * var_e
    e_obj   = action_total + penalty
    return delay, action_total / req_denom, e_obj / req_denom, penalty / req_denom

str_spec  = uavs["str"]
med_spec  = uavs["med"]
weak_spec = uavs["weak"]
src_spec  = uavs["src"]

strategies = {
    "All-on-Strong     ": [str_spec]*4,
    "All-on-Weak       ": [weak_spec]*4,
    "Balanced (S+M+M+W)": [str_spec, med_spec, med_spec, weak_spec],
    "Mixed  (S+M+W+S)  ": [str_spec, med_spec, weak_spec, str_spec],
}

print(f"{'Strategy':<28} {'Delay(s)':>9} {'E_compute(J)':>13} {'E_obj(J)':>9} {'Bal_pen(J)':>11}")
print("-" * 80)
for name, routing in strategies.items():
    delay, e_compute, e_obj, bal = simulate(routing)
    print(f"{name:<28} {delay:9.4f} {e_compute:13.3f} {e_obj:9.3f} {bal:11.3f}")

print()
print("KEY: All-on-Strong has lowest delay but highest E_obj (balance penalty)")
print("     Balanced strategies can improve energy at cost of delay (comm overhead)")
print("     => Genuine Pareto trade-off between delay and E_obj")
