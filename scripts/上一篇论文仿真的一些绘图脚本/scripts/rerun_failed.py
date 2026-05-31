#!/usr/bin/env python3
"""
补充运行失败的特定实验组合。
"""
import os
import sys
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从 run_batch.py 复制的超参数，只包含失败实验相关的算法
ALGORITHM_HYPERPARAMETERS = {
    "h_mappo_l": {
        "agent.num_steps": 200,
        "agent.learning_rate": 1e-4,
        "agent.clip_coef": 0.12,
        "agent.ent_coef": 0.01,
        "agent.num_epochs": 3,
        "agent.num_minibatches": 8,
        "agent.lagrangian_lr": 0.01,
        "agent.lagrangian_init": 0.1,
        "agent.sac_updates_per_epoch": 16,
        "agent.sac_batch_size": 512,
        "agent.sac_target_entropy_ratio": 0.55,
        "agent.deploy_ent_coef": 0.12,
        "weights.deployment_migration_weight": 0.02,
        "agent.sampling_temperature": 1.0,
        "agent.size_bias_beta": 0.0,
    },
    "mappo_no_constraint": {
        "agent.num_steps": 200,
        "agent.learning_rate": 1e-4,
        "agent.clip_coef": 0.12,
        "agent.ent_coef": 0.05,
        "agent.num_epochs": 4,
        "agent.num_minibatches": 8,
        "agent.sac_updates_per_epoch": 20,
        "agent.sac_batch_size": 512,
        "agent.sac_target_entropy_ratio": 0.50,
        "agent.deploy_ent_coef": 0.08,
        "weights.deployment_migration_weight": 0.02,
        "agent.sampling_temperature": 1.0,
        "agent.size_bias_beta": 0.0,
    },
}

# 从 sensitivity.yaml 复制的固定参数
FIXED_PARAMS = {
    "communication": {
        "shadowing_stdev": 8.0,
        "reference_pathloss_db": 30.0,
        "receiver_noise_figure_dB": 6.0,
        "carrier_frequency_Hz": 3500000000,
    },
    "user": {
        "zipf_param": 0.8,
        "epsilon_range": [1e-11, 1e-9],
        "tx_power_dBm_range": [20.0, 30.0],
    },
    "weights": {
        "deployment_hit_weight": 1.0,
        "deployment_migration_weight": 0.1,
    },
}

# 需要重新运行的失败实验列表
FAILED_EXPERIMENTS = [
    # {"algorithm": "mappo_no_constraint", "seed": 0, "energy_weight": 2.5},
    {"algorithm": "h_mappo_l", "seed": 10, "energy_weight": 3.0},
    {"algorithm": "mappo_no_constraint", "seed": 10, "energy_weight": 3.0},
    {"algorithm": "h_mappo_l", "seed": 1, "energy_weight": 7.0},
]

# 固定的实验设置
MAX_UPDATES = 1500
RESULTS_BASE_DIR = "experiment_result_sensitivity"
SENSITIVITY_EXPERIMENT_NAME = "energy_privacy_weights"

def build_config_overrides(param_overrides=None):
    """构建配置覆盖JSON字符串"""
    config_dict = {}
    # 添加固定参数
    for section, params in FIXED_PARAMS.items():
        for key, value in params.items():
            config_dict[f"{section}.{key}"] = value
    
    # 添加算法超参数和扫描参数
    if param_overrides:
        config_dict.update(param_overrides)
    
    return json.dumps(config_dict)

def run_single_rerun(experiment_details):
    """运行单个补充实验"""
    algorithm = experiment_details["algorithm"]
    seed = experiment_details["seed"]
    energy_weight = experiment_details["energy_weight"]

    try:
        # 构建配置覆盖
        algo_hps = ALGORITHM_HYPERPARAMETERS.get(algorithm, {})
        param_overrides = {**algo_hps, "weights.energy_weight": energy_weight}
        config_overrides_json = build_config_overrides(param_overrides)

        # 构建实验名称，与 run_batch.py 保持一致
        experiment_name = f"sensitivity_{SENSITIVITY_EXPERIMENT_NAME}_{energy_weight}"

        # 构建命令
        cmd = [
            sys.executable, "main.py",
            "--agent", algorithm,
            "--seed", str(seed),
            "--max-updates", str(MAX_UPDATES),
            "--results-base-dir", RESULTS_BASE_DIR,
            "--experiment-name", experiment_name,
            "--config-overrides", config_overrides_json
        ]

        # 设置运行环境，与 run_batch.py 保持一致
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONIOENCODING"] = "utf-8"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
        env["TORCH_NUM_THREADS"] = "1"

        print(f"🚀 启动补充实验: {algorithm} (种子: {seed}, energy_weight: {energy_weight})")

        # 运行子进程
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=env
        )

        if result.returncode == 0:
            print(f"✅ 完成补充实验: {algorithm} (种子: {seed}, energy_weight: {energy_weight})")
            return {"status": "success", **experiment_details}
        else:
            print(f"❌ 补充实验失败: {algorithm} (种子: {seed}, energy_weight: {energy_weight})")
            print(f"错误输出:\n{result.stderr}")
            return {"status": "failed", "stderr": result.stderr, **experiment_details}

    except Exception as e:
        print(f"❌ 补充实验异常: {algorithm} (种子: {seed}, energy_weight: {energy_weight}) - {str(e)}")
        return {"status": "error", "error": str(e), **experiment_details}

def main():
    """主函数，并行执行所有失败的实验"""
    # 同时运行所有4个任务
    max_workers = len(FAILED_EXPERIMENTS)
    print(f"🔧 开始补充 {len(FAILED_EXPERIMENTS)} 个失败的实验，并行度: {max_workers}")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_rerun, task) for task in FAILED_EXPERIMENTS]
        
        # 使用tqdm显示进度
        for future in tqdm(as_completed(futures), total=len(futures), desc="Rerunning progress", ncols=100):
            result = future.result()
            results.append(result)

    # 统计结果
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = len(results) - success_count

    print(f"\n📈 补充实验完成:")
    print(f"   成功: {success_count}/{len(results)}")
    print(f"   失败: {failed_count}/{len(results)}")

    # 如果有失败的实验，打印详细信息
    if failed_count > 0:
        print("\n--- 失败的实验详情 ---")
        for r in results:
            if r['status'] != 'success':
                print(f"  - 实验: {r['algorithm']} (种子: {r['seed']}, energy_weight: {r['energy_weight']})")
                if r['status'] == 'failed':
                    print(f"    状态: 失败\n    错误输出:\n{r['stderr']}")
                elif r['status'] == 'error':
                    print(f"    状态: 异常\n    错误信息: {r['error']}")
        print("------------------------")

    print(f"💾 结果已保存在 {RESULTS_BASE_DIR}/ 目录下")

if __name__ == "__main__":
    main()
