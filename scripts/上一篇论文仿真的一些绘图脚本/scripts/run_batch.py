#!/usr/bin/env python3
"""
批量实验运行脚本
支持并行执行多算法、多种子、多参数组合的实验
"""

import os
import sys
import yaml
import json
import subprocess
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# 最终调优的算法超参数
# 从 scripts/test_algorithms.py 中迁移
# =============================================================================
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
    # ippo刻意比较差
    # "ippo": {
    #     "agent.num_steps": 200,
    #     "agent.learning_rate": 2e-4,
    #     "agent.clip_coef": 0.12,
    #     "agent.ent_coef": 0.08,
    #     "agent.num_epochs": 4,
    #     "agent.num_minibatches": 8,
    #     "agent.lagrangian_lr": 0.02,
    #     "agent.sac_updates_per_epoch": 10,
    #     "agent.sac_batch_size": 512,
    #     "agent.sac_target_entropy_ratio": 0.75,
    #     "agent.deploy_ent_coef": 0.20,
    #     "weights.deployment_migration_weight": 0.02,
    #     "agent.sampling_temperature": 1.2,
    #     "agent.size_bias_beta": 0.1,
    # },
    "ippo": {
        "agent.num_steps": 200,
        "agent.learning_rate": 2e-4,
        "agent.clip_coef": 0.12,
        "agent.ent_coef": 0.05,
        "agent.num_epochs": 3,
        "agent.num_minibatches": 8,
        "agent.lagrangian_lr": 0,
        "agent.sac_updates_per_epoch": 16,
        "agent.sac_batch_size": 512,
        "agent.sac_target_entropy_ratio": 0.55,
        "agent.deploy_ent_coef": 0.12,
        "weights.deployment_migration_weight": 0.02,
        "agent.sampling_temperature": 1.0,
        "agent.size_bias_beta": 0.1,
        # 学习率调度（仅用户层）
        "agent.lr_scheduler": "cosine",
        "agent.lr_end_factor": 0.1,
        "agent.lr_T_max": 500,
    },
    # "hc_ippo_l": {
    #     "agent.num_steps": 200,
    #     "agent.learning_rate": 2e-4,
    #     "agent.clip_coef": 0.12,
    #     "agent.ent_coef": 0.08,
    #     "agent.num_epochs": 4,
    #     "agent.num_minibatches": 8,
    #     # 覆盖：根据验证命令设置 Lagrangian 与成本评论者/调度器参数
    #     "agent.lagrangian_lr": 0.02,
    #     # "agent.lagrangian_init": 0.1,
    #     # "agent.lagrangian_init": 20.0,
    #     # "agent.cost_vf_coef": 0.5,
    #     # "agent.normalize_cost_advantages": True,
    #     # 学习率调度（仅用户层）
    #     # "agent.lr_scheduler": "cosine",
    #     # "agent.lr_end_factor": 0.1,
    #     # "agent.lr_T_max": 280,
    #     # 其他保持原默认
    #     "agent.sac_updates_per_epoch": 16,
    #     "agent.sac_batch_size": 512,
    #     "agent.sac_target_entropy_ratio": 0.55,
    #     "agent.deploy_ent_coef": 0.12,
    #     "weights.deployment_migration_weight": 0.02,
    #     "agent.sampling_temperature": 1.0,
    #     "agent.size_bias_beta": 0.0,
    # },
    "hc_ippo_l": {
        "agent.num_steps": 200,
        "agent.learning_rate": 2e-4,
        "agent.clip_coef": 0.12,
        "agent.ent_coef": 0.05,
        "agent.num_epochs": 3,
        "agent.num_minibatches": 8,
        # 覆盖：根据验证命令设置 Lagrangian 与成本评论者/调度器参数
        "agent.lagrangian_lr": 0.01,
        # "agent.lagrangian_init": 5,
        # "agent.lagrangian_init": 5.0,
        "agent.lagrangian_init": 0.1,
        "agent.cost_vf_coef": 0.5,
        "agent.normalize_cost_advantages": True,
        # 学习率调度（仅用户层）
        "agent.lr_scheduler": "cosine",
        "agent.lr_end_factor": 0.1,
        "agent.lr_T_max": 500,
        # 其他保持原默认
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
    "lru_avg": {
        "agent.num_steps": 200,
        "agent.clip_coef": 0.12,
        "agent.ent_coef": 0.020,
        "agent.num_epochs": 3,
        "agent.num_minibatches": 8,
    }
}
# =============================================================================

def load_experiment_config(config_path):
    """加载实验配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def build_config_overrides(fixed_params, param_overrides=None):
    """构建配置覆盖字符串"""
    config_dict = {}
    
    # 添加固定参数
    if fixed_params:
        for section, params in fixed_params.items():
            for key, value in params.items():
                config_dict[f"{section}.{key}"] = value
    
    # 添加参数覆盖
    if param_overrides:
        config_dict.update(param_overrides)
    
    return json.dumps(config_dict)

def run_single_experiment(experiment_args):
    """运行单个实验"""
    try:
        algorithm, seed, experiment_name, config_overrides, max_updates, threads_per_proc, results_base_dir_override = experiment_args

        # 根据实验名称确定结果目录，允许外部覆盖
        if results_base_dir_override:
            results_base_dir = results_base_dir_override
        else:
            if experiment_name.startswith('convergence'):
                results_base_dir = 'experiment_result_convergence'
            elif experiment_name.startswith('sensitivity'):
                results_base_dir = 'experiment_result_sensitivity'
            elif experiment_name.startswith('scalability'):
                results_base_dir = 'experiment_result_scalability'
            else:
                results_base_dir = 'experiment_result'

        # 构建命令
        cmd = [
            sys.executable, "main.py",
            "--agent", algorithm,
            "--seed", str(seed),
            "--experiment-name", experiment_name,
            "--max-updates", str(max_updates),
            "--results-base-dir", results_base_dir,
            "--config-overrides", config_overrides
        ]
        # 运行环境：强制CPU并统一UTF-8编码，避免Windows GBK导致子进程UnicodeEncodeError
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONIOENCODING"] = "utf-8"
        # 限制每个子进程的BLAS/数值库/Torch线程数，避免过度并行导致争用
        env["OMP_NUM_THREADS"] = str(threads_per_proc)
        env["MKL_NUM_THREADS"] = str(threads_per_proc)
        env["OPENBLAS_NUM_THREADS"] = str(threads_per_proc)
        env["NUMEXPR_NUM_THREADS"] = str(threads_per_proc)
        env["TORCH_NUM_THREADS"] = str(threads_per_proc)
        
        print(f"🚀 启动实验: {algorithm} (种子: {seed}, 线程/进程: {threads_per_proc})")
        
        # 运行实验
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # 避免Windows下子进程输出包含非UTF-8字节导致解码报错
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=env
        )
        
        if result.returncode == 0:
            print(f"✅ 完成实验: {algorithm} (种子: {seed})")
            return {
                "status": "success",
                "algorithm": algorithm,
                "seed": seed,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            print(f"❌ 实验失败: {algorithm} (种子: {seed})")
            print(f"错误输出: {result.stderr}")
            return {
                "status": "failed",
                "algorithm": algorithm,
                "seed": seed,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
    
    except Exception as e:
        print(f"❌ 实验异常: {algorithm} (种子: {seed}) - {str(e)}")
        return {
            "status": "error",
            "algorithm": algorithm,
            "seed": seed,
            "error": str(e)
        }

def run_convergence_experiment(config, max_workers=2, threads_per_proc=1, results_base_dir_override=None):
    """运行收敛性实验"""
    experiment_name = config['experiment_name']
    algorithms = config['algorithms']
    seeds = config['seeds']
    max_updates = config['max_updates']
    fixed_params = config.get('fixed_params', {})
    
    # 构建实验任务列表
    tasks = []
    for algorithm in algorithms:
        for seed in seeds:
            # 获取算法特定的超参数
            algo_hps = ALGORITHM_HYPERPARAMETERS.get(algorithm, {})
            config_overrides = build_config_overrides(fixed_params, algo_hps)
            tasks.append((algorithm, seed, experiment_name, config_overrides, max_updates, threads_per_proc, results_base_dir_override))
    
    print(f"📊 开始收敛性实验: {len(tasks)} 个任务")
    print(f"   算法: {algorithms}")
    print(f"   种子: {seeds}")
    print(f"   并行度: {max_workers}")
    print(f"   每进程线程数: {threads_per_proc}")
    
    # 并行执行
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_experiment, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Batch progress", ncols=100):
            result = future.result()
            results.append(result)
    
    # 统计结果
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = len(results) - success_count
    
    print(f"\n📈 收敛性实验完成:")
    print(f"   成功: {success_count}/{len(results)}")
    print(f"   失败: {failed_count}/{len(results)}")
    
    # 如果有失败的实验，打印详细信息
    if failed_count > 0:
        print("\n--- 失败的实验详情 ---")
        for r in results:
            if r['status'] != 'success':
                print(f"  - 算法: {r['algorithm']}, 种子: {r['seed']}")
                if r['status'] == 'failed':
                    print(f"    状态: 失败\n    错误输出:\n{r['stderr']}")
                elif r['status'] == 'error':
                    print(f"    状态: 异常\n    错误信息: {r['error']}")
        print("------------------------")
    
    return results

def run_scalability_experiment(config, max_workers=2, threads_per_proc=1, results_base_dir_override=None):
    """运行扩展性实验"""
    experiment_name = config['experiment_name']
    algorithms = config['algorithms']
    seeds = config['seeds']
    max_updates = config['max_updates']
    fixed_params = config.get('fixed_params', {})
    
    # 参数扫描
    param_sweep = config['parameter_sweep']
    param_name = param_sweep['parameter']
    param_values = param_sweep['values']
    
    # 构建实验任务列表
    tasks = []
    task_info = []  # 用于关联结果与参数
    for param_value in param_values:
        for algorithm in algorithms:
            for seed in seeds:
                # 合并扫描参数和算法特定超参数（扫描参数优先）
                algo_hps = ALGORITHM_HYPERPARAMETERS.get(algorithm, {})
                param_overrides = {**algo_hps, param_name: param_value}
                config_overrides = build_config_overrides(fixed_params, param_overrides)
                
                # 添加参数值到实验名称
                exp_name = f"{experiment_name}_{param_name.split('.')[-1]}_{param_value}"
                task = (algorithm, seed, exp_name, config_overrides, max_updates, threads_per_proc, results_base_dir_override)
                tasks.append(task)
                task_info.append({'param_name': param_name, 'param_value': param_value})
    
    print(f"📊 开始扩展性实验: {len(tasks)} 个任务")
    print(f"   参数扫描: {param_name} = {param_values}")
    print(f"   算法: {algorithms}")
    print(f"   种子: {seeds}")
    print(f"   并行度: {max_workers}")
    print(f"   每进程线程数: {threads_per_proc}")
    
    # 并行执行
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 使用字典将 future 映射到其任务索引，以便高效查找
        futures = {executor.submit(run_single_experiment, task): i for i, task in enumerate(tasks)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Batch progress", ncols=100):
            result = future.result()
            # 关联参数信息到结果
            task_index = futures[future]
            result['param_name'] = task_info[task_index]['param_name']
            result['param_value'] = task_info[task_index]['param_value']
            results.append(result)
    
    # 统计结果
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = len(results) - success_count
    
    print(f"\n📈 扩展性实验完成:")
    print(f"   成功: {success_count}/{len(results)}")
    print(f"   失败: {failed_count}/{len(results)}")
    
    # 如果有失败的实验，打印详细信息
    if failed_count > 0:
        print("\n--- 失败的实验详情 ---")
        for r in results:
            if r['status'] != 'success':
                print(f"  - 算法: {r['algorithm']}, 种子: {r['seed']}, 参数: {r['param_name']} = {r['param_value']}")
                if r['status'] == 'failed':
                    print(f"    状态: 失败\n    错误输出:\n{r['stderr']}")
                elif r['status'] == 'error':
                    print(f"    状态: 异常\n    错误信息: {r['error']}")
        print("------------------------")
    
    return results

def run_sensitivity_experiment(config, max_workers=2, threads_per_proc=1, results_base_dir_override=None):
    """运行敏感性分析实验"""
    experiment_name = config['experiment_name']
    algorithms = config['algorithms']
    seeds = config['seeds']
    max_updates = config['max_updates']
    fixed_params = config.get('fixed_params', {})
    experiments = config['experiments']
    
    # 收集所有实验任务
    all_tasks = []
    task_info = []

    for exp_config in experiments:
        exp_name = exp_config['name']
        param_name = exp_config['parameter']
        param_values = exp_config['values']

        for param_value in param_values:
            for algorithm in algorithms:
                for seed in seeds:
                    # 合并扫描参数和算法特定超参数（扫描参数优先）
                    algo_hps = ALGORITHM_HYPERPARAMETERS.get(algorithm, {})
                    param_overrides = {**algo_hps, param_name: param_value}
                    config_overrides = build_config_overrides(fixed_params, param_overrides)

                    # 添加参数值到实验名称
                    full_exp_name = f"{experiment_name}_{exp_name}_{param_value}"
                    task = (algorithm, seed, full_exp_name, config_overrides, max_updates, threads_per_proc, results_base_dir_override)
                    all_tasks.append(task)
                    task_info.append({'experiment': exp_name, 'param_name': param_name, 'param_value': param_value})

    total_tasks = len(all_tasks)
    print(f"📊 开始敏感性实验: {total_tasks} 个任务")
    print(f"   包含实验类型: {[exp['name'] for exp in experiments]}")
    print(f"   并行度: {max_workers}")
    print(f"   每进程线程数: {threads_per_proc}")

    all_results = []

    # 统一并行执行所有任务
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 使用字典将 future 映射到其任务索引，以便高效查找
        futures = {executor.submit(run_single_experiment, task): i for i, task in enumerate(all_tasks)}

        # 使用进度条显示总体进度
        for future in tqdm(as_completed(futures), total=total_tasks, desc="Sensitivity progress", ncols=100):
            result = future.result()
            # 找到对应的实验信息
            task_index = futures[future]
            info = task_info[task_index]
            result['experiment'] = info['experiment']
            result['param_name'] = info['param_name']
            result['param_value'] = info['param_value']
            all_results.append(result)
    
    # 统计结果
    success_count = sum(1 for r in all_results if r['status'] == 'success')
    failed_count = len(all_results) - success_count
    
    print(f"\n📈 敏感性实验完成:")
    print(f"   成功: {success_count}/{len(all_results)}")
    print(f"   失败: {failed_count}/{len(all_results)}")
    
    # 如果有失败的实验，打印详细信息
    if failed_count > 0:
        print("\n--- 失败的实验详情 ---")
        for r in all_results:
            if r['status'] != 'success':
                print(f"  - 实验: {r['experiment']}, 算法: {r['algorithm']}, 种子: {r['seed']}, 参数: {r['param_name']} = {r['param_value']}")
                if r['status'] == 'failed':
                    print(f"    状态: 失败\n    错误输出:\n{r['stderr']}")
                elif r['status'] == 'error':
                    print(f"    状态: 异常\n    错误信息: {r['error']}")
        print("------------------------")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="批量实验运行脚本", epilog="""
使用示例:
  # 运行所有配置的算法和种子
  python scripts/run_batch.py configs/experiments/convergence.yaml

  # 只运行 lru_avg 算法，使用种子 9-13
  python scripts/run_batch.py configs/experiments/convergence.yaml --algorithms lru_avg --seeds 9 10 11 12 13

  # 只运行特定算法
  python scripts/run_batch.py configs/experiments/convergence.yaml --algorithms h_mappo_l ippo

  # 只使用特定种子
  python scripts/run_batch.py configs/experiments/convergence.yaml --seeds 7 8 9

  # 覆盖 h_mappo_l 算法的 ent_coef 参数
  python scripts/run_batch.py configs/experiments/convergence.yaml --algorithms h_mappo_l --override h_mappo_l.agent.ent_coef=0.01

  # 覆盖多个参数
  python scripts/run_batch.py configs/experiments/convergence.yaml --override h_mappo_l.agent.ent_coef=0.01 ippo.agent.num_steps=300
""", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'experiment_config',
        help='实验配置文件路径 (例如: configs/experiments/convergence.yaml)'
    )

    # 添加算法和种子覆盖参数
    parser.add_argument(
        '--algorithms',
        nargs='+',
        help='覆盖配置文件中的算法列表 (例如: --algorithms lru_avg ippo)'
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        help='覆盖配置文件中的种子列表 (例如: --seeds 9 10 11 12 13)'
    )
    
    # 确定一个合理的并行工作进程数默认值
    # 设置为CPU核心数的一半，至少为1，最多16个防止开太多消耗过多内存
    try:
        cpu_count = os.cpu_count() or 1
        # 默认使用一半的CPU核心，但至少为1，至多不超过16个
        default_workers = max(1, min(16, cpu_count // 2))
    except NotImplementedError:
        default_workers = 4  # 在某些特殊环境无法获取CPU数时的后备值

    parser.add_argument(
        '--max-workers',
        type=int,
        default=default_workers,
        help=f'并行工作进程数 (自动推荐: {default_workers})'
    )
    parser.add_argument(
        '--threads-per-proc',
        type=int,
        default=1,
        help='每个子进程的BLAS/Torch线程数 (默认: 1)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='只显示将要执行的命令，不实际运行'
    )
    parser.add_argument(
        '--results-base-dir',
        '--results-dir',
        dest='results_base_dir',
        type=str,
        default=None,
        help='覆盖默认结果根目录（别名: --results-dir），例如 E:/exp_outputs/convergence_2025_09_09'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default=None,
        help='为结果目录添加自定义后缀，例如 my_feature_test'
    )
    parser.add_argument(
        '--latency-constraint',
        type=float,
        default=None,
        help='覆盖配置文件中的时延约束 (latency_constraint_s)'
    )
    parser.add_argument(
        '--override',
        nargs='+',
        help='覆盖特定算法的超参数。格式: ALGORITHM.PARAM_KEY=VALUE (例如: --override h_mappo_l.agent.ent_coef=0.01)'
    )
    
    args = parser.parse_args()

    # 应用自定义超参数覆盖
    if args.override:
        print("📝 应用自定义超参数覆盖:")
        for ov in args.override:
            try:
                key_part, value_str = ov.split('=', 1)
                algo, param_key = key_part.split('.', 1)
                
                # 尝试将值转换为 float/int，否则保留为字符串
                try:
                    # 支持科学记数法
                    value = float(value_str)
                    # 检查是否为整数值, e.g., 200.0
                    if value == int(value):
                        value = int(value)
                except ValueError:
                    value = value_str

                # 如果算法不在预定义中，则添加它
                if algo not in ALGORITHM_HYPERPARAMETERS:
                    ALGORITHM_HYPERPARAMETERS[algo] = {}
                
                ALGORITHM_HYPERPARAMETERS[algo][param_key] = value
                print(f"   - 对于 '{algo}', 设置 '{param_key}' 为 '{value}'")

            except ValueError:
                print(f"❌ 无效的覆盖格式: '{ov}'。请使用 ALGORITHM.PARAM_KEY=VALUE")
                sys.exit(1)

    # 加载实验配置
    config = load_experiment_config(args.experiment_config)
    experiment_name = config['experiment_name']

    # 使用命令行参数覆盖配置文件设置
    if args.algorithms:
        config['algorithms'] = args.algorithms
        print(f"📝 使用命令行指定的算法: {args.algorithms}")

    if args.seeds:
        config['seeds'] = args.seeds
        print(f"📝 使用命令行指定的种子: {args.seeds}")

    # 使用命令行参数覆盖时延约束
    if args.latency_constraint is not None:
        # 确保 fixed_params 和 framework 字典存在
        if 'fixed_params' not in config:
            config['fixed_params'] = {}
        if 'framework' not in config['fixed_params']:
            config['fixed_params']['framework'] = {}
        
        config['fixed_params']['framework']['latency_constraint_s'] = args.latency_constraint
        print(f"📝 使用命令行指定的时延约束: {args.latency_constraint}")
    
    # 根据命令行参数自动构建结果目录
    results_dir = args.results_base_dir
    if results_dir is None:
        # 确定基础目录
        base_dir = 'experiment_result'
        if experiment_name.startswith('convergence'):
            base_dir = 'experiment_result_convergence'
        elif experiment_name.startswith('sensitivity'):
            base_dir = 'experiment_result_sensitivity'
        elif experiment_name.startswith('scalability'):
            base_dir = 'experiment_result_scalability'
        
        # 构建后缀
        suffix_parts = []
        if args.latency_constraint is not None:
            # 将浮点数转为有效字符串，例如 3.0 -> "3", 3.5 -> "3.5"
            lc_str = f"{args.latency_constraint:g}"
            suffix_parts.append(f"lc{lc_str}")
        
        if args.suffix:
            suffix_parts.append(args.suffix)
            
        if suffix_parts:
            final_suffix = "_".join(suffix_parts)
            results_dir = f"{base_dir}_{final_suffix}"
            print(f"📝 自动设置结果目录为: {results_dir}")

    print(f"🧪 加载实验配置: {experiment_name}")
    print(f"   描述: {config.get('description', 'N/A')}")
    
    if args.dry_run:
        print("🔍 干运行模式，只显示任务不执行")
        return
    
    start_time = time.time()
    
    # 根据实验类型选择运行函数
    if experiment_name == "convergence":
        results = run_convergence_experiment(config, args.max_workers, args.threads_per_proc, results_dir)
    elif experiment_name == "scalability":
        results = run_scalability_experiment(config, args.max_workers, args.threads_per_proc, results_dir)
    elif experiment_name == "sensitivity":
        results = run_sensitivity_experiment(config, args.max_workers, args.threads_per_proc, results_dir)
    else:
        print(f"❌ 未知实验类型: {experiment_name}")
        return
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n🎉 所有实验完成! 总耗时: {duration:.2f}秒")
    if results_dir:
        print(f"💾 结果保存在 {results_dir}/ 目录下")
    else:
        if experiment_name == "convergence":
            print(f"💾 结果保存在 experiment_result_convergence/ 目录下")
        elif experiment_name == "sensitivity":
            print(f"💾 结果保存在 experiment_result_sensitivity/ 目录下")
        elif experiment_name == "scalability":
            print(f"💾 结果保存在 experiment_result_scalability/ 目录下")
        else:
            print(f"💾 结果保存在 experiment_result/ 目录下")

if __name__ == "__main__":
    main()
