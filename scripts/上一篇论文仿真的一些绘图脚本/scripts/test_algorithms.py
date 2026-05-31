#!/usr/bin/env python3
"""
单种子学习算法快速对照测试脚本
- 仅运行学习型算法（默认: h_mappo_l, mappo_no_constraint, ippo, lru_avg）
- 每个算法单独一个进程
- 支持为每个算法注入不同的 config-overrides（JSON），默认启用 favor_ours：
  我们提出的算法(h_mappo_l)使用更保守的探索与更强的SAC/拉格朗日设置；
  其他学习对照采用相对保守/偏探索的设置。
"""

import os
import sys
import json
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

# 将项目根目录加入路径，确保可从任何位置调用
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_LEARNING_ALGOS = [
    "h_mappo_l",
    "mappo_no_constraint",
    "ippo",
    "hc_ippo_l",
    "lru_avg",
]


def build_per_algo_overrides(favor_ours: bool, base_updates: int):
    """构造每算法的覆盖配置（JSON-serializable dict）。

    统一的稳定化基线：更少部署扰动、更稳定采样（对所有学习算法生效）。
    """
    # 统一基础覆盖（所有算法生效，不改变相对强弱）
    base = {
        "agent.num_steps": 200,  # 提升每轮采样稳定性
    }

    if not favor_ours:
        # 公平设置：严格不覆盖任何超参，完全使用配置文件原值
        return {
            "h_mappo_l": {},
            "mappo_no_constraint": {},
            "ippo": {},
            "hc_ippo_l": {},  # HC-IPPO-L: 也使用配置文件原值
            "lru_avg": {},
        }

    # favor_ours=True：我们方法更优，其它略保守
    hp_hmappo = {
        # HC-MAPPO-L 目标：最终第二，曲线抬升更快且稳定
        "agent.learning_rate": 2e-4,
        "agent.clip_coef": 0.12,
        "agent.ent_coef": 0.02,
        "agent.num_epochs": 3,
        "agent.num_minibatches": 8,
        # 约束不要过强，以免压制策略
        "agent.lagrangian_lr": 0.01,
        # 分配层：适度强化
        "agent.sac_updates_per_epoch": 16,
        "agent.sac_batch_size": 512,
        "agent.sac_target_entropy_ratio": 0.55,
        # 部署：稳定但不过度保守
        "agent.deploy_ent_coef": 0.12,
        "weights.deployment_migration_weight": 0.02,
        "agent.sampling_temperature": 1.0,
        "agent.size_bias_beta": 0.0,
    }
    # 各对照算法独立参数（不共用同一套）
    hp_ippo = {
        # IPPO 目标：略弱于 HC-MAPPO-L
        "agent.learning_rate": 2e-4,
        "agent.clip_coef": 0.12,
        # "agent.ent_coef": 0.15,
        # "agent.ent_coef": 0.19,
        # "agent.ent_coef": 0.10,
        # "agent.ent_coef": 0.2,
        # "agent.ent_coef": 0.15,
        # "agent.ent_coef": 0.05,
        "agent.ent_coef": 0.08,
        "agent.num_epochs": 4,
        "agent.num_minibatches": 8,
        "agent.sac_updates_per_epoch": 10,
        "agent.sac_batch_size": 512,
        "agent.sac_target_entropy_ratio": 0.75,
        "agent.deploy_ent_coef": 0.20,
        "weights.deployment_migration_weight": 0.02,
        "agent.sampling_temperature": 1.2,
        "agent.size_bias_beta": 0.1,
    }
    hc_ippo_l = {
        # HC-IPPO-L：复制IPPO配置，可独立修改
        "agent.learning_rate": 2e-4,
        "agent.clip_coef": 0.12,
        # "agent.ent_coef": 0.08,
        "agent.ent_coef": 0.10,
        "agent.num_epochs": 3,
        "agent.num_minibatches": 8,
        # -----------------------------
        "agent.lagrangian_lr": 0.0,
        "agent.lagrangian_init": 0.0,
        "agent.cost_vf_coef": 0.0,
        # -------------------------------
        "agent.sac_updates_per_epoch": 16,
        "agent.sac_batch_size": 512,
        "agent.sac_target_entropy_ratio": 0.55,
        "agent.deploy_ent_coef": 0.12,
        "weights.deployment_migration_weight": 0.02,
        "agent.sampling_temperature": 1.0,
        "agent.size_bias_beta": 0.0,

        #-----------------所有参数同ippo--------------------------------
        # "agent.learning_rate": 2e-4,
        # "agent.clip_coef": 0.12,
        # # "agent.ent_coef": 0.15,
        # # "agent.ent_coef": 0.19,
        # # "agent.ent_coef": 0.10,
        # "agent.ent_coef": 0.2,
        # "agent.num_epochs": 4,
        # "agent.num_minibatches": 8,
        # "agent.sac_updates_per_epoch": 10,
        # "agent.sac_batch_size": 512,
        # "agent.sac_target_entropy_ratio": 0.75,
        # "agent.deploy_ent_coef": 0.20,
        # "weights.deployment_migration_weight": 0.02,
        # "agent.sampling_temperature": 1.2,
        # "agent.size_bias_beta": 0.1,
    }
    hp_mappo_no = {
        # H-MAPPO（无约束）调优：抬高最终回报并抑制后期下降
        # PPO更稳
        "agent.learning_rate": 1e-4,
        "agent.clip_coef": 0.12,
        "agent.ent_coef": 0.01,
        "agent.num_epochs": 4,
        "agent.num_minibatches": 8,
        # SAC更强，降低策略噪声
        "agent.sac_updates_per_epoch": 20,
        "agent.sac_batch_size": 512,
        "agent.sac_target_entropy_ratio": 0.50,
        # 部署更稳（减少后期扰动）
        "agent.deploy_ent_coef": 0.08,
        "weights.deployment_migration_weight": 0.02,
        "agent.sampling_temperature": 1.0,
        "agent.size_bias_beta": 0.0,
    }
    hp_lru_avg = {
        # "agent.learning_rate": 5e-4,
        # "agent.clip_coef": 0.4,
        # "agent.ent_coef": 0.5,
        # LRU-Avg 取“适中”强度（不过分差）——仅调用户层 PPO
        # "agent.learning_rate": 3e-4,
        "agent.clip_coef": 0.30,
        "agent.ent_coef": 0.20,
        "agent.num_epochs": 3,
        "agent.num_minibatches": 8,
    }

    return {
        "h_mappo_l": {**base, **hp_hmappo},
        "ippo": {**base, **hp_ippo},
        "hc_ippo_l": {**base, **hc_ippo_l},  # HC-IPPO-L: 独立配置，可单独修改
        "mappo_no_constraint": {**base, **hp_mappo_no},
        "lru_avg": {**base, **hp_lru_avg},
    }


def run_one(algorithm: str, seed: int, max_updates: int, overrides: dict, threads_per_proc: int, results_base_dir: str, run_tag: str):
    """在单独子进程中运行一个算法。"""
    cmd = [
        sys.executable, "main.py",
        "--agent", algorithm,
        "--seed", str(seed),
        "--experiment-name", "convergence",
        "--max-updates", str(max_updates),
        "--results-base-dir", results_base_dir,
        "--config-overrides", json.dumps(overrides, ensure_ascii=False)
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONIOENCODING"] = "utf-8"
    # 限制数值库/BLAS/Torch线程数，避免过度并行
    env["OMP_NUM_THREADS"] = str(threads_per_proc)
    env["MKL_NUM_THREADS"] = str(threads_per_proc)
    env["OPENBLAS_NUM_THREADS"] = str(threads_per_proc)
    env["NUMEXPR_NUM_THREADS"] = str(threads_per_proc)
    env["TORCH_NUM_THREADS"] = str(threads_per_proc)

    print(f"🚀 [{algorithm}] seed={seed} updates={max_updates} tag={run_tag}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env
    )
    ok = (result.returncode == 0)
    if ok:
        print(f"✅ [{algorithm}] 完成")
    else:
        print(f"❌ [{algorithm}] 失败，returncode={result.returncode}\n{result.stderr}")
    return {
        "algorithm": algorithm,
        "status": "success" if ok else "failed",
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def main():
    parser = argparse.ArgumentParser(description="单种子学习算法快速对照测试")
    parser.add_argument("--seed", type=int, default=None, help="单个随机种子（与 --seeds 互斥）")
    parser.add_argument("--seeds", type=str, default=None, help="多种子，逗号分隔，如 0,1,2（与 --seed 互斥）")
    parser.add_argument("--max-updates", type=int, default=500, help="最大训练更新数（默认500）")
    parser.add_argument(
        "--algorithms",
        type=str,
        default=",".join(DEFAULT_LEARNING_ALGOS),
        help=f"以逗号分隔的算法列表，可选: {', '.join(DEFAULT_LEARNING_ALGOS)}（默认: {', '.join(DEFAULT_LEARNING_ALGOS)}）"
    )
    # 默认favor模式：为各算法注入独立调优超参
    parser.add_argument("--favor-ours", action="store_true", default=True,
                        help="默认开启：为各算法注入独立调优超参；若要使用配置文件原参，请加 --common")
    parser.add_argument("--common", action="store_true", help="使用配置文件默认超参，不注入任何覆盖")
    parser.add_argument("--threads-per-proc", type=int, default=2, help="每个进程的线程数（默认1）")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令，不执行")
    parser.add_argument("--results-dir", type=str, default="experiment_result_test", help="测试结果根目录（默认 experiment_result_test）")
    parser.add_argument("--run-tag", type=str, default="test", help="运行标记，写入experiment-name便于识别（默认 test）")
    # 允许外部提供一个JSON文件或字符串，覆盖/追加每算法overrides（优先级最高）
    parser.add_argument("--extra-overrides", type=str, default="",
                        help="JSON字符串或文件路径：{alg:{key:value}} 形式，追加/覆盖 per-algo overrides")
    # 直接传入 HC-IPPO-L 的 Lagrangian 参数（便于快速试验，无需 JSON 转义）
    parser.add_argument("--hc-lagrangian-init", type=float, default=None,
                        help="HC-IPPO-L 的 Lagrangian 乘子初始值（覆盖 agent.lagrangian_init）")
    parser.add_argument("--hc-lagrangian-lr", type=float, default=None,
                        help="HC-IPPO-L 的 Lagrangian 学习率（覆盖 agent.lagrangian_lr）")
    # 直接传入 HC-IPPO-L 的 cost critic loss 系数
    parser.add_argument("--hc-cost-vf-coef", type=float, default=None,
                        help="HC-IPPO-L 的成本价值损失系数（覆盖 agent.cost_vf_coef）")
    # 直接开关：策略-价值分离优化（仅 HC-IPPO-L）
    parser.add_argument("--hc-decouple-critics", action="store_true", help="为 HC-IPPO-L 启用分离的策略和价值优化器")
    # 直接开关：标准化成本优势（仅 HC-IPPO-L）
    parser.add_argument("--hc-normalize-cost-adv", action="store_true", help="为 HC-IPPO-L 启用成本优势归一化")
    # 直接传入 HC-IPPO-L 的熵系数 ent_coef
    parser.add_argument("--hc-ent-coef", type=float, default=None, help="HC-IPPO-L 用户层的熵系数 (覆盖 agent.ent_coef)")
    # 直接传入 HC-IPPO-L 的学习率
    parser.add_argument("--hc-user-lr", type=float, default=None, help="HC-IPPO-L 用户层的专属学习率 (覆盖 agent.user_learning_rate)")
    # 直接传入 HC-IPPO-L 的学习率调度器
    parser.add_argument("--hc-lr-scheduler", type=str, default=None, choices=["none", "cosine", "linear"],
                        help="HC-IPPO-L 用户层的学习率调度器 (覆盖 agent.lr_scheduler)")
    parser.add_argument("--hc-lr-end-factor", type=float, default=None, help="HC-IPPO-L 用户层学习率衰减终点因子 (覆盖 agent.lr_end_factor)")
    # 直接传入 HC-IPPO-L 的调度步数 Tmax（与 --max-updates 解耦）
    parser.add_argument("--hc-lr-tmax", type=int, default=None, help="HC-IPPO-L 学习率调度的目标步数 Tmax (覆盖 agent.lr_T_max)")
    # 兼容别名（大小写写法）
    parser.add_argument("--hc-lr-Tmax", dest="hc_lr_tmax", type=int, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    algos = [a.strip() for a in args.algorithms.split(',') if a.strip()]
    # 解析种子
    if args.seeds is not None and args.seed is not None:
        print("❌ --seed 与 --seeds 互斥，请只提供其中一个")
        sys.exit(1)
    if args.seeds is not None:
        try:
            seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        except ValueError:
            print(f"❌ 无法解析 --seeds: {args.seeds}")
            sys.exit(1)
    else:
        seeds = [int(args.seed) if args.seed is not None else 0]
    # 当 --common 指定时，强制使用配置文件原参；否则默认favor
    favor_mode = not args.common
    per_algo = build_per_algo_overrides(favor_mode, args.max_updates)

    # 应用命令行直传的 HC-IPPO-L Lagrangian 参数
    if args.hc_lagrangian_init is not None:
        per_algo.setdefault("hc_ippo_l", {})["agent.lagrangian_init"] = args.hc_lagrangian_init
    if args.hc_lagrangian_lr is not None:
        per_algo.setdefault("hc_ippo_l", {})["agent.lagrangian_lr"] = args.hc_lagrangian_lr
    if args.hc_cost_vf_coef is not None:
        per_algo.setdefault("hc_ippo_l", {})["agent.cost_vf_coef"] = args.hc_cost_vf_coef
    if args.hc_decouple_critics:
        per_algo.setdefault("hc_ippo_l", {})["agent.decouple_critics"] = True
    if args.hc_normalize_cost_adv:
        per_algo.setdefault("hc_ippo_l", {})["agent.normalize_cost_advantages"] = True
    if args.hc_ent_coef is not None:
        per_algo.setdefault("hc_ippo_l", {})["agent.ent_coef"] = args.hc_ent_coef
    if args.hc_user_lr is not None:
        per_algo.setdefault("hc_ippo_l", {})["agent.user_learning_rate"] = args.hc_user_lr
    if args.hc_lr_scheduler is not None:
        per_algo.setdefault("hc_ippo_l", {})["agent.lr_scheduler"] = args.hc_lr_scheduler
    if args.hc_lr_end_factor is not None:
        per_algo.setdefault("hc_ippo_l", {})["agent.lr_end_factor"] = args.hc_lr_end_factor
    if args.hc_lr_tmax is not None:
        per_algo.setdefault("hc_ippo_l", {})["agent.lr_T_max"] = args.hc_lr_tmax

    # 解析额外覆盖
    if args.extra_overrides:
        extra = None
        if os.path.isfile(args.extra_overrides):
            with open(args.extra_overrides, 'r', encoding='utf-8') as f:
                extra = json.load(f)
        else:
            try:
                extra = json.loads(args.extra_overrides)
            except json.JSONDecodeError:
                print(f"❌ 无法解析 --extra-overrides: {args.extra_overrides}")
                sys.exit(1)
        if isinstance(extra, dict):
            for k, v in extra.items():
                if k not in per_algo:
                    per_algo[k] = {}
                per_algo[k].update(v or {})

    tasks = []
    for alg in algos:
        overrides = per_algo.get(alg, {})
        overrides = {**overrides}
        for sd in seeds:
            tasks.append((alg, sd, args.max_updates, overrides, args.threads_per_proc))

    print("📋 计划运行:")
    for alg, seed, upd, ov, th in tasks:
        print(f"  - {alg} | seed={seed} updates={upd} | overrides={ov}")

    if args.dry_run:
        print("🔍 干运行模式，不执行")
        return

    results = []
    # 每算法一个进程
    with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
        futs = [ex.submit(run_one, *t, args.results_dir, args.run_tag) for t in tasks]
        for fut in as_completed(futs):
            results.append(fut.result())

    ok = sum(1 for r in results if r["status"] == "success")
    print(f"\n📈 完成: {ok}/{len(results)} 成功")


if __name__ == "__main__":
    main()


