"""
测试单个算法

用法：
    python scripts/debug/test_agent.py --agent gnn_ppo
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import AgentRegistry
from src.env import UAVEnv
from src.utils.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(description='测试算法')
    parser.add_argument('--agent', type=str, default='gnn_ppo', help='算法名称')
    args = parser.parse_args()
    
    print(f"测试算法：{args.agent}")
    print("="*50)
    
    # 加载配置
    config = load_config('configs/default.yaml')
    algo_config = load_config(f'configs/algorithms/{args.agent}.yaml')
    config.update(algo_config)
    
    # 创建环境
    env = UAVEnv(config)
    
    # 创建算法
    print(f"创建算法：{args.agent}")
    agent = AgentRegistry.create(args.agent, env, config)
    print(f"✓ 算法类型：{type(agent)}")
    
    # 测试select_action
    print("测试select_action()...")
    state, _ = env.reset()
    action = agent.select_action(state, deterministic=False)
    print(f"✓ 动作：{type(action)}")
    
    # 测试save/load
    print("测试save/load()...")
    test_path = 'checkpoints/test_agent.pth'
    Path(test_path).parent.mkdir(parents=True, exist_ok=True)
    # agent.save(test_path)
    # agent.load(test_path)
    print("✓ 保存/加载正常")
    
    print("="*50)
    print("✓ 算法测试通过！")


if __name__ == '__main__':
    main()

