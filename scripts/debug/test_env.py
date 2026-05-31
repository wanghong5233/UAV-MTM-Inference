"""
测试环境是否正常

用法：
    python scripts/debug/test_env.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.env import UAVEnv
from src.utils.config_loader import load_config


def main():
    print("测试环境...")
    print("="*50)
    
    # 加载默认配置
    config = load_config('configs/default.yaml')
    
    # 创建环境
    env = UAVEnv(config)
    
    # 测试reset
    print("测试reset()...")
    state, info = env.reset()
    print(f"✓ 状态空间：{type(state)}")
    
    # 测试step
    print("测试step()...")
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Reward: {reward}")
    print(f"✓ Terminated: {terminated}, Truncated: {truncated}")
    
    # 运行一个完整episode
    print("运行完整episode...")
    state, _ = env.reset()
    total_reward = 0.0
    steps = 0
    max_steps = 10
    
    while steps < max_steps:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    print(f"✓ Episode完成：{steps}步，总奖励={total_reward:.4f}")
    
    print("="*50)
    print("✓ 环境测试通过！")


if __name__ == '__main__':
    main()

