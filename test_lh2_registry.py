#!/usr/bin/env python3

import gymnasium as gym
import isaaclab_tasks

def test_environment():
    """测试注册表环境是否正常工作"""
    
    print("测试 Isaac-Lh-2-Direct-v0 环境...")
    
    try:
        env = gym.make("Isaac-Lh-2-Direct-v0", num_envs=4)
        print("✅ 环境创建成功")
        
        obs, info = env.reset()
        print(f"✅ 观测空间: {env.observation_space}")
        print(f"✅ 动作空间: {env.action_space}")
        print(f"✅ 初始观测形状: {obs.shape}")
        
        print("\n运行 10 步测试...")
        for i in range(10):
            actions = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(actions)
            print(f"  步骤 {i+1}: 奖励={reward.mean():.3f}, 完成={done.any()}")
            
            if done.any():
                print("  环境重置")
                obs, _ = env.reset()
        
        env.close()
        print("\n✅ 测试完成！环境可以正常运行。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_environment()
    exit(0 if success else 1)
