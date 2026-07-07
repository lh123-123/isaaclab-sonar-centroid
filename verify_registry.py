#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))

import gymnasium as gym

import isaaclab_tasks  # 这会触发注册

def verify_registry():
    """验证注册表是否正常工作"""
    
    print("验证环境注册表...")
    
    try:
        all_envs = gym.registry
        print(f"已注册环境总数: {len(all_envs)}")
        
        target_env = "Isaac-Lh-2-Direct-v0"
        if target_env in all_envs:
            print(f"✅ 找到环境: {target_env}")
            spec = all_envs[target_env]
            print(f"   入口点: {spec.entry_point}")
            print(f"   参数: {spec.kwargs}")
            return True
        else:
            print(f"❌ 未找到环境: {target_env}")
            print("\n可用的环境:")
            for env_id in sorted(all_envs.keys()):
                if "Isaac" in env_id:
                    print(f"  - {env_id}")
            return False
            
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_registry()
    exit(0 if success else 1)
