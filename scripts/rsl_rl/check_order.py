import gym
import isaaclab.envs  # 确保导入了 env 注册
from isaaclab.envs import ManagerBasedRslRlCfg
from isaaclab.utils import config_utils

def check_policy_order():
    # 1. 设置你的任务名称 (根据你的实际任务修改)
    task_name = "Unitree-P1-Velocity" # 请确认这与你的任务名称一致
    
    # 2. 尝试解析该任务的配置
    try:
        env_cfg = gym.spec(task_name).entry_point.env_cfg_cls()
    except Exception as e:
        print(f"无法加载任务配置: {e}")
        return

    # 3. 提取 ActionManager 中的关节顺序
    print("\n" + "="*40)
    print("=== 正在解析 Policy 关节顺序 ===")
    
    # 寻找 JointPositionAction 定义
    found = False
    for term_name, term_cfg in env_cfg.actions.items():
        if "JointPositionAction" in str(type(term_cfg)):
            print(f"找到动作项: {term_name}")
            print(f"关节匹配规则: {term_cfg.joint_names}")
            
            # 如果是 .*，则需要遍历 robot 资产的 joint_names
            # 这里我们直接打印 asset 定义中的 joint_names
            robot_cfg = env_cfg.scene.robot
            print("\n机器人资产定义的物理关节顺序:")
            # 这是一个通用逻辑，遍历机器人定义的关节列表
            print(robot_cfg.joint_names) 
            found = True
            break
    
    if not found:
        print("未找到明确的 JointPositionAction 定义，请检查 env_cfg 结构。")
    
    print("="*40 + "\n")

if __name__ == "__main__":
    check_policy_order()