import argparse
import re
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect the current P1 default pose in simulation.")
parser.add_argument("--task", type=str, default="Unitree-P1-WalkX", help="Gym task to load.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--settle_steps", type=int, default=240, help="Zero-action simulation steps before reporting.")
parser.add_argument("--print_interval", type=int, default=60, help="Print interval during settle/hold.")
parser.add_argument("--joint_regex", type=str, default=".*", help="Only print joints matching this regex.")
parser.add_argument("--body_regex", type=str, default=".*ankle_roll.*", help="Body regex for foot/body position prints.")
parser.add_argument("--real_time", action="store_true", default=False, help="Sleep to approximate real time.")
parser.add_argument("--no_hold", action="store_true", default=False, help="Exit after printing instead of holding the GUI.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _as_float(value) -> float:
    return float(value.detach().cpu().item() if isinstance(value, torch.Tensor) else value)


def _print_pose(base_env, joint_pattern: re.Pattern, body_pattern: re.Pattern, label: str):
    robot = base_env.scene["robot"]
    root_z = _as_float(robot.data.root_pos_w[0, 2])
    projected_gravity = robot.data.projected_gravity_b[0].detach().cpu().numpy()

    target_height = getattr(getattr(base_env.cfg.rewards, "base_height", None), "params", {}).get(
        "target_height", None
    )
    target_text = "N/A" if target_height is None else f"{target_height:.4f}"

    print(f"\n[{label}]")
    print(f"root_z={root_z:.6f}  target_height={target_text}  projected_gravity_b={projected_gravity}")

    print("joints: name  default_pos  current_pos  delta")
    for joint_id, joint_name in enumerate(robot.data.joint_names):
        if not joint_pattern.search(joint_name):
            continue
        default_pos = _as_float(robot.data.default_joint_pos[0, joint_id])
        current_pos = _as_float(robot.data.joint_pos[0, joint_id])
        print(f"  {joint_name:24s} {default_pos: .6f} {current_pos: .6f} {current_pos - default_pos: .6f}")

    body_ids = [i for i, name in enumerate(robot.data.body_names) if body_pattern.search(name)]
    if body_ids:
        print("bodies: name  world_xyz")
        for body_id in body_ids:
            body_name = robot.data.body_names[body_id]
            pos = robot.data.body_pos_w[0, body_id].detach().cpu().numpy()
            print(f"  {body_name:24s} ({pos[0]: .6f}, {pos[1]: .6f}, {pos[2]: .6f})")
    else:
        print(f"bodies: no body matched regex {body_pattern.pattern!r}")


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    base_env = env.unwrapped

    joint_pattern = re.compile(args_cli.joint_regex)
    body_pattern = re.compile(args_cli.body_regex)
    dt = base_env.step_dt
    action_dim = base_env.action_manager.total_action_dim
    actions = torch.zeros((base_env.num_envs, action_dim), device=base_env.device)

    _print_pose(base_env, joint_pattern, body_pattern, "after reset")

    for step in range(args_cli.settle_steps):
        env.step(actions)
        if args_cli.real_time:
            time.sleep(dt)
        if args_cli.print_interval > 0 and (step + 1) % args_cli.print_interval == 0:
            _print_pose(base_env, joint_pattern, body_pattern, f"settle step {step + 1}")

    _print_pose(base_env, joint_pattern, body_pattern, "after settle")

    should_hold = not args_cli.no_hold and not getattr(args_cli, "headless", False)
    if should_hold:
        print("\n[INFO] Holding GUI with zero actions. Close the Isaac Sim window or press Ctrl+C to exit.")
        while simulation_app.is_running():
            env.step(actions)
            if args_cli.real_time:
                time.sleep(dt)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
