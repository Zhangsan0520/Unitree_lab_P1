import argparse
import math
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Sweep P1 joints one by one in simulation.")
parser.add_argument("--task", type=str, default="Unitree-P1-WalkX", help="Gym task to load.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--joint", type=str, default=None, help="Only test joints matching this regex.")
parser.add_argument("--steps_per_joint", type=int, default=240, help="Simulation steps for each joint.")
parser.add_argument("--settle_steps", type=int, default=120, help="Zero-action settle steps before each sweep.")
parser.add_argument("--cycles", type=float, default=1.0, help="Sine-wave cycles per joint sweep.")
parser.add_argument(
    "--air_height",
    type=float,
    default=0.0,
    help="If > 0, keep the robot root suspended this many meters above the env origin.",
)
parser.add_argument(
    "--amplitude",
    type=float,
    default=0.35,
    help="Raw action amplitude in [-1, 1]. The final target is offset + scale * amplitude.",
)
parser.add_argument("--real_time", action="store_true", default=False, help="Sleep to approximate real time.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


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
    robot = base_env.scene["robot"]
    action_term = base_env.action_manager.get_term("JointPositionAction")

    joint_ids = action_term._joint_ids
    if isinstance(joint_ids, slice):
        joint_names = list(action_term._joint_names)
    else:
        joint_names = [robot.data.joint_names[i] for i in joint_ids]

    if args_cli.joint is not None:
        matched_ids, matched_names = robot.find_joints(args_cli.joint, preserve_order=True)
        wanted = set(matched_names)
        test_indices = [i for i, name in enumerate(joint_names) if name in wanted]
    else:
        test_indices = list(range(len(joint_names)))

    if not test_indices:
        raise RuntimeError(f"No joints matched pattern: {args_cli.joint}")

    print("[INFO] Joint order used by JointPositionAction:")
    for i, name in enumerate(joint_names):
        scale = float(action_term._scale[0, i]) if isinstance(action_term._scale, torch.Tensor) else float(action_term._scale)
        offset = float(action_term._offset[0, i]) if isinstance(action_term._offset, torch.Tensor) else float(action_term._offset)
        if action_term.cfg.clip is not None:
            clip_lo = float(action_term._clip[0, i, 0])
            clip_hi = float(action_term._clip[0, i, 1])
            clip_str = f"[{clip_lo:.3f}, {clip_hi:.3f}]"
        else:
            clip_str = "None"
        print(f"  {i:02d}  {name:20s}  scale={scale:.3f}  offset={offset:.3f}  clip={clip_str}")

    dt = base_env.step_dt
    action_dim = base_env.action_manager.total_action_dim
    actions = torch.zeros((base_env.num_envs, action_dim), device=base_env.device)
    root_pose = robot.data.default_root_state[:, :7].clone()
    root_velocity = torch.zeros((base_env.num_envs, 6), device=base_env.device)

    if args_cli.air_height > 0.0:
        root_pose[:, :3] = base_env.scene.env_origins.clone()
        root_pose[:, 2] += args_cli.air_height
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(root_velocity)
        base_env.sim.step()
        base_env.scene.update(dt)

    def step_env():
        nonlocal actions
        env.step(actions)
        if args_cli.air_height > 0.0:
            robot.write_root_pose_to_sim(root_pose)
            robot.write_root_velocity_to_sim(root_velocity)
        if args_cli.real_time:
            time.sleep(dt)

    for action_index in test_indices:
        name = joint_names[action_index]
        print(f"[TEST] Sweeping joint {action_index:02d}: {name}")

        actions.zero_()
        for _ in range(args_cli.settle_steps):
            step_env()

        pos_min = float("inf")
        pos_max = float("-inf")

        for step in range(args_cli.steps_per_joint):
            phase = 2.0 * math.pi * args_cli.cycles * step / max(args_cli.steps_per_joint - 1, 1)
            actions.zero_()
            actions[:, action_index] = args_cli.amplitude * math.sin(phase)
            step_env()
            current_pos = float(robot.data.joint_pos[0, action_index].item())
            pos_min = min(pos_min, current_pos)
            pos_max = max(pos_max, current_pos)

        offset = float(action_term._offset[0, action_index]) if isinstance(action_term._offset, torch.Tensor) else float(action_term._offset)
        scale = float(action_term._scale[0, action_index]) if isinstance(action_term._scale, torch.Tensor) else float(action_term._scale)
        expected_amp = abs(scale * args_cli.amplitude)
        print(
            f"[RESULT] {name}: target offset={offset:.3f}, expected +/- {expected_amp:.3f}, "
            f"observed range=[{pos_min:.3f}, {pos_max:.3f}]"
        )

    print("[INFO] Joint sweep finished.")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
