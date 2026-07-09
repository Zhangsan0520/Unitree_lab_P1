# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import csv

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--zero-action",
    action="store_true",
    default=False,
    help="Bypass policy inference and apply zero actions to validate the default standing behavior.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--debug-base",
    action="store_true",
    default=False,
    help="Print root height and projected gravity for env 0 during playback.",
)
parser.add_argument(
    "--debug-base-interval",
    type=int,
    default=50,
    help="Step interval for --debug-base prints.",
)
parser.add_argument(
    "--disable-keyboard-control",
    action="store_true",
    default=False,
    help="Disable keyboard control of the base_velocity command in the Isaac Lab GUI.",
)
parser.add_argument("--print-obs", action="store_true", help="Print policy observation tensor periodically.")
parser.add_argument("--print-obs-interval", type=int, default=50, help="Print observation every N policy steps.")
parser.add_argument("--keyboard-vx", type=float, default=None, help="Keyboard x velocity limit in m/s.")
parser.add_argument("--keyboard-vy", type=float, default=None, help="Keyboard y velocity limit in m/s.")
parser.add_argument("--keyboard-wz", type=float, default=None, help="Keyboard yaw velocity limit in rad/s.")
parser.add_argument(
    "--keyboard-accel",
    type=float,
    default=0.5,
    help="Maximum linear command change per second for keyboard control.",
)
parser.add_argument(
    "--keyboard-yaw-accel",
    type=float,
    default=0.75,
    help="Maximum yaw command change per second for keyboard control.",
)
parser.add_argument(
    "--keyboard-push-velocity",
    type=float,
    default=0.1,
    help="Root velocity impulse in m/s applied by J/K/L/I/U/O/N/M during GUI playback.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import carb
import omni

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except Exception:
    get_published_pretrained_checkpoint = None
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


class KeyboardVelocityController:
    """Small GUI keyboard controller for base velocity commands."""

    _KEY_BINDINGS = {
        "W": (0, 1),
        "UP": (0, 1),
        "NUMPAD_8": (0, 1),
        "S": (0, -1),
        "DOWN": (0, -1),
        "NUMPAD_2": (0, -1),
        "A": (1, 1),
        "LEFT": (1, 1),
        "NUMPAD_4": (1, 1),
        "D": (1, -1),
        "RIGHT": (1, -1),
        "NUMPAD_6": (1, -1),
        "Q": (2, 1),
        "NUMPAD_7": (2, 1),
        "E": (2, -1),
        "NUMPAD_9": (2, -1),
    }
    _PUSH_BINDINGS = {
        "I": (1.0, 0.0),
        "K": (-1.0, 0.0),
        "J": (0.0, 1.0),
        "L": (0.0, -1.0),
        "U": (1.0, 1.0),
        "O": (1.0, -1.0),
        "N": (-1.0, 1.0),
        "M": (-1.0, -1.0),
    }
    _RESET_KEYS = {"X"}

    def __init__(
        self,
        command_ranges: tuple[tuple[float, float], ...],
        command_rates: tuple[float, float, float],
        push_velocity: float,
        device,
    ):
        self._device = device
        self._command_ranges = torch.tensor(command_ranges, dtype=torch.float32, device=device)
        self._command_rates = torch.tensor(command_rates, dtype=torch.float32, device=device)
        self._command = torch.zeros(3, dtype=torch.float32, device=device)
        self._target_command = torch.zeros_like(self._command)
        self._push_velocity = float(abs(push_velocity))
        self._pending_push = torch.zeros(2, dtype=torch.float32, device=device)
        self._pressed_keys: set[str] = set()

        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            raise RuntimeError("No Isaac Sim app window is available for keyboard control.")
        self._input = carb.input.acquire_input_interface()
        self._keyboard = app_window.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

    def close(self):
        if getattr(self, "_keyboard_sub", None) is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def reset(self):
        self._pressed_keys.clear()
        self._target_command.zero_()
        self._command.zero_()
        self._pending_push.zero_()

    def advance(self, dt: float) -> torch.Tensor:
        max_delta = self._command_rates * dt
        delta = torch.clamp(self._target_command - self._command, min=-max_delta, max=max_delta)
        self._command += delta
        return self._command

    def pop_push(self) -> torch.Tensor:
        push = self._pending_push.clone()
        self._pending_push.zero_()
        return push

    def _on_keyboard_event(self, event, *args, **kwargs):
        key_input = getattr(event, "input", None)

        if hasattr(key_input, "name"):
            key = key_input.name
        else:
            key = str(key_input)

        key = key.upper()
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key in self._RESET_KEYS:
                self.reset()
            elif key in self._PUSH_BINDINGS:
                direction = torch.tensor(self._PUSH_BINDINGS[key], dtype=torch.float32, device=self._device)
                direction = direction / torch.linalg.norm(direction)
                self._pending_push += direction * self._push_velocity
            elif key in self._KEY_BINDINGS:
                self._pressed_keys.add(key)
                self._update_target_command()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if key in self._KEY_BINDINGS:
                self._pressed_keys.discard(key)
                self._update_target_command()
        return True

    def _update_target_command(self):
        self._target_command.zero_()
        for axis in range(3):
            has_positive = any(self._KEY_BINDINGS[key] == (axis, 1) for key in self._pressed_keys)
            has_negative = any(self._KEY_BINDINGS[key] == (axis, -1) for key in self._pressed_keys)
            if has_positive and not has_negative:
                self._target_command[axis] = self._command_ranges[axis, 1]
            elif has_negative and not has_positive:
                self._target_command[axis] = self._command_ranges[axis, 0]


def _get_observations(env):
    observations = env.get_observations()
    if isinstance(observations, tuple):
        observations = observations[0]
    return observations

def _get_policy_obs_tensor(obs):
    """Return policy observation tensor from env observation or TensorDict."""
    if isinstance(obs, tuple):
        obs = obs[0]

    if torch.is_tensor(obs):
        return obs

    # TensorDict or dict with "policy" key
    if hasattr(obs, "keys") and "policy" in list(obs.keys()):
        policy_obs = obs["policy"]
        if torch.is_tensor(policy_obs):
            return policy_obs

        # Some TensorDict nesting cases
        if hasattr(policy_obs, "keys"):
            for key in policy_obs.keys():
                value = policy_obs[key]
                if torch.is_tensor(value):
                    return value

    # Generic dict / TensorDict fallback
    if hasattr(obs, "keys"):
        for key in obs.keys():
            value = obs[key]
            if torch.is_tensor(value):
                return value
            if hasattr(value, "keys"):
                for sub_key in value.keys():
                    sub_value = value[sub_key]
                    if torch.is_tensor(sub_value):
                        return sub_value

    raise RuntimeError(
        f"Could not extract policy observation tensor from obs. "
        f"type(obs)={type(obs)}, keys={list(obs.keys()) if hasattr(obs, 'keys') else None}"
    )

def _print_action_clip_debug(env, step: int, interval: int = 20):
    if step % interval != 0:
        return

    import torch

    action_manager = env.unwrapped.action_manager

    try:
        term = action_manager.get_term("JointPositionAction")
    except Exception:
        term = action_manager._terms["JointPositionAction"]

    raw = getattr(term, "raw_actions", None)
    if raw is None:
        raw = getattr(term, "_raw_actions", None)

    processed = getattr(term, "processed_actions", None)
    if processed is None:
        processed = getattr(term, "_processed_actions", None)

    if raw is None or processed is None:
        print("\n[ACTION DEBUG] Cannot find raw_actions or processed_actions.")
        print("term attrs:", [k for k in dir(term) if "action" in k.lower()])
        return

    raw = raw.detach()
    processed = processed.detach()

    print("\n================ ACTION DEBUG ================")
    print(f"step = {step}")
    print("raw shape      :", tuple(raw.shape))
    print("processed shape:", tuple(processed.shape))

    print("\n[raw action]")
    print("min/max/mean/std:",
          raw.min().item(),
          raw.max().item(),
          raw.mean().item(),
          raw.std().item())
    print("finite:", torch.isfinite(raw).all().item())

    print("\n[processed action / joint target]")
    print("min/max/mean/std:",
          processed.min().item(),
          processed.max().item(),
          processed.mean().item(),
          processed.std().item())
    print("finite:", torch.isfinite(processed).all().item())

    scale = getattr(term, "_scale", None)
    offset = getattr(term, "_offset", None)

    if scale is not None and offset is not None:
        before_clip = raw * scale + offset
        after_clip = processed
        diff = after_clip - before_clip
        hit_clip = diff.abs() > 1e-6

        print("\n[clip check]")
        print("clip ratio    :", hit_clip.float().mean().item())
        print("max clip diff :", diff.abs().max().item())
        print("before min/max:", before_clip.min().item(), before_clip.max().item())
        print("after  min/max:", after_clip.min().item(), after_clip.max().item())

        joint_names = getattr(term, "_joint_names", None)
        if joint_names is not None:
            print("\n[clipped joints]")
            any_clipped = False
            for j, name in enumerate(joint_names):
                ratio = hit_clip[:, j].float().mean().item()
                if ratio > 0.01:
                    any_clipped = True
                    print(
                        name,
                        "ratio =", round(ratio, 4),
                        "before =", round(before_clip[0, j].item(), 4),
                        "after =", round(after_clip[0, j].item(), 4),
                        "diff =", round(diff[0, j].item(), 4),
                    )
            if not any_clipped:
                print("No joint clip ratio > 1%.")

    robot = env.unwrapped.scene["robot"]
    joint_pos = robot.data.joint_pos.detach()
    joint_vel = robot.data.joint_vel.detach()

    print("\n[actual joint state]")
    print("joint_pos min/max/mean/std:",
          joint_pos.min().item(),
          joint_pos.max().item(),
          joint_pos.mean().item(),
          joint_pos.std().item())
    print("joint_vel min/max/mean/std:",
          joint_vel.min().item(),
          joint_vel.max().item(),
          joint_vel.mean().item(),
          joint_vel.std().item())

    print("==============================================\n")

def _abs_max_range(range_values) -> float:
    return max(abs(float(range_values[0])), abs(float(range_values[1])))


def _resolve_keyboard_axis_range(command_cfg, axis_name: str, cli_value: float | None, fallback: float):
    if cli_value is not None:
        limit = abs(cli_value)
        return -limit, limit

    ranges_cfg = getattr(command_cfg, "limit_ranges", None)
    if ranges_cfg is None:
        ranges_cfg = getattr(command_cfg, "ranges", None)
    range_values = getattr(ranges_cfg, axis_name, None)
    if range_values is not None and _abs_max_range(range_values) > 0.0:
        return float(range_values[0]), float(range_values[1])

    return -fallback, fallback


def _configure_keyboard_command_cfg(env_cfg):
    if not hasattr(env_cfg, "commands") or not hasattr(env_cfg.commands, "base_velocity"):
        return

    command_cfg = env_cfg.commands.base_velocity
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_heading_envs = 0.0
    command_cfg.heading_command = False


def _create_keyboard_controller(env):
    if not hasattr(env.unwrapped, "command_manager"):
        print("[INFO]: Keyboard control disabled. The environment has no command manager.")
        return None

    try:
        command_term = env.unwrapped.command_manager.get_term("base_velocity")
    except Exception:
        print("[INFO]: Keyboard control disabled. The environment has no base_velocity command term.")
        return None

    command_ranges = (
        _resolve_keyboard_axis_range(command_term.cfg, "lin_vel_x", args_cli.keyboard_vx, 0.4),
        _resolve_keyboard_axis_range(command_term.cfg, "lin_vel_y", args_cli.keyboard_vy, 0.25),
        _resolve_keyboard_axis_range(command_term.cfg, "ang_vel_z", args_cli.keyboard_wz, 0.8),
    )
    command_rates = (
        abs(args_cli.keyboard_accel),
        abs(args_cli.keyboard_accel),
        abs(args_cli.keyboard_yaw_accel),
    )
    try:
        controller = KeyboardVelocityController(
            command_ranges, command_rates, args_cli.keyboard_push_velocity, env.unwrapped.device
        )
    except RuntimeError as exc:
        print(f"[INFO]: Keyboard control disabled. {exc}")
        return None

    print("[INFO]: Keyboard control enabled: W/S forward/back, A/D lateral, Q/E yaw, X stop.")
    print("[INFO]: Manual push enabled: I/K forward/back, J/L left/right, U/O/N/M diagonals.")
    print(
        "[INFO]: Keyboard command ranges:"
        f" vx={command_ranges[0]}, vy={command_ranges[1]}, wz={command_ranges[2]}"
    )
    print(f"[INFO]: Manual push velocity impulse: {args_cli.keyboard_push_velocity:.3f} m/s")
    return controller


def _apply_keyboard_command(env, keyboard_controller: KeyboardVelocityController, dt: float):
    command = keyboard_controller.advance(dt)
    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    command_term.vel_command_b[:] = command
    if hasattr(command_term, "is_standing_env"):
        command_term.is_standing_env[:] = False
    if hasattr(command_term, "is_heading_env"):
        command_term.is_heading_env[:] = False

    push = keyboard_controller.pop_push()
    if torch.any(push != 0.0):
        robot = env.unwrapped.scene["robot"]
        root_vel = robot.data.root_vel_w.clone()
        root_vel[:, 0] += push[0]
        root_vel[:, 1] += push[1]
        robot.write_root_velocity_to_sim(root_vel)
        print(f"[PUSH] root velocity impulse: vx={push[0].item():+.3f}, vy={push[1].item():+.3f} m/s")


def main():

# --- 强行插入打印代码 ---
    print("\n" + "="*60)
    print("=== 正在强制读取机器人关节顺序 ===")
    # 注意：此时可能还没 env，我们先不读 robot
    # 只要这行打印出来了，说明脚本跑对了，我们再写下一步
    print("代码已运行到 main() 函数入口")
    print("="*60 + "\n")
    # -----------------------

    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    if not args_cli.disable_keyboard_control and not getattr(args_cli, "headless", False):
        _configure_keyboard_command_cfg(env_cfg)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = None
    log_dir = log_root_path
    if not args_cli.zero_action:
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    obs_csv_file = None
    obs_csv_writer = None
    if args_cli.print_obs:
        obs_csv_file = open("policy_obs_log.csv", "w", newline="")
        obs_csv_writer = csv.writer(obs_csv_file)
        header = ["step"] + [f"obs_{i}" for i in range(225)]
        obs_csv_writer.writerow(header)
        print("[INFO] Recording policy observations to policy_obs_log.csv")
    # -------------------------------
    # IMU debug setup
    # -------------------------------
    base_env = env.unwrapped
    robot = base_env.scene["robot"]

    imu_debug_interval = 20   # 每 20 个 policy step 打印一次
    imu_debug_step = 0

    print("[IMU DEBUG] robot body names:", robot.body_names)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    keyboard_controller = None
    if not args_cli.disable_keyboard_control and not getattr(args_cli, "headless", False):
        keyboard_controller = _create_keyboard_controller(env)
    elif getattr(args_cli, "headless", False):
        print("[INFO]: Keyboard control disabled in headless mode.")

    if args_cli.zero_action:
        print("[INFO]: Running in zero-action mode. No policy checkpoint will be loaded.")
        policy = None
    else:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            from rsl_rl.runners import DistillationRunner

            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # extract the neural network module
        # we do this in a try-except to maintain backwards compatibility.
        try:
            # version 2.3 onwards
            policy_nn = runner.alg.policy
        except AttributeError:
            # version 2.2 and below
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export policy to onnx/jit
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = _get_observations(env)
    timestep = 0

    # 在循环开始前初始化一个步数计数器（如果没有的话）
    if not hasattr(env.unwrapped, "_print_obs_step"):
        env.unwrapped._print_obs_step = 0

    # 在 obs 获取之后
    if args_cli.print_obs and env.unwrapped._print_obs_step % args_cli.print_obs_interval == 0:
        # 提取策略观测张量
        policy_obs = _get_policy_obs_tensor(obs)  # shape: (num_envs, 225)
        # 取第一个环境的观测
        obs0 = policy_obs[0].detach().cpu().numpy()
        print(f"\n[OBS] step={env.unwrapped._print_obs_step} shape={obs0.shape}")
        print("min={:.6f} max={:.6f} mean={:.6f} std={:.6f}".format(
            obs0.min(), obs0.max(), obs0.mean(), obs0.std()))
        # 打印前10个和后10个元素
        print("first 10:", obs0[:10])
        print("last 10: ", obs0[-10:])
        # 如果需要打印全部225个值（可能很多），可以取消注释下一行
        # print("full:", obs0)
        print("=====================================\n")

    env.unwrapped._print_obs_step += 1

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        if keyboard_controller is not None:
            _apply_keyboard_command(env, keyboard_controller, dt)
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if args_cli.zero_action:
                action_dim = env.unwrapped.action_manager.total_action_dim
                actions = torch.zeros((env.num_envs, action_dim), device=env.unwrapped.device)
            else:
                actions = policy(obs)


            if obs_csv_writer is not None:
                policy_obs_tensor = _get_policy_obs_tensor(obs)
                obs0 = policy_obs_tensor[0].detach().cpu().numpy().reshape(-1)
                if len(obs0) == 225:
                    obs_csv_writer.writerow([imu_debug_step] + obs0.tolist())
                else:
                    print(f"[WARNING] Obs length {len(obs0)} != 225, skip logging.")
            # -------------------------------
            # -------------------------------
            # ACTION DELTA DEBUG
            # 放在 actions = policy(obs) 后面，env.step(actions) 前面
            # -------------------------------

            # 如果你有 clamp，就先 clamp，再做 delta debug
            # actions = torch.clamp(actions, -2.0, 2.0)

            if not hasattr(env.unwrapped, "_debug_last_actions"):
                env.unwrapped._debug_last_actions = actions.clone()

            prev_actions = env.unwrapped._debug_last_actions.clone()
            action_delta = actions - prev_actions
            env.unwrapped._debug_last_actions = actions.clone()

            if imu_debug_step % 20 == 0:
                print("\n[ACTION DELTA DEBUG]")
                print("step =", imu_debug_step)

                print("action min/max/std:",
                    actions.min().item(),
                    actions.max().item(),
                    actions.std().item())

                print("delta min/max/std:",
                    action_delta.min().item(),
                    action_delta.max().item(),
                    action_delta.std().item())

                print("delta l2 mean:",
                    torch.sum(action_delta ** 2, dim=1).mean().item())

                print("action prev:", prev_actions[0].detach().cpu().numpy())
                print("action now :", actions[0].detach().cpu().numpy())
                print("delta      :", action_delta[0].detach().cpu().numpy())
                    # env stepping
            obs, _, _, _ = env.step(actions)
            _print_action_clip_debug(env, imu_debug_step, interval=20)
            # -------------------------------
            # IMU debug: robot.data + policy obs
            # -------------------------------
            if imu_debug_step % imu_debug_interval == 0:
                # 方法一：IsaacLab robot.data 的干净 IMU-like 状态
                gyro_b = robot.data.root_ang_vel_b[0].detach().cpu()
                projected_gravity_b = robot.data.projected_gravity_b[0].detach().cpu()

                # 方法二：policy 实际输入 observation
                policy_obs = _get_policy_obs_tensor(obs)

                if policy_obs.ndim == 1:
                    policy_obs_0 = policy_obs.detach().cpu()
                else:
                    policy_obs_0 = policy_obs[0].detach().cpu()

                print("\n================ IMU DEBUG ================")
                print(f"step = {imu_debug_step}")
                print("policy_obs type :", type(policy_obs))
                print("policy_obs shape:", tuple(policy_obs.shape))

                print("\n[robot.data clean state]")
                print("root_ang_vel_b raw              :", gyro_b.numpy())
                print("root_ang_vel_b * 0.2            :", (gyro_b * 0.2).numpy())
                print("projected_gravity_b             :", projected_gravity_b.numpy())

                # 你的 policy obs shape 是 225：
                # 0:15  base_ang_vel history, 5 * 3
                # 15:30 projected_gravity history, 5 * 3
                if policy_obs_0.numel() >= 30:
                    base_ang_vel_hist = policy_obs_0[0:15].reshape(5, 3)
                    projected_gravity_hist = policy_obs_0[15:30].reshape(5, 3)

                    print("\n[policy obs latest frame]")
                    print("base_ang_vel latest in obs      :", base_ang_vel_hist[-1].numpy())
                    print("projected_gravity latest in obs :", projected_gravity_hist[-1].numpy())

                    print("\n[policy obs history]")
                    print("base_ang_vel_hist:")
                    print(base_ang_vel_hist.numpy())
                    print("projected_gravity_hist:")
                    print(projected_gravity_hist.numpy())
                else:
                    print("[WARNING] policy_obs has fewer than 30 elements, cannot parse IMU history.")

                print("===========================================\n")

            imu_debug_step += 1
        if args_cli.debug_base and timestep % max(args_cli.debug_base_interval, 1) == 0:
            robot = env.unwrapped.scene["robot"]
            root_z = robot.data.root_pos_w[0, 2].item()
            projected_gravity = robot.data.projected_gravity_b[0].detach().cpu().numpy()
            orientation_penalty = float((robot.data.projected_gravity_b[0, :2] ** 2).sum().item())
            target_height = getattr(getattr(env.unwrapped.cfg.rewards, "base_height", None), "params", {}).get(
                "target_height", None
            )
            if target_height is not None:
                base_height_penalty = float((root_z - target_height) ** 2)
                base_height_str = f"{base_height_penalty:.6f} (target={target_height:.3f})"
            else:
                base_height_str = "N/A"
            print(
                "[DEBUG_BASE]"
                f" step={timestep}"
                f" root_z={root_z:.6f}"
                f" projected_gravity_b={projected_gravity}"
                f" flat_orientation_l2={orientation_penalty:.6f}"
                f" base_height_l2={base_height_str}"
            )
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        else:
            timestep += 1

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    # 关闭 CSV 文件（必须在 env.close() 之前）
    if obs_csv_file is not None:
        obs_csv_file.close()
        print("[INFO] Closed policy observation log.")

    # close the simulator
    if keyboard_controller is not None:
        keyboard_controller.close()
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
    
