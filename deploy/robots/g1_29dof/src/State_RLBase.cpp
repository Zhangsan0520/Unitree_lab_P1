#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static const float vx_pos = cfg["lin_vel_x"][1].as<float>();
    static const float vx_neg = cfg["lin_vel_x"][0].as<float>();
    static const float vy_pos = cfg["lin_vel_y"][1].as<float>();
    static const float vy_neg = cfg["lin_vel_y"][0].as<float>();
    static const float wz_pos = cfg["ang_vel_z"][1].as<float>();
    static const float wz_neg = cfg["ang_vel_z"][0].as<float>();

    // 目标命令改成“静态保持”
    static std::vector<float> target_cmd = {0.0f, 0.0f, 0.0f};
    static std::vector<float> cmd = {0.0f, 0.0f, 0.0f};

    // 单键锁存控制
    if (key == "w") target_cmd = {vx_pos, 0.0f, 0.0f};
    else if (key == "s") target_cmd = {vx_neg, 0.0f, 0.0f};
    else if (key == "a") target_cmd = {0.0f, vy_pos, 0.0f};
    else if (key == "d") target_cmd = {0.0f, vy_neg, 0.0f};
    else if (key == "q") target_cmd = {0.0f, 0.0f, 1.5f * wz_pos};
    else if (key == "e") target_cmd = {0.0f, 0.0f, 1.5f * wz_neg};
    else if (key == "x") target_cmd = {0.0f, 0.0f, 0.0f};   // 停止/清零

    auto step_towards = [](float current, float target, float max_delta) -> float {
        float diff = target - current;
        if (diff > max_delta) diff = max_delta;
        if (diff < -max_delta) diff = -max_delta;
        return current + diff;
    };

    const float dvx = 0.01f;
    const float dvy = 0.01f;
    const float dwz = 0.015f;

    cmd[0] = step_towards(cmd[0], target_cmd[0], dvx);
    cmd[1] = step_towards(cmd[1], target_cmd[1], dvy);
    cmd[2] = step_towards(cmd[2], target_cmd[2], dwz);

    return cmd;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}