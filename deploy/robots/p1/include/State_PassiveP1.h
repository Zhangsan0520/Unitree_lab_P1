#pragma once

#include "FSM/FSMState.h"

class State_PassiveP1 : public FSMState
{
public:
    State_PassiveP1(int state, std::string state_string = "Passive")
    : FSMState(state, state_string)
    {
        auto motor_mode = param::config["FSM"]["Passive"]["mode"];
        if (motor_mode.IsDefined())
        {
            auto values = motor_mode.as<std::vector<int>>();
            for (int i = 0; i < values.size(); ++i)
            {
                lowcmd->msg_.motor_cmd()[i].mode() = values[i];
            }
        }

        this->registered_checks.emplace_back(std::make_pair(
            []() -> bool {
                return FSMState::keyboard && FSMState::keyboard->on_pressed && FSMState::keyboard->key() == "f";
            },
            FSMStringMap.right.at("FixStand")
        ));
    }

    void enter()
    {
        static auto kd = param::config["FSM"]["Passive"]["kd"].as<std::vector<float>>();
        static auto centering_kp = param::config["FSM"]["Passive"]["centering_kp"].as<std::vector<float>>();
        static auto centering_q = param::config["FSM"]["Passive"]["centering_q"].as<std::vector<float>>();
        for (int i = 0; i < kd.size(); ++i)
        {
            auto& motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = centering_kp[i];
            motor.kd() = kd[i];
            motor.dq() = 0.0f;
            motor.tau() = 0.0f;
            motor.q() = centering_kp[i] > 0.0f ? centering_q[i] : lowstate->msg_.motor_state()[i].q();
        }
    }

    void run()
    {
        static auto centering_kp = param::config["FSM"]["Passive"]["centering_kp"].as<std::vector<float>>();
        static auto centering_q = param::config["FSM"]["Passive"]["centering_q"].as<std::vector<float>>();
        for (int i = 0; i < lowcmd->msg_.motor_cmd().size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].q() =
                centering_kp[i] > 0.0f ? centering_q[i] : lowstate->msg_.motor_state()[i].q();
        }
    }
};

REGISTER_FSM(State_PassiveP1)
