#pragma once

#include "FSM/FSMState.h"
#include "LinearInterpolator.h"

class State_FixStandP1 : public FSMState
{
public:
    State_FixStandP1(int state, std::string state_string = "FixStand")
    : FSMState(state, state_string)
    {
        ts_ = param::config["FSM"]["FixStand"]["ts"].as<std::vector<float>>();
        qs_ = param::config["FSM"]["FixStand"]["qs"].as<std::vector<std::vector<float>>>();

        this->registered_checks.emplace_back(std::make_pair(
            []() -> bool {
                return FSMState::keyboard && FSMState::keyboard->on_pressed && FSMState::keyboard->key() == "p";
            },
            FSMStringMap.right.at("Passive")
        ));

        this->registered_checks.emplace_back(std::make_pair(
            []() -> bool {
                return FSMState::keyboard && FSMState::keyboard->on_pressed && FSMState::keyboard->key() == "v";
            },
            FSMStringMap.right.at("Velocity")
        ));
    }

    void enter()
    {
        static auto kp = param::config["FSM"]["FixStand"]["kp"].as<std::vector<float>>();
        static auto kd = param::config["FSM"]["FixStand"]["kd"].as<std::vector<float>>();
        for (int i = 0; i < kp.size(); ++i)
        {
            auto& motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = kp[i];
            motor.kd() = kd[i];
            motor.dq() = 0.0f;
            motor.tau() = 0.0f;
        }

        std::vector<float> q0;
        for (int i = 0; i < kp.size(); ++i) {
            q0.push_back(lowcmd->msg_.motor_cmd()[i].q());
        }
        qs_[0] = q0;
        t0_ = (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3;
    }

    void run()
    {
        float t = (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3 - t0_;
        auto q = linear_interpolate(t, ts_, qs_);

        for (int i = 0; i < q.size(); ++i) {
            lowcmd->msg_.motor_cmd()[i].q() = q[i];
        }
    }

private:
    double t0_ = 0.0;
    std::vector<float> ts_;
    std::vector<std::vector<float>> qs_;
};

REGISTER_FSM(State_FixStandP1)
