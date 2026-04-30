#pragma once

#include "unitree/dds_wrapper/robots/g1/g1.h"

// P1 shares the humanoid low-level DDS message layout with the g1 wrapper.
using LowCmd_t = unitree::robot::g1::publisher::LowCmd;
using LowState_t = unitree::robot::g1::subscription::LowState;
