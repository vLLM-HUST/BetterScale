#pragma once
// Build-bound opt-in for native, per-source whole-layer route plans.
// Metadata occupies existing aligned IPC padding after maximum hidden payload.
namespace ExpertRoutePlan {
constexpr int MIN_ROWS = 0;
constexpr int OFFSET_WORDS = 50176 + 4096 * 2048 / 2;
}
