#include <cassert>
#define __aicore__
#include "../../device-service/priority_policy.hpp"
using namespace Persistent;
int main() {
  assert(Promoted(4, 4));
  assert(!Promoted(4, 3));
  assert(!Promoted(4, 5));
  assert(!Promoted(0, 0));
  PriorityPolicy p;
  bool both[2] = {true, true};
  int classes[2] = {1, 0};
  for (int cycle = 0; cycle < 100; cycle++) {
    for (int j = 0; j < 3; j++) {
      assert(p.Choose(both, classes) == 1);
      p.Grant(1, 0);
    }
    assert(p.Choose(both, classes) == 0);
    assert(p.Grant(0, 1));
  }
  int promoted[2] = {0, 0};
  p.turn = 0;
  assert(p.Choose(both, promoted) == 0);
  p.Grant(0, 1);
  assert(p.Choose(both, promoted) == 1);
  assert(PriorityBefore(-1, 99, 0, 1));
  assert(PriorityBefore(0, 9, 1, 0));
  assert(PriorityBefore(0, 1, 0, 2));
  p.decodeBurst = 0;
  int stagedRank = 1;
  for (int j = 0; j < 2; j++) {
    p.Grant(1, 0);
    assert(!p.ProtectWaitingPrefill(1, stagedRank));
  }
  p.Grant(1, 0);
  assert(p.ProtectWaitingPrefill(1, stagedRank));
  assert(stagedRank == -1);
  assert(p.decodeBurst == 0);
  bool alone[2] = {true, false};
  assert(p.Choose(alone, classes) == 0);
  bool none[2] = {false, false};
  assert(p.Choose(none, classes) == -1);
}
