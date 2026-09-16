#pragma once
// Homogeneous token frames carry an explicit class, independent of row count.
// Layer affects batch compatibility, never service priority. No running GEMM
// is preempted. A bounded decode burst grants the oldest waiting prefill a
// turn.
namespace Persistent {
__aicore__ inline bool Promoted(int generation, int signal) {
  return generation > 0 && generation == signal;
}
struct PriorityPolicy {
  int decodeBurst = 0;
  int turn = 0;
  int nextTicket = 0;
  static constexpr int MAX_DECODE_BURST = 3;

  __aicore__ inline int Choose(const bool *pending, const int *priority) const {
    int decode = -1, prefill = -1;
    for (int j = 0; j < 2; ++j) {
      int c = (turn + j) % 2;
      if (!pending[c])
        continue;
      if (priority[c] == 0 && decode < 0)
        decode = c;
      if (priority[c] == 1 && prefill < 0)
        prefill = c;
    }
    if (prefill >= 0 && (decode < 0 || decodeBurst >= MAX_DECODE_BURST))
      return prefill;
    return decode;
  }
  // An already-staged prefill must not fall out of the fairness accounting
  // merely because later decode frames keep reusing the other slot.
  __aicore__ inline bool ProtectWaitingPrefill(int priority, int &serviceRank) {
    if (priority != 1 || serviceRank < 0 || decodeBurst < MAX_DECODE_BURST)
      return false;
    serviceRank = -1;
    decodeBurst = 0;
    return true;
  }
  __aicore__ inline bool Grant(int source, int priority) {
    bool protectedTurn = priority == 1 && decodeBurst >= MAX_DECODE_BURST;
    decodeBurst =
        priority == 0
            ? (decodeBurst < MAX_DECODE_BURST ? decodeBurst + 1 : decodeBurst)
            : 0;
    turn = 1 - source;
    return protectedTurn;
  }
};
__aicore__ inline bool PriorityBefore(int rankA, int ticketA, int rankB,
                                      int ticketB) {
  return rankA < rankB || (rankA == rankB && ticketA < ticketB);
}
} // namespace Persistent
