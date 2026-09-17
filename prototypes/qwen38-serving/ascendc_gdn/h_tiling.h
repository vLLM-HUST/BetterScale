#pragma once
#include <cstdint>
struct ChunkGatedDeltaRuleFwdHTilingData {
    int64_t batch;
    int64_t seqlen;
    int64_t kNumHead;
    int64_t vNumHead;
    int64_t kHeadDim;
    int64_t vHeadDim;
    int64_t chunkSize;
    int64_t initalStateStride0;
    bool useInitialState;
    bool storeFinalState;
    int64_t dataType;
    int64_t gDataType;
    int64_t stateDataType;
    int64_t isVariedLen;
    int64_t shapeBatch;
    int64_t tokenBatch;
    int64_t vWorkspaceOffset;
    int64_t vUpdateWorkspaceOffset;
    int64_t hWorkspaceOffset;
    int64_t numSeqWorkspaceOffset;
    int64_t numChunksWorkspaceOffset;
    int64_t chunkCapacity;
    bool statePoolMode;
};
