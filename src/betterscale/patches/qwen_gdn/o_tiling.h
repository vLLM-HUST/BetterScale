#pragma once
#include <cstdint>
struct ChunkFwdOTilingData {
    int64_t shapeBatch;
    int64_t seqlen;
    int64_t kNumHead;
    int64_t vNumHead;
    int64_t kHeadDim;
    int64_t vHeadDim;
    int64_t chunkSize;
    int64_t isVariedLen;
    int64_t tokenBatch;
    int64_t dataType;
    int64_t gDataType;
    int64_t vWorkspaceOffset;
    int64_t hWorkspaceOffset;
    int64_t attnWorkspaceOffset;
    int64_t aftermaskWorkspaceOffset;
    int64_t maskWorkspaceOffset;
    float scale;
    int64_t chunkCapacity;
};
