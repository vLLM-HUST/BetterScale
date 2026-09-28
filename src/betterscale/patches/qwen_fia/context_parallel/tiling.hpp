/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// CANN9.0.1 FIA tiling ABI mirror; scheduler fields are populated by plan.py.
#pragma once
#include <cstdint>
#include <cstddef>
struct coreNode {
 int32_t startBIdx[26],startN1Idx[26],startS1Idx[26],startS2Idx[26];
 int32_t endBIdx[26],endN1Idx[26],endS1Idx[26],endS2Idx[26];
 int64_t firstSplitKVTaskLseOffset[26],firstSplitKVTaskOOffset[26];
};
struct splitNode {
 int32_t batchIdx[26],headStartIdx[26],headEndIdx[26],qStartIdx[26],qEndIdx[26],splitNum[26];
 int64_t lseTaskOffset[26],oTaskOffset[26];
};
struct FAInferTilingData {
 uint32_t numHeads,embeddingSize,embeddingSizeV,numBlocks,blockSize,maxQSeqlen,maxKvSeqlen,kvHeads,batch,maxNumBlocksPerBatch,firstBatchTaskNum,totalTaskNum,maskType;
 uint64_t mm1OutSize,smOnlineOutSize,mm2OutSize,UpdateSize,workSpaceSize;
 float scaleValue;uint64_t pseQ,pseKv;uint32_t padding3;
 int64_t preToken,nextToken;uint32_t sparseMode;
 uint64_t splitLseTotalSize,splitOTotalSize;
 uint32_t totalSplitNodeNum,needCoreNum,mainLoopTaskNum,tailLoopTaskNum,tailStartBatch,tailStartN2,tailKvNBlockTile;
 coreNode coreInfo;splitNode splitInfo;
};
static_assert(offsetof(FAInferTilingData,coreInfo)==200);
static_assert(offsetof(FAInferTilingData,splitInfo)==1448);
static_assert(sizeof(FAInferTilingData)==2488);

static_assert(offsetof(FAInferTilingData,needCoreNum)==172);
