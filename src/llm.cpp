#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-network.hpp"
#include "mmap.hpp"
#include "llm.hpp"
#include <stdexcept>
#include <string>

static const char *hiddenActToString(LlmHiddenAct act) {
    if (act == HIDDEN_ACT_GELU) return "Gelu";
    if (act == HIDDEN_ACT_SILU) return "Silu";
    throw std::runtime_error("Unsupported hidden act");
}

static const char *ropeTypeToString(NnRopeType type) {
    if (type == ROPE_LLAMA) return "Llama";
    if (type == ROPE_LLAMA3_1) return "Llama3.1";
    if (type == ROPE_FALCON) return "Falcon";
    throw std::runtime_error("Unsupported rope type");
}

static const char *archTypeToString(LlmArchType type) {
    if (type == LLAMA) return "Llama";
    if (type == QWEN3) return "Qwen3";
    throw std::runtime_error("Unsupported architecture");
}

static float convertNormEpsilon(int value) {
    if (value == 5) return 1e-05f;
    if (value == 6) return 1e-06f;
    throw std::runtime_error("Unsupported norm epsilon");
}

void buildFfnSegment(NnSegmentConfigBuilder& ff, LlmHeader* h, NnUint layerIndex,
                           NnUint yBufferIndex, NnUint yqBufferIndex, NnUint dBufferIndex,
                           NnUint dqBufferIndex, NnUint lBufferIndex, 
                           NnRowMatmulSlice& w1Slice, NnColMatmulSlice& w2Slice, NnRowMatmulSlice& w3Slice,
                           std::string expert_id) {
    if (yBufferIndex != yqBufferIndex) {
        ff.addOp(
            OP_CAST, "block_cast_y3", layerIndex,
            pointerBatchConfig(SRC_BUFFER, yBufferIndex),
            pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
            size0(),
            NnCastOpCodeConfig{});
    }
    std::string w1_str = "block_matmul_w1" + expert_id;
    std::string w2_str = "block_matmul_w2" + expert_id;
    std::string w3_str = "block_matmul_w3" + expert_id;

    ff.addOp(
        OP_MATMUL, w1_str.c_str(), layerIndex,
        pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
        pointerBatchConfig(SRC_BUFFER, dBufferIndex),
        size2D(h->weightType, w1Slice.n, w1Slice.d0),
        NnMatmulOpConfig{});
    ff.addOp(
        OP_MATMUL, w3_str.c_str(), layerIndex,
        pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
        pointerBatchConfig(SRC_BUFFER, lBufferIndex),
        size2D(h->weightType, w3Slice.n, w3Slice.d0),
        NnMatmulOpConfig{});
    ff.addOp(
        OP_SILU, "block_act", layerIndex,
        pointerBatchConfig(SRC_BUFFER, dBufferIndex),
        pointerBatchConfig(SRC_BUFFER, dBufferIndex),
        size0(),
        NnSiluOpCodeConfig{});
    ff.addOp(
        OP_MUL, "block_mul", layerIndex,
        pointerBatchConfig(SRC_BUFFER, dBufferIndex),
        pointerBatchConfig(SRC_BUFFER, dBufferIndex),
        size0(),
        NnMulOpCodeConfig{lBufferIndex});
    if (dBufferIndex != dqBufferIndex) {
        ff.addOp(
            OP_CAST, "block_cast_d2", layerIndex,
            pointerBatchConfig(SRC_BUFFER, dBufferIndex),
            pointerBatchConfig(SRC_BUFFER, dqBufferIndex),
            size0(),
            NnCastOpCodeConfig{});
    }
    ff.addOp(
        OP_MATMUL, w2_str.c_str(), layerIndex,
        pointerBatchConfig(SRC_BUFFER, dqBufferIndex),
        pointerBatchConfig(SRC_BUFFER, yBufferIndex),
        size2D(h->weightType, w2Slice.n0, w2Slice.d),
        NnMatmulOpConfig{});
}

void buildMoeSegment(NnSegmentConfigBuilder& ff, LlmHeader* h, NnUint layerIndex,
                           NnUint yBufferIndex, NnUint yqBufferIndex, NnUint routerLogitsBufferIndex,
                           NnUint expertIndicesBufferIndex, NnUint routingWeightsBufferIndex,
                           NnUint* expertBufferIndices, NnUint *weightVectorBufferIndices,
                           NnUint* dqBufferIndices, NnUint* lBufferIndices,
                           NnRowMatmulSlice& routerSlice,
                           NnRowMatmulSlice* expertW1Slices, NnColMatmulSlice* expertW2Slices, 
                           NnRowMatmulSlice* expertW3Slices) {

    // GENERALIZED MOE: Router + N experts + top-k selection + weighted accumulation
    // 1. Router computes logits for ALL N experts, selects top-k, computes softmax over top-k
    // 2. Top-k experts compute their outputs (always process k experts regardless of actual expert IDs)
    // 3. Weighted sum combines top-k expert outputs using routing weights

    // === ROUTER OPERATION ===
    // Router: input -> logits for ALL N experts -> top-k selection -> softmax weights over top-k
    ff.addOp(OP_ROUTER, "block_router", layerIndex,
             pointerBatchConfig(SRC_BUFFER, yqBufferIndex),           // Input
             pointerBatchConfig(SRC_BUFFER, routerLogitsBufferIndex), // Router logits output (all N experts)
             size2D(F_32, routerSlice.n, routerSlice.d0),
             NnRouterOpConfig{h->nExperts, h->nActiveExperts, 
                             expertIndicesBufferIndex, routingWeightsBufferIndex,
                             size2D(F_32, routerSlice.n, routerSlice.d0)});  // Router weights must be F32

    // === TOP-K EXPERT COMPUTATIONS ===
    // Process top-k active experts sequentially
    // We always process exactly k experts, but their weights come from different expert IDs
    
    for (NnUint k = 0; k < h->nActiveExperts; k++) {
        char expertSuffix[8];
        snprintf(expertSuffix, sizeof(expertSuffix), ".%d", k);
        
        // Expert k W1 matmul: input -> hidden
        ff.addOp(
            OP_MATMUL, ("block_matmul_w1" + std::string(expertSuffix)).c_str(), layerIndex,
            pointerBatchConfig(SRC_BUFFER, yqBufferIndex),                    // Input
            pointerBatchConfig(SRC_BUFFER, weightVectorBufferIndices[k]),     // Hidden
            size2D(h->weightType, expertW1Slices[k].n, expertW1Slices[k].d0),
            NnMatmulOpConfig{});

        // Expert k W3 matmul: input -> gate
        ff.addOp(
            OP_MATMUL, ("block_matmul_w3" + std::string(expertSuffix)).c_str(), layerIndex,
            pointerBatchConfig(SRC_BUFFER, yqBufferIndex),                    // Input
            pointerBatchConfig(SRC_BUFFER, lBufferIndices[k]),                // Gate
            size2D(h->weightType, expertW3Slices[k].n, expertW3Slices[k].d0),
            NnMatmulOpConfig{});

        // Expert k SiLU activation: hidden -> SiLU(hidden) (in-place)
        ff.addOp(
            OP_SILU, ("expert" + std::to_string(k) + "_silu").c_str(), layerIndex,
            pointerBatchConfig(SRC_BUFFER, weightVectorBufferIndices[k]),
            pointerBatchConfig(SRC_BUFFER, weightVectorBufferIndices[k]),     // in-place
            size0(),
            NnSiluOpCodeConfig{});

        // Expert k Element-wise multiply: SiLU(W1) * W3
        ff.addOp(
            OP_MUL, ("expert" + std::to_string(k) + "_mul").c_str(), layerIndex,
            pointerBatchConfig(SRC_BUFFER, weightVectorBufferIndices[k]),
            pointerBatchConfig(SRC_BUFFER, weightVectorBufferIndices[k]),     // in-place
            size0(),
            NnMulOpCodeConfig{lBufferIndices[k]});                            // multiplier buffer

        // Expert k Cast to quantized buffer (like dense FFN)
        if (weightVectorBufferIndices[k] != dqBufferIndices[k]) {
            ff.addOp(
                OP_CAST, ("expert" + std::to_string(k) + "_cast").c_str(), layerIndex,
                pointerBatchConfig(SRC_BUFFER, weightVectorBufferIndices[k]),
                pointerBatchConfig(SRC_BUFFER, dqBufferIndices[k]),
                size0(),
                NnCastOpCodeConfig{});
        }

        // Expert k W2 matmul: hidden -> output (use quantized buffer like dense FFN)
        ff.addOp(
            OP_MATMUL, ("block_matmul_w2" + std::string(expertSuffix)).c_str(), layerIndex,
            pointerBatchConfig(SRC_BUFFER, dqBufferIndices[k]),               // Use quantized buffer (Q80)
            pointerBatchConfig(SRC_BUFFER, expertBufferIndices[k]),           // Expert k final output buffer
            size2D(h->weightType, expertW2Slices[k].n0, expertW2Slices[k].d),
            NnMatmulOpConfig{});
    }

    // === GENERALIZED WEIGHTED ACCUMULATION ===
    // Use OP_WEIGHTED_SUM for any number of experts k >= 1
    // Computes: output = sum(weight[k] * expert[k]) for k=0..nActiveExperts-1
    
    // Copy expert buffer indices into config struct (fixed array)
    NnWeightedSumOpConfig weightedSumConfig;
    weightedSumConfig.nActiveExperts = h->nActiveExperts;
    weightedSumConfig.routingWeightsBufferIndex = routingWeightsBufferIndex;
    for (NnUint k = 0; k < h->nActiveExperts; k++) {
        weightedSumConfig.expertBufferIndices[k] = expertBufferIndices[k];
    }
    
    // Note: We use a dummy input since OP_WEIGHTED_SUM accesses experts via config->expertBufferIndices
    ff.addOp(
        OP_WEIGHTED_SUM, "moe_weighted_sum", layerIndex,
        pointerBatchConfig(SRC_BUFFER, yqBufferIndex),            // Dummy input (not used)
        pointerBatchConfig(SRC_BUFFER, yBufferIndex),             // Final output
        size0(),
        weightedSumConfig);
}

LlmHeader loadLlmHeader(const char *path, const NnUint maxSeqLen, NnFloatType syncType) {
    LlmHeader header;
    std::memset(&header, 0, sizeof(LlmHeader));
    header.weightType = F_UNK;
    header.hiddenAct = HIDDEN_ACT_SILU;
    header.ropeType = ROPE_LLAMA;
    header.ropeTheta = 10000.0f;
    header.ropeScalingFactor = 1.0f;
    header.normEpsilon = 1e-5f;

    std::unique_ptr<FILE, int(*)(FILE *)> fdPtr(fopen(path, "rb"), fclose);
    FILE *fd = fdPtr.get();
    if (fd == NULL)
        throw std::runtime_error("Cannot open model file");

    int magic;
    if (fread(&magic, sizeof(int), 1, fd) != 1)
        throw std::runtime_error("Cannot read magic value");

    if (magic == 0xABCD00 || magic == 0xABCD01)
        throw std::runtime_error("Old model format is not supported");
    if (magic != 0xA00ABCD)
        throw std::runtime_error("Unsupported magic number");

    if (fread(&header.headerSize, sizeof(int), 1, fd) != 1)
        throw std::runtime_error("Cannot read header size");

    std::vector<int> bufferPtr(header.headerSize);
    int *buffer = &bufferPtr[0];
    if (fread(buffer, header.headerSize, 1, fd) != 1)
        throw std::runtime_error("Cannot read header values");

    int nKv = (header.headerSize - 2 * sizeof(int)) / sizeof(int);

    for (int i = 0; i < nKv; i += 2) {
        int key = buffer[i];
        int value = buffer[i + 1];
        if (key == VERSION) header.version = value;
        else if (key == ARCH_TYPE) header.archType = (LlmArchType)value;
        else if (key == DIM) header.dim = value;
        else if (key == HIDDEN_DIM) header.hiddenDim = value;
        else if (key == N_LAYERS) header.nLayers = value;
        else if (key == N_HEADS) header.nHeads = value;
        else if (key == N_KV_HEADS) header.nKvHeads = value;
        else if (key == N_EXPERTS) header.nExperts = value;
        else if (key == N_ACTIVE_EXPERTS) header.nActiveExperts = value;
        else if (key == VOCAB_SIZE) header.vocabSize = value;
        else if (key == SEQ_LEN) header.seqLen = value;
        else if (key == HIDDEN_ACT) header.hiddenAct = (LlmHiddenAct)value;
        else if (key == ROPE_THETA) header.ropeTheta = (float)value;
        else if (key == WEIGHT_FLOAT_TYPE) header.weightType = (NnFloatType)value;
        else if (key == ROPE_SCALING_FACTOR) header.ropeScalingFactor = (float)value;
        else if (key == ROPE_SCALING_LOW_FREQ_FACTOR) header.ropeScalingLowFreqFactor = (float)value;
        else if (key == ROPE_SCALING_HIGH_FREQ_FACTORY) header.ropeScalingHighFreqFactory = (float)value;
        else if (key == ROPE_SCALING_ORIG_MAX_SEQ_LEN) header.ropeScalingOrigMaxSeqLen = value;
        else if (key == ROPE_TYPE) header.ropeType = (NnRopeType)value;
        else if (key == HEAD_DIM) header.headDim = value;
        else if (key == NORM_EPSILON) header.normEpsilon = convertNormEpsilon(value);
        else throw std::runtime_error("Unsupported header key");
    }

    if (header.weightType == F_UNK)
        throw std::runtime_error("Model does not specify weight type");

    header.origSeqLen = header.seqLen;
    if (maxSeqLen > 0 && header.seqLen > maxSeqLen)
        header.seqLen = maxSeqLen;

    if (header.headDim == 0)
        header.headDim = header.dim / header.nHeads;
    header.qDim = header.headDim * header.nHeads;
    header.kvDim = header.headDim * header.nKvHeads;
    header.syncType = syncType;
    header.fileSize = (NnSize)seekToEnd(fd);

    if (header.archType == QWEN3)
        header.ropeType = ROPE_FALCON;
    return header;
}

void printLlmHeader(LlmHeader *header) {
    printf("💡 Arch: %s\n", archTypeToString(header->archType));
    printf("💡 HiddenAct: %s\n", hiddenActToString(header->hiddenAct));
    printf("💡 Dim: %u\n", header->dim);
    printf("💡 HeadDim: %u\n", header->headDim);
    printf("💡 QDim: %u\n", header->qDim);
    printf("💡 KvDim: %u\n", header->kvDim);
    printf("💡 HiddenDim: %u\n", header->hiddenDim);
    printf("💡 VocabSize: %u\n", header->vocabSize);
    printf("💡 nLayers: %u\n", header->nLayers);
    printf("💡 nHeads: %u\n", header->nHeads);
    printf("💡 nKvHeads: %u\n", header->nKvHeads);
    if (header->seqLen != header->origSeqLen) {
        printf("💡 OrigSeqLen: %u\n", header->origSeqLen);
    }
    printf("💡 SeqLen: %u\n", header->seqLen);
    printf("💡 NormEpsilon: %f\n", header->normEpsilon);
    printf("💡 RopeType: %s\n", ropeTypeToString(header->ropeType));
    printf("💡 RopeTheta: %.0f\n", header->ropeTheta);
    if (header->ropeType == ROPE_LLAMA3_1) {
        printf("💡 RopeScaling: f=%.1f, l=%.1f, h=%.1f, o=%d\n",
            header->ropeScalingFactor,
            header->ropeScalingLowFreqFactor,
            header->ropeScalingHighFreqFactory,
            header->ropeScalingOrigMaxSeqLen);
    }
}

LlmNet buildLlmNet(LlmHeader *h, NnUint nNodes, NnUint nBatches) {
    LlmNet n;
    n.tokenEmbeddingSize = size2D(F_32, h->vocabSize, h->dim);
    n.rmsNormSize = size1D(F_32, h->dim);
    n.qkRmsNormSize = size1D(F_32, h->headDim);

    NnKvCacheSlice kvCacheSlice = sliceKvCache(h->kvDim, h->seqLen, nNodes);
    NnMultiHeadAttSlice multiHeadAttSlice = sliceMultiHeadAtt(h->nHeads, h->seqLen, nNodes, nBatches);

    n.qSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->qDim);
    n.kSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->kvDim);
    n.vSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->kvDim);
    n.woSlice = sliceColMatmul(h->weightType, nNodes, h->qDim, h->dim);

    n.w1Slice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->hiddenDim);
    n.w2Slice = sliceColMatmul(h->weightType, nNodes, h->hiddenDim, h->dim);
    n.w3Slice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->hiddenDim);
    n.wclsSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->vocabSize);
    

    NnNetConfigBuilder netBuilder(nNodes, nBatches);

    n.positionPipeIndex = netBuilder.addPipe("POS", size2D(F_32, nBatches, 1));
    n.tokenPipeIndex = netBuilder.addPipe("TOK", size2D(F_32, nBatches, 1));
    n.xPipeIndex = netBuilder.addPipe("X", size2D(F_32, nBatches, h->dim));
    n.logitsPipeIndex = netBuilder.addPipe("LG", size2D(F_32, nBatches, h->vocabSize));
    const NnUint zqPipeIndex = netBuilder.addPipe("ZQ", size2D(h->syncType, nBatches, h->dim * nNodes));

    netBuilder.addPreSync(n.positionPipeIndex);

    n.header = h;
    n.netConfig = netBuilder.build();
    n.nodeConfigs = new NnNodeConfig[nNodes];

    for (NnUint nodeIndex = 0; nodeIndex < nNodes; nodeIndex++) {
        NnRopeSlice ropeSlice = sliceRope(h->ropeType, h->qDim, h->kvDim, h->nKvHeads, nNodes, h->seqLen, h->headDim, h->ropeTheta, nodeIndex);
        NnNodeConfigBuilder nodeBuilder(nodeIndex);

        const NnUint xBufferIndex = nodeBuilder.addBuffer("x", size2D(F_32, nBatches, h->dim));
        const NnUint yBufferIndex = nodeBuilder.addBuffer("y", size2D(F_32, nBatches, h->dim));
        const NnUint yqBufferIndex = h->syncType == F_32
            ? yBufferIndex
            : nodeBuilder.addBuffer("q_y", size2D(h->syncType, nBatches, h->dim));

        const NnUint zBufferIndex = nodeBuilder.addBuffer("z", size2D(F_32, nBatches, h->qDim));
        const NnUint zqSliceBufferIndex = nodeBuilder.addBuffer("q_z_slice", size2D(h->syncType, nBatches, h->qDim / nNodes));

        const NnUint qBufferIndex = nodeBuilder.addBuffer("q", size2D(F_32, nBatches, n.qSlice.d0));
        const NnUint kTempBufferIndex = nodeBuilder.addBuffer("k_temp", size2D(F_32, nBatches, n.kSlice.d0));
        const NnUint vTempBufferIndex = nodeBuilder.addBuffer("v_temp", size2D(F_32, nBatches, n.vSlice.d0));

        ASSERT_EQ(n.qSlice.d0 % n.header->headDim, 0);
        ASSERT_EQ(n.kSlice.d0 % n.header->headDim, 0);
        const NnUint nQNormColumns = n.qSlice.d0 / n.header->headDim;
        const NnUint nKNormColumns = n.kSlice.d0 / n.header->headDim;
        const NnUint nInvBufferColumns = std::max(nQNormColumns, nKNormColumns);

        const NnUint invRmsBufferIndex = nodeBuilder.addBuffer("inv_rms", size2D(F_32, nBatches,
            h->archType == QWEN3 ? nInvBufferColumns : 1));

        const NnUint dBufferIndex = nodeBuilder.addBuffer("d", size2D(F_32, nBatches, n.w1Slice.d0));
        const NnUint dqBufferIndex = h->syncType == F_32
            ? dBufferIndex
            : nodeBuilder.addBuffer("q_d", size2D(h->syncType, nBatches, n.w1Slice.d0));
        const NnUint lBufferIndex = nodeBuilder.addBuffer("l", size2D(F_32, nBatches, n.w3Slice.d0));
        
        // MOE-specific buffers (allocated conditionally for MOE models)
        NnUint routerLogitsBufferIndex = 0;
        NnUint expertIndicesBufferIndex = 0;
        NnUint routingWeightsBufferIndex = 0;
        NnUint dBufferIndices[8];        // Max 8 active experts for now
        NnUint weightVectorBufferIndices[8];
        NnUint dqBufferIndices[8];
        NnUint lBufferIndices[8];
        
        if (h->nExperts > 0) {
            // MOE model - allocate additional buffers
            routerLogitsBufferIndex = nodeBuilder.addBuffer("router_logits", size2D(F_32, nBatches, h->nExperts));
            expertIndicesBufferIndex = nodeBuilder.addBuffer("expert_indices", size2D(F_32, nBatches, h->nActiveExperts));
            routingWeightsBufferIndex = nodeBuilder.addBuffer("routing_weights", size2D(F_32, nBatches, h->nActiveExperts));
            
            // Per-expert intermediate buffers (follow same pattern as dense FFN)
            NnUint expertHiddenSize = 768; // Qwen3 MOE moe_intermediate_size
            
            for (NnUint k = 0; k < h->nActiveExperts && k < 8; k++) {
                char bufferName[64];
                
                // Expert final output buffer (F32 like dense dBufferIndex)
                snprintf(bufferName, sizeof(bufferName), "d_expert_%d", k);
                dBufferIndices[k] = nodeBuilder.addBuffer(bufferName, size2D(F_32, nBatches, h->dim));
                
                // Expert intermediate computation buffer (F32 like dense dBufferIndex) 
                snprintf(bufferName, sizeof(bufferName), "weight_vector_%d", k);
                weightVectorBufferIndices[k] = nodeBuilder.addBuffer(bufferName, size2D(F_32, nBatches, expertHiddenSize));
                
                // Expert intermediate quantized buffer (follows syncType like dense dqBufferIndex)
                snprintf(bufferName, sizeof(bufferName), "dq_expert_%d", k);
                dqBufferIndices[k] = (h->syncType == F_32) 
                    ? weightVectorBufferIndices[k]  // Use same buffer if no quantization
                    : nodeBuilder.addBuffer(bufferName, size2D(h->syncType, nBatches, expertHiddenSize));
                
                // Expert l buffer (F32 like dense lBufferIndex)
                snprintf(bufferName, sizeof(bufferName), "l_expert_%d", k);
                lBufferIndices[k] = nodeBuilder.addBuffer(bufferName, size2D(F_32, nBatches, expertHiddenSize));
            }
        }
        const NnUint ropeCacheBufferIndex = nodeBuilder.addBuffer("rope_cache", ropeSlice.cacheSize);
        const NnUint attBufferIndex = nodeBuilder.addBuffer("att", multiHeadAttSlice.attSize);
        const NnUint logitsSliceBufferIndex = nodeBuilder.addBuffer("lg", size2D(F_32, nBatches, h->vocabSize / nNodes));

        NnSegmentConfigBuilder start;
        if (nodeIndex == 0) {
            start.addOp(
                OP_EMBEDDING, "embedding", 0,
                pointerBatchConfig(SRC_PIPE, n.tokenPipeIndex),
                pointerBatchConfig(SRC_PIPE, n.xPipeIndex),
                n.tokenEmbeddingSize,
                NnEmbeddingOpConfig{});
        }
        start.addSync(n.xPipeIndex, SYNC_WITH_ROOT);
        nodeBuilder.addSegment(start.build());

        for (NnUint layerIndex = 0; layerIndex < h->nLayers; layerIndex++) {
            const NnUint kBufferIndex = nodeBuilder.addBuffer("k", kvCacheSlice.keySize);
            const NnUint vBufferIndex = nodeBuilder.addBuffer("v", kvCacheSlice.valueSize);

            NnSegmentConfigBuilder att;
            NnSegmentConfigBuilder ff;

            // att
            if (layerIndex == 0) {
                att.addOp(
                    OP_CAST, "block_cast_x", layerIndex,
                    pointerBatchConfig(SRC_PIPE, n.xPipeIndex),
                    pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                    size0(),
                    NnCastOpCodeConfig{});
            } else {
                att.addOp(
                    OP_MERGE_ADD, "block_merge_add", layerIndex,
                    pointerBatchConfig(SRC_PIPE, zqPipeIndex),
                    pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                    size0(),
                    NnMergeAddOpCodeConfig{});
            }

            att.addOp(
                OP_INV_RMS, "block_norm_pre_0", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                size0(),
                NnInvRmsOpConfig{h->normEpsilon, 1});
            att.addOp(
                OP_RMS_NORM, "block_norm_0", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                n.rmsNormSize,
                NnRmsNormOpConfig{invRmsBufferIndex, 1});
            if (yBufferIndex != yqBufferIndex) {
                att.addOp(
                    OP_CAST, "block_cast_y", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                    size0(),
                    NnCastOpCodeConfig{});
            }
            att.addOp(
                OP_MATMUL, "block_matmul_q", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                size2D(h->weightType, n.qSlice.n, n.qSlice.d0),
                NnMatmulOpConfig{});
            att.addOp(
                OP_MATMUL, "block_matmul_k", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                size2D(h->weightType, n.kSlice.n, n.kSlice.d0),
                NnMatmulOpConfig{});
            att.addOp(
                OP_MATMUL, "block_matmul_v", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                pointerBatchConfig(SRC_BUFFER, vTempBufferIndex),
                size2D(h->weightType, n.vSlice.n, n.vSlice.d0),
                NnMatmulOpConfig{});

            if (h->archType == QWEN3) {
                att.addOp(OP_INV_RMS, "block_norm_pre_q", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                    size0(),
                    NnInvRmsOpConfig{h->normEpsilon, nQNormColumns});
                att.addOp(
                    OP_RMS_NORM, "block_norm_q", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                    size2D(F_32, 1, n.header->headDim),
                    NnRmsNormOpConfig{invRmsBufferIndex, nQNormColumns});

                att.addOp(OP_INV_RMS, "block_norm_pre_k", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                    size0(),
                    NnInvRmsOpConfig{h->normEpsilon, nKNormColumns});
                att.addOp(
                    OP_RMS_NORM, "block_norm_k", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                    size2D(F_32, 1, n.header->headDim),
                    NnRmsNormOpConfig{invRmsBufferIndex, nKNormColumns});
            }

            att.addOp(
                OP_ROPE, "block_rope_q", layerIndex,
                pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                size0(),
                NnRopeOpConfig{n.header->ropeType, 1, n.positionPipeIndex, ropeCacheBufferIndex, 
                    h->ropeScalingFactor, h->ropeScalingLowFreqFactor, h->ropeScalingHighFreqFactory, h->ropeScalingOrigMaxSeqLen,
                    ropeSlice});
            att.addOp(
                OP_ROPE, "block_rope_k", layerIndex,
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                size0(),
                NnRopeOpConfig{n.header->ropeType, 0, n.positionPipeIndex, ropeCacheBufferIndex, 
                    h->ropeScalingFactor, h->ropeScalingLowFreqFactor, h->ropeScalingHighFreqFactory, h->ropeScalingOrigMaxSeqLen,
                    ropeSlice});
            att.addOp(
                OP_SHIFT, "block_shift_k", layerIndex,
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                pointerRawConfig(SRC_BUFFER, kBufferIndex),
                size0(),
                NnShiftOpCodeConfig{n.positionPipeIndex});
            att.addOp(
                OP_SHIFT, "block_shift_v", layerIndex,
                pointerBatchConfig(SRC_BUFFER, vTempBufferIndex),
                pointerRawConfig(SRC_BUFFER, vBufferIndex),
                size0(),
                NnShiftOpCodeConfig{n.positionPipeIndex});
            att.addOp(
                OP_MULTIHEAD_ATT, "block_multihead_att", layerIndex,
                pointerBatchedSliceConfig(SRC_BUFFER, zBufferIndex),
                pointerBatchedSliceConfig(SRC_BUFFER, zBufferIndex),
                size0(),
                NnMultiHeadAttOpConfig{
                    multiHeadAttSlice.nHeads, multiHeadAttSlice.nHeads0,
                    h->nKvHeads, h->headDim, h->seqLen, n.qSlice.d0, kvCacheSlice.kvDim0,
                    n.positionPipeIndex, qBufferIndex, kBufferIndex, vBufferIndex, attBufferIndex});
            att.addOp(
                OP_CAST, "block_cast_y2", layerIndex,
                pointerBatchedSliceConfig(SRC_BUFFER, zBufferIndex),
                pointerBatchConfig(SRC_BUFFER, zqSliceBufferIndex),
                size0(),
                NnCastOpCodeConfig{});
            att.addOp(
                OP_MATMUL, "block_matmul_wo", layerIndex,
                pointerBatchConfig(SRC_BUFFER, zqSliceBufferIndex),
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                size2D(h->weightType, n.woSlice.n0, n.woSlice.d),
                NnMatmulOpConfig{});
            att.addOp(
                OP_CAST, "block_cast_d", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                pointerBatchedSliceConfig(SRC_PIPE, zqPipeIndex),
                size0(),
                NnCastOpCodeConfig{});
            att.addSync(zqPipeIndex, SYNC_NODE_SLICES);

            // ff
            ff.addOp(
                OP_MERGE_ADD, "block_merge_add2", layerIndex,
                pointerBatchConfig(SRC_PIPE, zqPipeIndex),
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                size0(),
                NnMergeAddOpCodeConfig{});
            ff.addOp(
                OP_INV_RMS, "block_norm_pre_1", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                size0(),
                NnInvRmsOpConfig{h->normEpsilon, 1});
            ff.addOp(
                OP_RMS_NORM, "block_norm_1", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                n.rmsNormSize,
                NnRmsNormOpConfig{invRmsBufferIndex, 1});
            
            // FFN implementation: Dense vs MOE
            if (h->nExperts > 0) {
                // MOE model - use buildMoeSegment
                printf("🧠 MOE model with %d experts, %d active\n", h->nExperts, h->nActiveExperts);
                
                // Create router slice (force F32 for router - OP_ROUTER only supports F32)
                NnRowMatmulSlice routerSlice = sliceRowMatmul(F_32, nNodes, h->dim, h->nExperts);
                
                // Create expert weight slices (for active experts)
                NnRowMatmulSlice expertW1Slices[8];  // Max 8 active experts
                NnColMatmulSlice expertW2Slices[8];
                NnRowMatmulSlice expertW3Slices[8];
                
                NnUint expertHiddenSize = 768; // Qwen3 MOE moe_intermediate_size
                for (NnUint k = 0; k < h->nActiveExperts && k < 8; k++) {
                    expertW1Slices[k] = sliceRowMatmul(h->weightType, nNodes, h->dim, expertHiddenSize);      // W1: 2048→768
                    expertW2Slices[k] = sliceColMatmul(h->weightType, nNodes, expertHiddenSize, h->dim);      // W2: 768→2048
                    expertW3Slices[k] = sliceRowMatmul(h->weightType, nNodes, h->dim, expertHiddenSize);      // W3: 2048→768
                }
                
                buildMoeSegment(ff, h, layerIndex, yBufferIndex, yqBufferIndex, routerLogitsBufferIndex,
                               expertIndicesBufferIndex, routingWeightsBufferIndex,
                               dBufferIndices, weightVectorBufferIndices,
                               dqBufferIndices, lBufferIndices,
                               routerSlice,
                               expertW1Slices, expertW2Slices, expertW3Slices);
            } else {
                // Dense FFN for regular models  
                buildFfnSegment(ff, h, layerIndex, yBufferIndex, yqBufferIndex, 
                               dBufferIndex, dqBufferIndex, lBufferIndex,
                               n.w1Slice, n.w2Slice, n.w3Slice, "");
            }
            
            ff.addOp(
                OP_CAST, "block_cast_d3", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                pointerBatchedSliceConfig(SRC_PIPE, zqPipeIndex),
                size0(),
                NnCastOpCodeConfig{});
            ff.addSync(zqPipeIndex, SYNC_NODE_SLICES);

            nodeBuilder.addSegment(att.build());
            nodeBuilder.addSegment(ff.build());
        }

        NnSegmentConfigBuilder end;
        end.addOp(
            OP_MERGE_ADD, "final_merge_add", 0,
            pointerBatchConfig(SRC_PIPE, zqPipeIndex),
            pointerBatchConfig(SRC_BUFFER, xBufferIndex),
            size0(),
            NnMergeAddOpCodeConfig{});
        end.addOp(
            OP_INV_RMS, "final_norm_pre", 0,
            pointerBatchConfig(SRC_BUFFER, xBufferIndex),
            pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
            size0(),
            NnInvRmsOpConfig{h->normEpsilon, 1});
        end.addOp(
            OP_RMS_NORM, "final_norm", 0,
            pointerBatchConfig(SRC_BUFFER, xBufferIndex),
            pointerBatchConfig(SRC_BUFFER, yBufferIndex),
            n.rmsNormSize,
            NnRmsNormOpConfig{invRmsBufferIndex, 1});
        if (yBufferIndex != yqBufferIndex) {
            end.addOp(
                OP_CAST, "final_cast_y", 0,
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                size0(),
                NnCastOpCodeConfig{});
        }
        end.addOp(
            OP_MATMUL, "final_matmul_logits", 0,
            pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
            pointerBatchConfig(SRC_BUFFER, logitsSliceBufferIndex),
            size2D(h->weightType, n.wclsSlice.n, n.wclsSlice.d0),
            NnMatmulOpConfig{});
        end.addOp(
            OP_CAST, "final_cast_logits", 0,
            pointerBatchConfig(SRC_BUFFER, logitsSliceBufferIndex),
            pointerBatchedSliceConfig(SRC_PIPE, n.logitsPipeIndex),
            size0(),
            NnCastOpCodeConfig{});
        end.addSync(n.logitsPipeIndex, SYNC_NODE_SLICES_EXCEPT_ROOT);

        nodeBuilder.addSegment(end.build());
        n.nodeConfigs[nodeIndex] = nodeBuilder.build();
    }
    return n;
}

void releaseLlmNet(LlmNet *net) {
    for (NnUint nodeIndex = 0; nodeIndex < net->netConfig.nNodes; nodeIndex++)
        releaseNodeConfig(&net->nodeConfigs[nodeIndex]);
    releaseNetConfig(&net->netConfig);
    delete[] net->nodeConfigs;
}

void loadLlmNetWeight(const char *path, LlmNet *net, NnRootWeightLoader *loader) {
    MmapFile file;
    openMmapFile(&file, path, net->header->fileSize);
#if DEBUG_USE_MMAP_FOR_WEIGHTS
    assert(net->netConfig.nNodes == 1);
#else
    std::unique_ptr<MmapFile, void(*)(MmapFile *)> fdPtr(&file, closeMmapFile);
    printf("💿 Loading weights...\n");
#endif

    NnByte *data = (NnByte *)file.data;
    NnByte *b = &data[net->header->headerSize];
    NnUint nodeIndex = 0;
    b += loader->loadRoot("embedding", 0, net->tokenEmbeddingSize.nBytes, b);

    for (NnUint layerIndex = 0; layerIndex < net->header->nLayers; layerIndex++) {
        b += loader->loadRowMatmulSlices("block_matmul_q", layerIndex, &net->qSlice, b);
        b += loader->loadRowMatmulSlices("block_matmul_k", layerIndex, &net->kSlice, b);
        b += loader->loadRowMatmulSlices("block_matmul_v", layerIndex, &net->vSlice, b);
        b += loader->loadColMatmulSlices("block_matmul_wo", layerIndex, &net->woSlice, b);
        
        if (net->header->nExperts > 0) {
            // MOE model - load router + expert weights
            // Router weights: dim × nExperts (force F32 - OP_ROUTER only supports F32)
            NnRowMatmulSlice routerSlice = sliceRowMatmul(F_32, net->netConfig.nNodes, 
                                                         net->header->dim, net->header->nExperts);
            b += loader->loadRowMatmulSlices("block_router", layerIndex, &routerSlice, b);
            
            // Expert weights: load ALL experts (not just active ones)
            // The weight file contains all N experts, buildMoeSegment selects top-k
            for (NnUint expertId = 0; expertId < net->header->nExperts; expertId++) {
                NnUint expertHiddenSize = 768; // Qwen3 MOE moe_intermediate_size
                
                // Expert expertId W3 (up_proj): dim → expertHiddenSize
                NnRowMatmulSlice expertW3Slice = sliceRowMatmul(net->header->weightType, net->netConfig.nNodes, 
                                                               net->header->dim, expertHiddenSize);
                char expertW3Name[64];
                snprintf(expertW3Name, sizeof(expertW3Name), "block_matmul_w3.%d", expertId);
                b += loader->loadRowMatmulSlices(expertW3Name, layerIndex, &expertW3Slice, b);
                
                // Expert expertId W1 (gate_proj): dim → expertHiddenSize
                NnRowMatmulSlice expertW1Slice = sliceRowMatmul(net->header->weightType, net->netConfig.nNodes, 
                                                               net->header->dim, expertHiddenSize);
                char expertW1Name[64];
                snprintf(expertW1Name, sizeof(expertW1Name), "block_matmul_w1.%d", expertId);
                b += loader->loadRowMatmulSlices(expertW1Name, layerIndex, &expertW1Slice, b);
                
                // Expert expertId W2 (down_proj): expertHiddenSize → dim
                NnColMatmulSlice expertW2Slice = sliceColMatmul(net->header->weightType, net->netConfig.nNodes, 
                                                               expertHiddenSize, net->header->dim);
                char expertW2Name[64];
                snprintf(expertW2Name, sizeof(expertW2Name), "block_matmul_w2.%d", expertId);
                b += loader->loadColMatmulSlices(expertW2Name, layerIndex, &expertW2Slice, b);
            }
        } else {
            // Dense model - load standard FFN weights
            b += loader->loadRowMatmulSlices("block_matmul_w1", layerIndex, &net->w1Slice, b);
            b += loader->loadColMatmulSlices("block_matmul_w2", layerIndex, &net->w2Slice, b);
            b += loader->loadRowMatmulSlices("block_matmul_w3", layerIndex, &net->w3Slice, b);
        }
        if (net->header->archType == QWEN3) {
            b += loader->loadAll("block_norm_q", layerIndex, net->qkRmsNormSize.nBytes, b);
            b += loader->loadAll("block_norm_k", layerIndex, net->qkRmsNormSize.nBytes, b);
        }
        b += loader->loadAll("block_norm_0", layerIndex, net->rmsNormSize.nBytes, b);
        b += loader->loadAll("block_norm_1", layerIndex, net->rmsNormSize.nBytes, b);
    }

    b += loader->loadAll("final_norm", 0, net->rmsNormSize.nBytes, b);
    b += loader->loadRowMatmulSlices("final_matmul_logits", 0, &net->wclsSlice, b);

    long long missingBytes = (long long)(b - data) - net->header->fileSize;
    if (missingBytes != 0)
        throw std::runtime_error("Missing bytes in weight file: " + std::to_string(missingBytes));
    printf("💿 Weights loaded\n");

    loader->finish();
}
