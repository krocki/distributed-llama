#ifndef LLM_HPP
#define LLM_HPP

#include "nn/nn-core.hpp"
#include "nn/nn-executor.hpp"
#include "nn/nn-network.hpp"

enum LlmHeaderKey {
    VERSION = 0,
    ARCH_TYPE = 1,
    DIM = 2,
    HIDDEN_DIM = 3,
    N_LAYERS = 4,
    N_HEADS = 5,
    N_KV_HEADS = 6,
    N_EXPERTS = 7,
    N_ACTIVE_EXPERTS = 8,
    VOCAB_SIZE = 9,
    SEQ_LEN = 10,
    HIDDEN_ACT = 11,
    ROPE_THETA = 12,
    WEIGHT_FLOAT_TYPE = 13,
    ROPE_SCALING_FACTOR = 14,
    ROPE_SCALING_LOW_FREQ_FACTOR = 15,
    ROPE_SCALING_HIGH_FREQ_FACTORY = 16,
    ROPE_SCALING_ORIG_MAX_SEQ_LEN = 17,
    ROPE_TYPE = 18,
    HEAD_DIM = 19,
    NORM_EPSILON = 20
};

enum LlmHiddenAct {
    HIDDEN_ACT_GELU,
    HIDDEN_ACT_SILU,
};

enum LlmArchType {
    LLAMA = 0xABCD00,      // Standard transformer (Llama, Mistral, etc.)
    QWEN3 = 0xABCD01,      // Qwen3 with Q/K normalization, dense FFN
    QWEN3_MOE = 0xABCD02   // Qwen3 with MoE layers using weight-split tensor parallelism
    
    // QWEN3_MOE Implementation Notes:
    // - Uses weight-split approach: expert weights distributed across nodes, not experts
    // - All 128 experts present on each node with partial weights (tensor parallelism)
    // - Router weights shared across all nodes (not distributed)
    // - Reuses existing MATMUL, SILU operations for expert computation
    // - Compatible with 2^n node configurations (2, 4, 8, etc.)
    // - Memory scales linearly: 2 nodes = 1/2 memory per node, 4 nodes = 1/4, etc.
    // - Perfect load balancing: all nodes process all experts equally
};

typedef struct {
    NnSize headerSize;
    NnSize fileSize;
    int version;
    LlmArchType archType;
    NnUint dim;
    NnUint nLayers;
    NnUint nHeads;
    NnUint headDim;
    NnUint nKvHeads;
    NnUint nExperts;
    NnUint nActiveExperts;
    NnUint origSeqLen; // Original model context length
    NnUint seqLen; // Limited context length by the `--max-seq-len` argument
    NnUint hiddenDim;
    LlmHiddenAct hiddenAct;
    NnUint qDim;
    NnUint kvDim;
    NnUint vocabSize;
    float ropeTheta;
    NnRopeType ropeType;
    float ropeScalingFactor;
    float ropeScalingLowFreqFactor;
    float ropeScalingHighFreqFactory;
    NnUint ropeScalingOrigMaxSeqLen;
    float normEpsilon;

    NnFloatType weightType;
    NnFloatType syncType;
} LlmHeader;

typedef struct {
    LlmHeader *header;
    NnNetConfig netConfig;
    NnNodeConfig *nodeConfigs;
    NnRowMatmulSlice qSlice;
    NnRowMatmulSlice kSlice;
    NnRowMatmulSlice vSlice;
    NnColMatmulSlice woSlice;
    NnRowMatmulSlice w1Slice;
    NnColMatmulSlice w2Slice;
    NnRowMatmulSlice w3Slice;
    NnRowMatmulSlice wclsSlice;
    
    // MoE-specific weight slices (only used for QWEN3_MOE)
    // Weight-split tensor parallelism approach: all experts on each node with partial weights
    NnRowMatmulSlice routerSlice;          // Router weights [dim, nExperts] - shared across all nodes
    NnRowMatmulSlice *expertUpSlices;      // Expert up projections [nExperts*hiddenDim, dim/nNodes] per node
    NnRowMatmulSlice *expertGateSlices;    // Expert gate projections [nExperts*hiddenDim, dim/nNodes] per node
    NnColMatmulSlice *expertDownSlices;    // Expert down projections [dim/nNodes, nExperts*hiddenDim] per node
    NnUint expertsPerNode;                 // All experts on each node (= nExperts for weight-split)
    NnUint *expertToNodeMap;               // Unused in weight-split approach (kept for compatibility)
    
    NnUint positionPipeIndex;
    NnUint tokenPipeIndex;
    NnUint xPipeIndex;
    NnUint logitsPipeIndex;
    NnSize2D tokenEmbeddingSize;
    NnSize2D rmsNormSize;
    NnSize2D qkRmsNormSize;
} LlmNet;

LlmHeader loadLlmHeader(const char* path, const unsigned int maxSeqLen, NnFloatType syncType);
void printLlmHeader(LlmHeader *header);
LlmNet buildLlmNet(LlmHeader *h, NnUint nNodes, NnUint nBatches);
void releaseLlmNet(LlmNet *net);
void loadLlmNetWeight(const char* path, LlmNet *net, NnRootWeightLoader *loader);

#endif