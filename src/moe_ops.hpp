#ifndef MOE_OPS_HPP
#define MOE_OPS_HPP

// Mixture of Experts (MoE) Operations for distributed-llama
// 
// This header defines MoE-specific operations and data structures
// for implementing full Qwen3MoE support with dynamic expert selection.
//
// CURRENT STATUS: Prepared for implementation (not yet integrated)
// 
// USAGE: Include this header when implementing full MoE support
// to replace the current expert 0 regression approach.

#include "nn/nn-core.hpp"

// ==== MoE Operation Codes ====
// Add these to NnOpCode enum in nn-core.hpp when implementing full MoE

enum MoeOpCode {
    OP_MOE_ROUTER = 100,        // Router network computation (gating)
    OP_MOE_TOPK = 101,          // Top-k expert selection  
    OP_MOE_EXPERT_FFN = 102,    // Expert-specific FFN computation
    OP_MOE_COMBINE = 103        // Weighted combination of expert outputs
};

// ==== MoE Configuration Structures ====

typedef struct {
    NnUint nExperts;           // Total number of experts (e.g., 128)
    NnUint nActiveExperts;     // Active experts per token (e.g., 8)  
    NnUint expertHiddenDim;    // Expert FFN hidden dimension (e.g., 768)
    NnUint routerDim;          // Router output dimension (= nExperts)
} MoeConfig;

typedef struct {
    NnUint expertId;           // Expert index (0 to nExperts-1)
    float gatingWeight;        // Router gating weight for this expert
    NnUint nodeId;             // Node where this expert is located (distributed)
} MoeExpertSelection;

typedef struct {
    MoeConfig config;
    NnUint routerBufferIndex;        // Router output buffer
    NnUint expertSelectionBufferIndex; // Top-k expert selections
    NnUint expertOutputBufferIndex;    // Combined expert outputs
    
    // Expert weight slices for tensor parallelism
    NnRowMatmulSlice* expertUpSlices;    // up_proj weights per expert
    NnRowMatmulSlice* expertGateSlices;  // gate_proj weights per expert  
    NnColMatmulSlice* expertDownSlices;  // down_proj weights per expert
} MoeLayerConfig;

// ==== MoE Operation Configurations ====

typedef struct {
    NnUint inputBufferIndex;     // Input activations
    NnUint routerWeightIndex;    // Router weight matrix
    NnUint outputBufferIndex;    // Router logits output
    MoeConfig moeConfig;
} NnMoeRouterOpConfig;

typedef struct {
    NnUint routerLogitsIndex;    // Router output logits
    NnUint selectionBufferIndex; // Selected expert indices + weights
    NnUint k;                    // Number of experts to select
} NnMoeTopKOpConfig;

typedef struct {
    NnUint inputBufferIndex;     // Input activations
    NnUint expertSelectionIndex; // Selected experts from top-k
    NnUint outputBufferIndex;    // Expert outputs
    MoeLayerConfig layerConfig;
} NnMoeExpertOpConfig;

typedef struct {
    NnUint expertOutputsIndex;   // Individual expert outputs
    NnUint expertWeightsIndex;   // Gating weights from router
    NnUint combinedOutputIndex;  // Final combined output
    NnUint nActiveExperts;       // Number of active experts
} NnMoeCombineOpConfig;

// ==== MoE Tensor Slicing for Distributed Inference ====

// Slice router weights across nodes for tensor parallelism
NnRowMatmulSlice sliceMoeRouter(NnFloatType weightType, NnUint nNodes, 
                               NnUint hiddenSize, NnUint nExperts);

// Distribute experts across nodes for load balancing
// Returns expert assignment: which experts are handled by which node
NnUint* distributeMoeExperts(NnUint nExperts, NnUint nNodes, 
                            NnUint nActiveExperts);

// Slice expert weights for tensor parallelism (per expert)
typedef struct {
    NnRowMatmulSlice upSlice;    // Expert up_proj slice
    NnRowMatmulSlice gateSlice;  // Expert gate_proj slice  
    NnColMatmulSlice downSlice;  // Expert down_proj slice
} MoeExpertSlices;

MoeExpertSlices sliceMoeExpert(NnFloatType weightType, NnUint nNodes,
                              NnUint hiddenSize, NnUint expertHiddenSize);

// ==== MoE Layer Construction ====

// Build complete MoE layer configuration for network
MoeLayerConfig buildMoeLayer(NnUint layerIndex, MoeConfig config,
                           NnUint nNodes, NnUint nBatches);

// Add MoE operations to segment configuration
void addMoeOperations(NnSegmentConfigBuilder& builder, 
                     MoeLayerConfig& layerConfig,
                     NnUint layerIndex);

// ==== Implementation Notes ====
//
// INTEGRATION STEPS:
// 1. Add MoeOpCode values to NnOpCode enum in nn-core.hpp
// 2. Implement MoE operations in nn-cpu-ops.cpp
// 3. Add MoE layer construction to buildLlmNet() in llm.cpp
// 4. Update weight loading in loadLlmNetWeight() for MoE weights
// 5. Test with full MoE weight extraction from convert-hf.py
//
// TENSOR PARALLELISM STRATEGY:
// - Router weights: slice by output dimension (experts)
// - Expert weights: distribute experts across nodes
// - Expert computation: parallel execution on assigned nodes
// - Output combination: gather expert results and combine
//
// PERFORMANCE OPTIMIZATIONS:
// - Expert caching: keep frequently used experts in memory
// - Batch processing: group tokens by selected experts
// - Communication optimization: minimize expert weight transfers

#endif // MOE_OPS_HPP