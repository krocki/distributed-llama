# Distributed Llama Project Structure

## Overview
Distributed Llama is a tensor-parallel inference system for large language models. It distributes model computations across multiple nodes using slice-based parallelism.

## Architecture Components

### Core Files
- **src/llm.cpp/hpp**: Main LLM model definition and network building logic
- **src/app.cpp/hpp**: Application entry point and command line interface  
- **src/dllama.cpp**: Main distributed llama executable
- **src/dllama-api.cpp**: REST API server implementation
- **src/tokenizer.cpp/hpp**: Tokenizer implementation

### Neural Network Core (`src/nn/`)
- **nn-core.cpp/hpp**: Core data structures, operation definitions, and utilities
- **nn-config-builder.hpp**: Builder classes for constructing network configurations
- **nn-cpu.cpp/hpp**: CPU execution backend
- **nn-cpu-ops.cpp/hpp**: CPU implementations of neural network operations
- **nn-vulkan.cpp/hpp**: Vulkan GPU execution backend
- **nn-executor.cpp/hpp**: Main execution engine that orchestrates operations
- **nn-network.cpp/hpp**: Network management and weight loading
- **nn-quants.cpp/hpp**: Quantization utilities

## Key Data Structures

### LlmHeader
Contains model metadata:
- Architecture type (LLAMA, QWEN3)
- Dimensions (dim, hiddenDim, headDim)
- Layer counts (nLayers, nHeads, nKvHeads) 
- **MOE fields**: nExperts, nActiveExperts (currently parsed but unused)
- Sequence length, vocab size, activation function
- RoPE configuration

### LlmNet  
Network configuration with tensor slices:
- Matrix multiplication slices: qSlice, kSlice, vSlice, woSlice, w1Slice, w2Slice, w3Slice
- Pipe indices for data flow
- Node configurations for distributed execution

### Operation System
Operations are defined by `NnOpCode` enum:
- `OP_MATMUL`: Matrix multiplication (main compute operation)
- `OP_SILU`, `OP_GELU`: Activation functions
- `OP_RMS_NORM`: Layer normalization  
- `OP_ROPE`: Rotary position embedding
- `OP_MULTIHEAD_ATT`: Multi-head attention
- `OP_MERGE_ADD`: Residual connections
- `OP_CAST`: Type conversions for quantization

## Current FFN Implementation

The FFN implementation has been refactored for MOE readiness. In `buildLlmNet()` (llm.cpp:441-443), each layer's feed-forward network now uses:

```cpp
// Dense FFN for regular models
buildFfnSegment(ff, h, layerIndex, yBufferIndex, yqBufferIndex, 
               dBufferIndex, dqBufferIndex, lBufferIndex,
               n.w1Slice, n.w2Slice, n.w3Slice);
```

The `buildFfnSegment()` helper function (llm.cpp:34-84) implements:
- `w1` gate projection: `yq -> d` 
- `w3` up projection: `yq -> l`
- SiLU activation: `silu(d)`
- Element-wise multiply: `d * l -> d`
- `w2` down projection: `d -> output`

This implements: `w2(silu(w1(x)) * w3(x))` - the standard SwiGLU FFN used in Llama/Qwen3.

## Tensor Slicing for Distribution

Models are distributed via tensor parallelism:
- **Row slicing**: Input tensors split across nodes (e.g., w1, w3 gates)
- **Column slicing**: Output tensors split across nodes (e.g., w2, wo projections)
- **Synchronization**: Results merged via `SYNC_NODE_SLICES` operations

Matrix slices defined by:
- `NnRowMatmulSlice`: For splitting input dimensions
- `NnColMatmulSlice`: For splitting output dimensions

## Weight Loading System

Weights loaded in `loadLlmNetWeight()`:
- Uses memory mapping for efficient access
- Loads weights per layer in order: q,k,v,wo,w1,w2,w3,norms
- Distributes weight slices to appropriate nodes
- Handles both F32 and quantized (Q40/Q80) formats

## Execution Flow

1. **Network Building**: `buildLlmNet()` constructs operation graph per node
2. **Weight Loading**: `loadLlmNetWeight()` distributes model parameters  
3. **Execution**: `NnExecutor` runs operation sequences with synchronization
4. **Output**: Final logits merged from all nodes

## Key Functions

### llm.cpp
- `loadLlmHeader()`: Parse model metadata from file header
- `buildLlmNet()`: Construct distributed computation graph
- `loadLlmNetWeight()`: Load and distribute model weights
- `printLlmHeader()`: Display model information

### nn-core.cpp  
- `sliceRowMatmul()`: Calculate row-wise tensor slices
- `sliceColMatmul()`: Calculate column-wise tensor slices
- `opCodeToString()`: Operation code debugging utilities

### nn-executor.cpp
- `NnExecutor::forward()`: Execute one forward pass
- Node synchronization and data movement

## MOE Extension Requirements

To support Qwen3-MOE (30B with 128 experts, top-8 routing):

1. **New Operations Needed**:
   - `OP_ROUTER`: Expert selection/routing 
   - `OP_MOE_MERGE`: Combine expert outputs with routing weights
   - Expert weight management

2. **Header Extensions**: 
   - Already has `nExperts`, `nActiveExperts` fields

3. **Weight Layout Changes**:
   - Current: Single w1/w2/w3 per layer  
   - MOE: Multiple expert weights per layer + router weights

4. **Refactoring Approach**:
   - ✅ `buildFfnSegment()` helper extracted for easy replacement
   - ✅ `OP_ROUTER` and `OP_MOE_MERGE` operations implemented
   - ✅ `buildMoeSegment()` framework created
   - ✅ Dense models still work with MOE infrastructure in place
   - Ready: Route between dense vs MOE based on `nExperts` field

5. **MOE Operations Implemented**:
   - `OP_ROUTER`: Top-k expert selection with softmax routing
   - `OP_MOE_MERGE`: Weighted combination of expert outputs
   - Both operations support tensor parallelism distribution

6. **Next Steps for Full MOE**:
   - Add expert weight management to LlmNet structure
   - Implement weight loading for multiple experts per layer
   - Add dynamic expert selection in buildMoeSegment
   - Test with actual Qwen3-MOE model weights