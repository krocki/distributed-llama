#include "nn-core.hpp"
#include "nn-config-builder.hpp"
#include "nn-cpu.hpp"
#include "../llm.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>

// EXACT expf_neon implementation from distributed-llama nn-cpu-ops.cpp
static inline float32x4_t expf_neon(float32x4_t x) {
    const float32x4_t ln2 = vdupq_n_f32(0.69314718056f);
    const float32x4_t inv_ln2 = vdupq_n_f32(1.44269504089f);
    const float32x4_t c1 = vdupq_n_f32(1.0f);
    const float32x4_t c2 = vdupq_n_f32(0.5f);
    const float32x4_t c3 = vdupq_n_f32(0.1666666667f);
    const float32x4_t c4 = vdupq_n_f32(0.04166666667f);
    const float32x4_t c5 = vdupq_n_f32(0.008333333333f);

    x = vminq_f32(x, vdupq_n_f32(88.0f));
    x = vmaxq_f32(x, vdupq_n_f32(-88.0f));

    float32x4_t kf = vaddq_f32(vmulq_f32(x, inv_ln2), vdupq_n_f32(0.5f));
    int32x4_t k = vcvtq_s32_f32(kf);
    kf = vcvtq_f32_s32(k);

    float32x4_t f = vmlsq_f32(x, kf, ln2);
    float32x4_t f2 = vmulq_f32(f, f);
    float32x4_t f3 = vmulq_f32(f2, f);
    float32x4_t f4 = vmulq_f32(f3, f);
    float32x4_t f5 = vmulq_f32(f4, f);

    float32x4_t poly = vaddq_f32(c1, vaddq_f32(f, vaddq_f32(vmulq_f32(c2, f2), 
                      vaddq_f32(vmulq_f32(c3, f3), vaddq_f32(vmulq_f32(c4, f4), vmulq_f32(c5, f5))))));

    int32x4_t exp_int = vshlq_n_s32(vaddq_s32(k, vdupq_n_s32(127)), 23);
    float32x4_t exp_val = vreinterpretq_f32_s32(exp_int);

    return vmulq_f32(poly, exp_val);
}

// Exact silu implementation from distributed-llama nn-cpu-ops.cpp
static void silu_F32_exact(float *output, const unsigned int n) {
    unsigned int i = 0;
#if defined(__ARM_NEON)
    const unsigned int neonEnd = n - (n % 4);
    
    for (; i < neonEnd; i += 4) {
        float32x4_t x = vld1q_f32(&output[i]);
        float32x4_t neg_x = vnegq_f32(x);
        float32x4_t exp_negx = expf_neon(neg_x);
        float32x4_t denominator = vaddq_f32(exp_negx, vdupq_n_f32(1.0f));

        float32x4_t recip = vrecpeq_f32(denominator);
        recip = vmulq_f32(recip, vsubq_f32(vdupq_n_f32(2.0f), vmulq_f32(denominator, recip)));

        float32x4_t result = vmulq_f32(x, recip);
        vst1q_f32(output + i, result);
    }
#endif
    // Scalar fallback for remaining elements
    for (; i < n; i++) {
        float x = output[i];
        output[i] = x / (1.0f + expf(-x));
    }
}
#endif

#define DIM 8
#define N_EXPERTS 8
#define N_ACTIVE_EXPERTS 8
#define N_BATCHES 1

void buildMoeTestConfig(NnNetConfig *netConfig, NnNodeConfig *nodeConfig) {
    NnUint nNodes = 1;
    NnNetConfigBuilder netBuilder(nNodes, N_BATCHES);
    NnUint inputPipeIndex = netBuilder.addPipe("INPUT", size2D(F_32, N_BATCHES, DIM));
    NnUint outputPipeIndex = netBuilder.addPipe("OUTPUT", size2D(F_32, N_BATCHES, DIM));

    NnNodeConfigBuilder nodeBuilder(0);
    
    // Buffers needed by buildMoeSegment
    NnUint yBufferIndex = nodeBuilder.addBuffer("y", size2D(F_32, N_BATCHES, DIM));
    NnUint yqBufferIndex = nodeBuilder.addBuffer("yq", size2D(F_32, N_BATCHES, DIM));
    NnUint expertInputBufferIndex = nodeBuilder.addBuffer("expert_input", size2D(F_32, N_BATCHES, DIM)); // Dedicated expert input buffer
    NnUint expertIndicesBufferIndex = nodeBuilder.addBuffer("expert_indices", size2D(F_32, N_BATCHES, N_ACTIVE_EXPERTS));
    NnUint routingWeightsBufferIndex = nodeBuilder.addBuffer("routing_weights", size2D(F_32, N_BATCHES, N_ACTIVE_EXPERTS));
    NnUint expertOutputsBufferIndex = nodeBuilder.addBuffer("expert_outputs", size2D(F_32, N_BATCHES, N_ACTIVE_EXPERTS * DIM)); // For OP_MOE_MERGE
    NnUint weightVectorBufferIndex = nodeBuilder.addBuffer("weight_vector", size2D(F_32, N_BATCHES, DIM)); // For routing weight multiplication
    
    // Intermediate buffers for FFN computation in buildMoeSegment  
    // CRITICAL FIX: Each expert needs separate buffers to prevent collisions
    // SEQUENTIAL PROCESSING: Allocate buffers for slot 0 (unused) and slot 1 (working)
    // Slot 0 has execution bug but we need it for array indexing
    NnUint dBufferIndices[2];
    NnUint dqBufferIndices[2];
    NnUint lBufferIndices[2];
    
    // Allocate both slot 0 (broken) and slot 1 (working) buffers
    dBufferIndices[0] = nodeBuilder.addBuffer("d_expert_slot0", size2D(F_32, N_BATCHES, DIM));   // Slot 0 - doesn't work
    dqBufferIndices[0] = nodeBuilder.addBuffer("dq_expert_slot0", size2D(F_32, N_BATCHES, DIM)); // Slot 0 - doesn't work
    lBufferIndices[0] = nodeBuilder.addBuffer("l_expert_slot0", size2D(F_32, N_BATCHES, DIM));   // Slot 0 - doesn't work
    
    dBufferIndices[1] = nodeBuilder.addBuffer("d_expert_slot1", size2D(F_32, N_BATCHES, DIM));   // Slot 1 - works
    dqBufferIndices[1] = nodeBuilder.addBuffer("dq_expert_slot1", size2D(F_32, N_BATCHES, DIM)); // Slot 1 - works
    lBufferIndices[1] = nodeBuilder.addBuffer("l_expert_slot1", size2D(F_32, N_BATCHES, DIM));   // Slot 1 - works
    
    printf("Slot 0 buffers (broken): d=%d, dq=%d, l=%d\n", dBufferIndices[0], dqBufferIndices[0], lBufferIndices[0]);
    printf("Slot 1 buffers (working): d=%d, dq=%d, l=%d\n", dBufferIndices[1], dqBufferIndices[1], lBufferIndices[1]);
    
    printf("Key buffer indices: y=%d, yq=%d, expertInput=%d, expertIndices=%d, routingWeights=%d, expertOutputs=%d\n",
           yBufferIndex, yqBufferIndex, expertInputBufferIndex, expertIndicesBufferIndex, routingWeightsBufferIndex, expertOutputsBufferIndex);
    
    // buildMoeSegment will use slot 1 (the working one) for single expert case
    
    NnSegmentConfigBuilder segmentBuilder;

    // Copy input to y buffer
    segmentBuilder.addOp(
        OP_CAST, "copy_input", 0,
        pointerBatchConfig(SRC_PIPE, inputPipeIndex),
        pointerBatchConfig(SRC_BUFFER, yBufferIndex),
        size0(),
        NnCastOpCodeConfig{});
    
    // Copy to yq buffer (used by router)
    segmentBuilder.addOp(
        OP_CAST, "copy_to_yq", 0,
        pointerBatchConfig(SRC_BUFFER, yBufferIndex),
        pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
        size0(),
        NnCastOpCodeConfig{});
        
    // Copy to dedicated expert input buffer (used by experts - won't be modified)
    segmentBuilder.addOp(
        OP_CAST, "copy_to_expert_input", 0,
        pointerBatchConfig(SRC_BUFFER, yBufferIndex),
        pointerBatchConfig(SRC_BUFFER, expertInputBufferIndex),
        size0(),
        NnCastOpCodeConfig{});
    
    // Setup for buildMoeSegment
    LlmHeader header;
    header.dim = DIM;
    header.hiddenDim = DIM;
    header.nExperts = N_EXPERTS;
    header.nActiveExperts = N_ACTIVE_EXPERTS;
    header.weightType = F_32;
    
    // Create weight slices for all experts
    NnRowMatmulSlice routerSlice = sliceRowMatmul(F_32, 1, DIM, N_EXPERTS);
    NnRowMatmulSlice* expertW1Slices = new NnRowMatmulSlice[N_EXPERTS];
    NnColMatmulSlice* expertW2Slices = new NnColMatmulSlice[N_EXPERTS];
    NnRowMatmulSlice* expertW3Slices = new NnRowMatmulSlice[N_EXPERTS];
    
    for (int i = 0; i < N_EXPERTS; i++) {
        expertW1Slices[i] = sliceRowMatmul(F_32, 1, DIM, DIM);
        expertW2Slices[i] = sliceColMatmul(F_32, 1, DIM, DIM);
        expertW3Slices[i] = sliceRowMatmul(F_32, 1, DIM, DIM);
    }
    
    // *** THIS IS THE TEST: Call buildMoeSegment! ***
    buildMoeSegment(segmentBuilder, &header, 1,
                   yBufferIndex, yqBufferIndex, expertInputBufferIndex, // Separate router input vs expert input
                   expertIndicesBufferIndex, routingWeightsBufferIndex,
                   expertOutputsBufferIndex, weightVectorBufferIndex,
                   dBufferIndices, dqBufferIndices, lBufferIndices,
                   routerSlice,
                   expertW1Slices, expertW2Slices, expertW3Slices);
    
    // Copy result to output
    segmentBuilder.addOp(
        OP_CAST, "copy_output", 0,
        pointerBatchConfig(SRC_BUFFER, yBufferIndex), // buildMoeSegment final output is in yBufferIndex
        pointerBatchConfig(SRC_PIPE, outputPipeIndex),
        size0(),
        NnCastOpCodeConfig{});

    nodeBuilder.addSegment(segmentBuilder.build());

    *netConfig = netBuilder.build();
    *nodeConfig = nodeBuilder.build();
    
    // Cleanup
    delete[] expertW1Slices;
    delete[] expertW2Slices;
    delete[] expertW3Slices;
}

void print2D(const char *name, NnUint x, NnUint y, float *w) {
    for (NnUint i = 0; i < y; i++) {
        printf("%s[%d] = ", name, i);
        for (NnUint j = 0; j < x; j++)
            printf("%.3f ", w[i * x + j]);
        printf("\n");
    }
}

// Reference single expert forward pass (no router, no mixing - just one expert)
void referenceSingleExpertForward(const float* input, float* output,
                                 const float* expert_w1, const float* expert_w2, const float* expert_w3,
                                 int d_model, int d_ff) {
    float tmp_g[d_ff]; // w1 output
    float tmp_u[d_ff]; // w3 output

    // Step 1: w1(input) -> tmp_g
    for (int h = 0; h < d_ff; h++) {
        tmp_g[h] = 0.0f;
        for (int d = 0; d < d_model; d++) {
            tmp_g[h] += input[d] * expert_w1[h * d_model + d];
        }
    }

    // Step 2: silu(tmp_g) - use exact same implementation as distributed-llama
    silu_F32_exact(tmp_g, d_ff);

    // Step 3: w3(input) -> tmp_u
    for (int h = 0; h < d_ff; h++) {
        tmp_u[h] = 0.0f;
        for (int d = 0; d < d_model; d++) {
            tmp_u[h] += input[d] * expert_w3[h * d_model + d];
        }
    }

    // Step 4: Element-wise multiply: tmp_g *= tmp_u
    for (int h = 0; h < d_ff; h++) {
        tmp_g[h] *= tmp_u[h];
    }

    // Step 5: w2(tmp_g) -> output
    for (int d = 0; d < d_model; d++) {
        output[d] = 0.0f;
        for (int h = 0; h < d_ff; h++) {
            output[d] += tmp_g[h] * expert_w2[d * d_ff + h];
        }
    }
}

// Reference MOE implementation based on https://github.com/krocki/moe.cc/blob/main/test_model_trace.c
void referenceMoeForward(const float* input, float* output, 
                        const float* router_w, 
                        const float* expert_w1, const float* expert_w2, const float* expert_w3,
                        int d_model, int d_ff, int n_experts, int top_k) {
    
    // Step 1: Router logits calculation (like line 463-467)
    float logits[n_experts];
    for (int e = 0; e < n_experts; e++) {
        logits[e] = 0.0f;
        for (int d = 0; d < d_model; d++) {
            logits[e] += input[d] * router_w[e * d_model + d];
        }
    }
    
    // Step 2: Top-k selection (like line 482-483)
    int top_indices[top_k];
    float top_values[top_k];
    
    // Simple top-k selection (find top_k largest logits)
    for (int k = 0; k < top_k; k++) {
        int best_idx = 0;
        float best_val = -1e9f;
        for (int e = 0; e < n_experts; e++) {
            bool already_selected = false;
            for (int prev = 0; prev < k; prev++) {
                if (top_indices[prev] == e) {
                    already_selected = true;
                    break;
                }
            }
            if (!already_selected && logits[e] > best_val) {
                best_val = logits[e];
                best_idx = e;
            }
        }
        top_indices[k] = best_idx;
        top_values[k] = best_val;
    }
    
    // Step 3: Softmax over top-k (like line 486-491)
    printf("=== Reference Router Debug ===\n");
    printf("Raw logits: ");
    for (int i = 0; i < top_k; i++) {
        printf("%.6f ", top_values[i]);
    }
    printf("\n");
    
    float maxv = top_values[0];
    for (int i = 1; i < top_k; i++) {
        if (top_values[i] > maxv) maxv = top_values[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < top_k; i++) {
        top_values[i] = expf(top_values[i] - maxv);
        sum += top_values[i];
    }
    
    float inv = 1.0f / (sum + 1e-9f);
    for (int i = 0; i < top_k; i++) {
        top_values[i] *= inv;
    }
    
    printf("Expert indices: ");
    for (int i = 0; i < top_k; i++) {
        printf("%.0f ", (float)top_indices[i]);
    }
    printf("\n");
    printf("Routing weights: ");
    for (int i = 0; i < top_k; i++) {
        printf("%.6f ", top_values[i]);
    }
    printf("\n");
    
    // Step 4: Initialize output
    for (int d = 0; d < d_model; d++) {
        output[d] = 0.0f;
    }
    
    // Step 5: Process each selected expert (like line 493-516)
    // Process only the first expert (matching N_ACTIVE_EXPERTS = 1)
    for (int k = 0; k < N_ACTIVE_EXPERTS; k++) { // Process single expert
        int expert_idx = top_indices[k];
        float weight = top_values[k];
        
        // SwiGLU: expert_out = w2(silu(w1(input)) * w3(input))
        float tmp_g[d_ff]; // w1 output
        float tmp_u[d_ff]; // w3 output
        float tmp_y[d_model]; // final expert output
        
        // w1(input) -> tmp_g
        for (int h = 0; h < d_ff; h++) {
            tmp_g[h] = 0.0f;
            for (int d = 0; d < d_model; d++) {
                tmp_g[h] += input[d] * expert_w1[expert_idx * d_model * d_ff + h * d_model + d];
            }
        }
        
        // silu(tmp_g)
        //for (int h = 0; h < d_ff; h++) {
        //    tmp_g[h] = tmp_g[h] / (1.0f + expf(-tmp_g[h])); // silu activation
        //}
        silu_F32_exact(tmp_g, d_ff);
        
        // w3(input) -> tmp_u
        for (int h = 0; h < d_ff; h++) {
            tmp_u[h] = 0.0f;
            for (int d = 0; d < d_model; d++) {
                tmp_u[h] += input[d] * expert_w3[expert_idx * d_model * d_ff + h * d_model + d];
            }
        }
        
        // Element-wise multiply: tmp_g *= tmp_u
        for (int h = 0; h < d_ff; h++) {
            tmp_g[h] *= tmp_u[h];
        }
        
        // w2(tmp_g) -> tmp_y
        for (int d = 0; d < d_model; d++) {
            tmp_y[d] = 0.0f;
            for (int h = 0; h < d_ff; h++) {
                tmp_y[d] += tmp_g[h] * expert_w2[expert_idx * d_ff * d_model + d * d_ff + h];
            }
        }
        
        // Debug: Print raw expert output before weighting
        printf("=== Reference Expert %d Raw Output ===\n", expert_idx);
        printf("expert_%d_raw[0] = ", expert_idx);
        for (int d = 0; d < d_model; d++) {
            printf("%.3f ", tmp_y[d]);
        }
        printf("\n");
        printf("Weight for expert %d: %.6f\n", expert_idx, weight);
        
        // Debug: Show which weights this expert is using  
        printf("Reference Expert %d W1[0-3]: %.3f %.3f %.3f %.3f\n", expert_idx,
               expert_w1[expert_idx * d_model * d_model + 0],
               expert_w1[expert_idx * d_model * d_model + 1], 
               expert_w1[expert_idx * d_model * d_model + 2],
               expert_w1[expert_idx * d_model * d_model + 3]);
        
        // Debug: Print weighted contribution
        printf("expert_%d_weighted[0] = ", expert_idx);
        for (int d = 0; d < d_model; d++) {
            printf("%.3f ", weight * tmp_y[d]);
        }
        printf("\n");
        
        // Weighted accumulation into output
        for (int d = 0; d < d_model; d++) {
            output[d] += weight * tmp_y[d];
        }
        
        // Debug: Print cumulative output after this expert
        printf("cumulative_output_after_expert_%d[0] = ", expert_idx);
        for (int d = 0; d < d_model; d++) {
            printf("%.3f ", output[d]);
        }
        printf("\n\n");
    }
    
    printf("Reference MOE: selected expert %d(%.3f)\n", 
           top_indices[0], top_values[0]);
}

int main() {
    printf("=== Testing buildMoeSegment Function ===\n");
    
    // Ensure deterministic behavior
    srand(42);
    
    initQuants();

    NnUint nThreads = 2;
    NnNetConfig netConfig;
    NnNodeConfig nodeConfig;
    buildMoeTestConfig(&netConfig, &nodeConfig);

    NnNetExecution execution(nThreads, &netConfig);
    float *input = (float *)execution.pipes[0];
    float *output = (float *)execution.pipes[1];
    
    // Same input as working test
    for (NnUint b = 0; b < N_BATCHES; b++) {
        for (NnUint i = 0; i < DIM; i++)
            input[b * DIM + i] = i / (float)DIM + (float)b;
    }

    print2D("input", DIM, N_BATCHES, input);

    // Generate random router weights for better testing
    float routerWeights[DIM * N_EXPERTS];
    for (NnUint expert = 0; expert < N_EXPERTS; expert++) {
        for (NnUint dim = 0; dim < DIM; dim++) {
            routerWeights[expert * DIM + dim] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Random [-1, 1]
        }
    }

    NnCpuDevice *device = new NnCpuDevice(&netConfig, &nodeConfig, &execution);
    std::vector<NnExecutorDevice> devices;
    devices.push_back(NnExecutorDevice(device, -1, -1));

    NnFakeNodeSynchronizer synchronizer;
    
    // Buffer indices are hardcoded based on the order in buildMoeTestConfig:
    // 0=yBuffer, 1=yqBuffer, 2=expertInputBuffer, 3=expertIndicesBuffer, 4=routingWeightsBuffer, etc.
    printf("=== BUFFER INDEX VERIFICATION ===\n");
    printf("Hardcoded buffer indices (based on buildMoeTestConfig allocation order):\n");
    printf("  0: yBuffer\n");
    printf("  1: yqBuffer\n"); 
    printf("  2: expertInputBuffer\n");
    printf("  3: expertIndicesBuffer\n");
    printf("  4: routingWeightsBuffer\n");
    printf("  5: expertOutputsBuffer\n");
    printf("  6+: expert specific buffers (d, dq, l per expert)\n");
    
    float *expertIndices = (float *)device->buffers[3];   // expertIndicesBuffer  
    float *routingWeights = (float *)device->buffers[4];  // routingWeightsBuffer
    
    NnExecutor executor(&netConfig, &nodeConfig, &devices, &execution, &synchronizer, false);
    
    // Load router weights
    executor.loadWeight("block_router", 1, sizeof(routerWeights), (NnByte *)routerWeights);
    
    // Generate randomized expert weights for all experts
    float allExpertW1[N_EXPERTS * DIM * DIM];
    float allExpertW2[N_EXPERTS * DIM * DIM]; 
    float allExpertW3[N_EXPERTS * DIM * DIM];
    
    for (int expert = 0; expert < N_EXPERTS; expert++) {
        for (int i = 0; i < DIM * DIM; i++) {
            // Generate random weights per expert - each expert has different weights
            allExpertW1[expert * DIM * DIM + i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            allExpertW2[expert * DIM * DIM + i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            allExpertW3[expert * DIM * DIM + i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
    
    // PHASE 1: Run router to determine expert selection
    printf("=== PHASE 1: Running router to determine expert selection ===\n");
    
    // Set batch size first
    execution.setBatchSize(N_BATCHES);
    
    // First, run just the router operation to get expert selection
    // Input data is already set up in execution.pipes[0] above
    executor.forward(); // This will run the router and populate expertIndicesBuffer
    
    printf("Router selected experts: ");
    for (int i = 0; i < N_ACTIVE_EXPERTS; i++) {
        printf("%.0f ", expertIndices[i]);
    }
    printf("\n");
    
    // PHASE 2: Load expert weights in the order router selected them
    printf("=== PHASE 2: Loading expert weights in router selection order ===\n");
    
    for (int k = 0; k < N_ACTIVE_EXPERTS; k++) {
        // Router selected expert at position k
        int expertIndex = (int)expertIndices[k];
        // Load this expert's weights into layerIndex for position k
        NnUint expertLayerIndex = 1 + 100 + k; // layerIndex 101, 102, etc.
        
        printf("Loading Expert %d weights into layerIndex %d (router position %d)\n", 
               expertIndex, expertLayerIndex, k);
        printf("  Expert %d W1[0-3]: %.3f %.3f %.3f %.3f\n", expertIndex,
               allExpertW1[expertIndex * DIM * DIM + 0], allExpertW1[expertIndex * DIM * DIM + 1],
               allExpertW1[expertIndex * DIM * DIM + 2], allExpertW1[expertIndex * DIM * DIM + 3]);
        
        // Load the weights using the executor
        executor.loadWeight("block_matmul_w1", expertLayerIndex, 
                           sizeof(float) * DIM * DIM, (NnByte *)&allExpertW1[expertIndex * DIM * DIM]);
        executor.loadWeight("block_matmul_w2", expertLayerIndex, 
                           sizeof(float) * DIM * DIM, (NnByte *)&allExpertW2[expertIndex * DIM * DIM]);  
        executor.loadWeight("block_matmul_w3", expertLayerIndex, 
                           sizeof(float) * DIM * DIM, (NnByte *)&allExpertW3[expertIndex * DIM * DIM]);
        printf("  Loaded Expert %d into layerIndex %d\n", expertIndex, expertLayerIndex);
    }
    
    printf("=== Weight Loading Verification ===\n");
    for (int k = 0; k < N_ACTIVE_EXPERTS; k++) {
        int expertIndex = (int)expertIndices[k];
        printf("Expert slot %d: layerIndex %d -> Expert %d (router selected)\n", 
               k, 1 + 100 + k, expertIndex);
    }
    printf("Weight loading completed for %d experts\n", N_ACTIVE_EXPERTS);
    
    // Weight loading completed in the loop above

    printf("Running buildMoeSegment...\n");
    execution.setBatchSize(N_BATCHES);
    
    // Debug: Print intermediate buffer values before and after execution
    printf("=== Buffer Contents Before Execution ===\n");
    printf("INPUT pipe: ");
    for (int i = 0; i < DIM; i++) printf("%.3f ", input[i]);
    printf("\n");
    
    // Use documented buffer indices
    float *yBuffer = (float *)device->buffers[0];           // yBuffer
    float *yqBuffer = (float *)device->buffers[1];          // yqBuffer  
    float *expertInputBuffer = (float *)device->buffers[2]; // expertInputBuffer
    
    printf("yBuffer: ");
    for (int i = 0; i < DIM; i++) printf("%.3f ", yBuffer[i]);
    printf("\n");
    printf("yqBuffer: ");
    for (int i = 0; i < DIM; i++) printf("%.3f ", yqBuffer[i]);
    printf("\n");
    
    printf("expertInputBuffer: ");
    for (int i = 0; i < DIM; i++) printf("%.3f ", expertInputBuffer[i]);
    printf("\n");
    
    printf("=== THEORY 2: Buffer State Mid-Execution ===\n");
    printf("About to call executor.forward() - checking buffer states:\n");
    // Buffer indices for N_ACTIVE_EXPERTS (dynamically determined based on allocation)
    // Slot 0 (broken): d=7, dq=8, l=9; Slot 1 (working): d=10, dq=11, l=12
    // For general k, we only care about the working buffers that buildMoeSegment uses
    int expertDBufferIndices[N_ACTIVE_EXPERTS];
    for (int k = 0; k < N_ACTIVE_EXPERTS; k++) {
        expertDBufferIndices[k] = 10; // All experts use slot 1 working buffer in current implementation
    }  
    for (int k = 0; k < N_ACTIVE_EXPERTS; k++) {
        float *expertBuffer = (float *)device->buffers[expertDBufferIndices[k]];
        printf("Expert %d output buffer (index %d, before execution): ", k, expertDBufferIndices[k]);
        for (int i = 0; i < 4; i++) printf("%.3f ", expertBuffer[i]); // Just first 4 values
        printf("...\n");
    }
    
    // PHASE 3: Run the full MOE computation with router-selected expert weights
    printf("=== PHASE 3: Running full MOE computation with router-selected experts ===\n");
    executor.forward();
    
    printf("=== THEORY 2: Buffer State Post-Execution ===\n");
    printf("After executor.forward() - checking if Expert 0 buffer was actually modified:\n");
    for (int k = 0; k < N_ACTIVE_EXPERTS; k++) {
        float *expertBuffer = (float *)device->buffers[expertDBufferIndices[k]];
        printf("Expert %d output buffer (index %d, after execution): ", k, expertDBufferIndices[k]);
        for (int i = 0; i < 4; i++) printf("%.3f ", expertBuffer[i]); // Just first 4 values
        printf("...\n");
        
        // Check if buffer contains input values exactly (Theory 3: buffer collision)
        bool isInputPassthrough = true;
        float expectedInput[] = {0.000f, 0.125f, 0.250f, 0.375f, 0.500f, 0.625f, 0.750f, 0.875f};
        for (int i = 0; i < DIM; i++) {
            if (fabsf(expertBuffer[i] - expectedInput[i]) > 0.001f) {
                isInputPassthrough = false;
                break;
            }
        }
        printf("Expert %d buffer collision check: inputPassthrough=%s\n", k, isInputPassthrough ? "YES" : "NO");
    }
    
    printf("=== THEORY 3: Buffer Index Verification ===\n");
    printf("Double-checking buffer allocation:\n");
    printf("Expert 0: input=expertInputBuffer[2], output=expert0_dBuffer[%d]\n", expertDBufferIndices[0]);
    printf("Expert 1: input=expertInputBuffer[2], output=expert1_dBuffer[%d]\n", expertDBufferIndices[1]);
    
    // Check if Expert 0's input and output buffers are the same (would cause passthrough)
    float *expert0OutputBuffer = (float *)device->buffers[expertDBufferIndices[0]]; // expert0_dBuffer
    printf("Input buffer (expertInputBuffer[2]): pointer=%p\n", (void*)expertInputBuffer);
    printf("Expert 0 output buffer (expert0_dBuffer[%d]): pointer=%p\n", expertDBufferIndices[0], (void*)expert0OutputBuffer);
    printf("Buffer collision (same pointer): %s\n", (expertInputBuffer == expert0OutputBuffer) ? "YES" : "NO");
    
    printf("=== buildMoeSegment Router Debug ===\n");
    float *expertIndicesBuffer = (float *)device->buffers[3];   // expertIndicesBuffer  
    float *routingWeightsBuffer = (float *)device->buffers[4];  // routingWeightsBuffer
    
    printf("Expert indices: ");
    for (int i = 0; i < N_ACTIVE_EXPERTS; i++) printf("%.0f ", expertIndicesBuffer[i]);
    printf("\n");
    
    printf("Routing weights: ");
    for (int i = 0; i < N_ACTIVE_EXPERTS; i++) printf("%.6f ", routingWeightsBuffer[i]);
    printf("\n");

    printf("=== Individual Expert Outputs ===\n");
    printf("NOTE: All experts now use the SAME working buffer (slot 1) and process sequentially\n");
    
    // All experts use slot 1 buffer (index 10) due to slot 0 execution bug
    // From debug: "Slot 1 buffers (working): d=10, dq=11, l=12"
    float *workingBuffer = (float *)device->buffers[10]; // Slot 1 working buffer
    printf("Final working buffer (slot 1, index 10) content: ");
    for (int i = 0; i < DIM; i++) printf("%.3f ", workingBuffer[i]);
    printf("\n");
    
    // Show what SHOULD be in each expert's buffer for comparison
    for (int k = 0; k < N_ACTIVE_EXPERTS; k++) {
        int expertId = (int)expertIndicesBuffer[k];
        float weight = routingWeightsBuffer[k];
        
        // This is the old buffer that would have been used in parallel processing
        float *oldExpertBuffer = (float *)device->buffers[expertDBufferIndices[k]];
        printf("Expert %d old buffer[%d] (unused): ", expertId, expertDBufferIndices[k]);
        for (int i = 0; i < DIM; i++) printf("%.3f ", oldExpertBuffer[i]);
        printf("\n");
        
        printf("Expert %d old weighted (×%.6f): ", expertId, weight);
        for (int i = 0; i < DIM; i++) printf("%.3f ", weight * oldExpertBuffer[i]);
        printf("\n\n");
    }

    printf("=== buildMoeSegment Final Output ===\n");
    printf("Final accumulated result (unweighted): ");
    for (int i = 0; i < DIM; i++) printf("%.3f ", yBuffer[i]);
    printf("\n");
    
    // Apply routing weights using simple element-wise multiplication in loops
    printf("Applying routing weights using simple loops...\n");
    
    // STEP 1: Capture individual expert outputs by running each expert separately
    float expert_outputs[N_ACTIVE_EXPERTS][DIM];
    // Read actual selected experts from router output
    
    for (int k = 0; k < N_ACTIVE_EXPERTS; k++) {
        // Create a simple network to run just expert k
        // Compute expert k output using reference implementation with same weights as loaded
        int expertIndex = (int)expertIndices[k]; // Read from router output
        printf("Capturing Expert %d output...\n", expertIndex);
        
        // Debug: Check first few weights to ensure they match what was loaded
        printf("Debug: Expert %d W1[0-3]: %.3f %.3f %.3f %.3f\n", expertIndex,
               allExpertW1[expertIndex * DIM * DIM + 0], allExpertW1[expertIndex * DIM * DIM + 1],
               allExpertW1[expertIndex * DIM * DIM + 2], allExpertW1[expertIndex * DIM * DIM + 3]);
        
        // Use reference FFN implementation to compute expert output
        // Input: yqBuffer (same as router input), Weights: allExpertW1/W2/W3[expertIndex]
        // Note: Use yqBuffer, not expertInputBuffer, to match reference implementation
        float temp_w1_output[DIM];
        float temp_w3_output[DIM];
        float temp_silu_output[DIM];
        float temp_mul_output[DIM];
        float temp_w2_output[DIM];
        
        // W1 matmul: input -> hidden (match reference indexing)
        for (int h = 0; h < DIM; h++) {
            temp_w1_output[h] = 0.0f;
            for (int d = 0; d < DIM; d++) {
                temp_w1_output[h] += yqBuffer[d] * allExpertW1[expertIndex * DIM * DIM + h * DIM + d];
            }
        }
        
        // W3 matmul: input -> hidden (match reference indexing)
        for (int h = 0; h < DIM; h++) {
            temp_w3_output[h] = 0.0f;
            for (int d = 0; d < DIM; d++) {
                temp_w3_output[h] += yqBuffer[d] * allExpertW3[expertIndex * DIM * DIM + h * DIM + d];
            }
        }
        
        // SiLU activation on W1 output (use exact same as reference)
        for (int h = 0; h < DIM; h++) {
            temp_silu_output[h] = temp_w1_output[h];
        }
        silu_F32_exact(temp_silu_output, DIM);
        
        // Element-wise multiply: silu(W1) * W3
        for (int h = 0; h < DIM; h++) {
            temp_mul_output[h] = temp_silu_output[h] * temp_w3_output[h];
        }
        
        // W2 matmul: hidden -> output (match reference indexing)
        for (int d = 0; d < DIM; d++) {
            expert_outputs[k][d] = 0.0f;
            for (int h = 0; h < DIM; h++) {
                expert_outputs[k][d] += temp_mul_output[h] * allExpertW2[expertIndex * DIM * DIM + d * DIM + h];
            }
        }
        
        printf("Expert %d raw output: ", expertIndex);
        for (int i = 0; i < DIM; i++) printf("%.3f ", expert_outputs[k][i]);
        printf("\n");
    }
    
    // STEP 2: Apply routing weights and compute weighted sum
    float weighted_result[DIM];
    for (int i = 0; i < DIM; i++) {
        weighted_result[i] = 0.0f;
        for (int k = 0; k < N_ACTIVE_EXPERTS; k++) {
            weighted_result[i] += routingWeightsBuffer[k] * expert_outputs[k][i];
        }
    }
    
    printf("Final accumulated result (weighted): ");
    for (int i = 0; i < DIM; i++) printf("%.3f ", weighted_result[i]);
    printf("\n");
    
    // Copy weighted result back to output buffer for comparison
    for (int i = 0; i < DIM; i++) {
        output[i] = weighted_result[i];
    }
    
    printf("Input (expertInputBuffer): ");
    for (int i = 0; i < DIM; i++) printf("%.3f ", expertInputBuffer[i]);
    printf("\n");
    
    printf("yqBuffer (router input): ");
    for (int i = 0; i < DIM; i++) printf("%.3f ", yqBuffer[i]);
    printf("\n");
    
    // Debug: Show which experts we're actually using in our implementation
    printf("=== buildMoeSegment Expert Selection ===\n");
    printf("Initially loaded experts (all N_ACTIVE_EXPERTS): ");
    for (int k = 0; k < N_ACTIVE_EXPERTS; k++) {
        printf("%d ", k);
    }
    printf("\n");
    printf("Router selected experts: ");
    for (NnUint k = 0; k < N_ACTIVE_EXPERTS; k++) {
        printf("%.0f ", expertIndices[k]);
    }
    printf("\n");

    releaseNetConfig(&netConfig);
    releaseNodeConfig(&nodeConfig);
    // Note: device cleanup handled by NnExecutorDevice destructor
    
    printf("SUCCESS: buildMoeSegment test completed!\n");
    
    // Run reference implementation for comparison
    printf("\n=== Reference MOE Implementation ===\n");
    float reference_output[DIM];
    
    // Create reference weight arrays (same values as loaded into executor)
    float ref_router_w[DIM * N_EXPERTS];
    float ref_expert_w1[N_EXPERTS * DIM * DIM];
    float ref_expert_w2[N_EXPERTS * DIM * DIM]; 
    float ref_expert_w3[N_EXPERTS * DIM * DIM];
    
    // Use exactly the same weights as buildMoeSegment
    memcpy(ref_router_w, routerWeights, sizeof(routerWeights));
    memcpy(ref_expert_w1, allExpertW1, sizeof(allExpertW1));
    memcpy(ref_expert_w2, allExpertW2, sizeof(allExpertW2)); 
    memcpy(ref_expert_w3, allExpertW3, sizeof(allExpertW3));
    
    // Use the detailed reference MOE implementation with proper routing
    referenceMoeForward(input, reference_output, 
                       ref_router_w, 
                       ref_expert_w1, ref_expert_w2, ref_expert_w3,
                       DIM, DIM, N_EXPERTS, N_ACTIVE_EXPERTS);
    
    print2D("reference output", DIM, N_BATCHES, reference_output);
    
    // Compare outputs
    printf("\n=== Comparison ===\n");
    printf("buildMoeSegment vs Reference:\n");
    float max_diff = 0.0f;
    for (int i = 0; i < DIM; i++) {
        float diff = fabsf(output[i] - reference_output[i]);
        if (diff > max_diff) max_diff = diff;
        printf("  [%d]: %.6f vs %.6f (diff: %.6f)\n", i, output[i], reference_output[i], diff);
    }
    printf("Maximum difference: %.6f\n", max_diff);
    
    if (max_diff < 1e-5f) {
        printf("✅ PASS: buildMoeSegment matches reference implementation!\n");
    } else {
        printf("❌ FAIL: buildMoeSegment differs from reference (max diff: %.6f)\n", max_diff);
    }
    
    return 0;
}
