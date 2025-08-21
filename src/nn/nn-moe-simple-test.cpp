#include "nn-core.hpp"
#include "nn-config-builder.hpp"
#include "nn-cpu.hpp"
#include "../llm.hpp"
#include <cstdio>
#include <cstdlib>

#define DIM 8
#define N_EXPERTS 4
#define N_ACTIVE_EXPERTS 2
#define N_BATCHES 1

void buildRouterTestConfig(NnNetConfig *netConfig, NnNodeConfig *nodeConfig) {
    NnUint nNodes = 1;
    NnNetConfigBuilder netBuilder(nNodes, N_BATCHES);
    NnUint inputPipeIndex = netBuilder.addPipe("INPUT", size2D(F_32, N_BATCHES, DIM));
    NnUint outputPipeIndex = netBuilder.addPipe("OUTPUT", size2D(F_32, N_BATCHES, DIM)); // Changed to DIM for matmul output

    NnNodeConfigBuilder nodeBuilder(0);
    NnUint expertIndicesBufferIndex = nodeBuilder.addBuffer("expert_indices", size2D(F_32, N_BATCHES, N_ACTIVE_EXPERTS));
    NnUint routingWeightsBufferIndex = nodeBuilder.addBuffer("routing_weights", size2D(F_32, N_BATCHES, N_ACTIVE_EXPERTS));
    NnUint routerOutputBufferIndex = nodeBuilder.addBuffer("router_output", size2D(F_32, N_BATCHES, N_EXPERTS));
    
    // Add FFN buffers for buildFfnSegment
    NnUint ffnInputBufferIndex = nodeBuilder.addBuffer("ffn_input", size2D(F_32, N_BATCHES, DIM));
    NnUint ffnWorkBufferIndex = nodeBuilder.addBuffer("ffn_work", size2D(F_32, N_BATCHES, DIM));
    NnUint ffnD1BufferIndex = nodeBuilder.addBuffer("ffn_d1", size2D(F_32, N_BATCHES, DIM)); // HIDDEN_DIM -> DIM for simplicity
    NnUint ffnD2BufferIndex = nodeBuilder.addBuffer("ffn_d2", size2D(F_32, N_BATCHES, DIM));
    NnUint ffnLBufferIndex = nodeBuilder.addBuffer("ffn_l", size2D(F_32, N_BATCHES, DIM));
    
    NnSegmentConfigBuilder segmentBuilder;

    segmentBuilder.addOp(OP_ROUTER, "test_router", 0,
        pointerBatchConfig(SRC_PIPE, inputPipeIndex),
        pointerBatchConfig(SRC_BUFFER, routerOutputBufferIndex), // Router output to buffer
        size2D(F_32, DIM, N_EXPERTS),
        NnRouterOpConfig{N_EXPERTS, N_ACTIVE_EXPERTS, expertIndicesBufferIndex, 
                        routingWeightsBufferIndex, size2D(F_32, DIM, N_EXPERTS)});

    // Copy input to FFN input buffer  
    segmentBuilder.addOp(
        OP_CAST, "copy_to_ffn", 0,
        pointerBatchConfig(SRC_PIPE, inputPipeIndex),
        pointerBatchConfig(SRC_BUFFER, ffnInputBufferIndex),
        size0(),
        NnCastOpCodeConfig{});
    
    // Call buildFfnSegment - THIS IS THE KEY INTEGRATION!
    LlmHeader header;
    header.dim = DIM;
    header.hiddenDim = DIM; // Keep same as DIM for simplicity
    header.weightType = F_32;
    
    NnRowMatmulSlice w1Slice = sliceRowMatmul(F_32, 1, DIM, DIM);
    NnColMatmulSlice w2Slice = sliceColMatmul(F_32, 1, DIM, DIM);
    NnRowMatmulSlice w3Slice = sliceRowMatmul(F_32, 1, DIM, DIM);
    
    buildFfnSegment(segmentBuilder, &header, 1, // layer 1
                   ffnInputBufferIndex,  // yBufferIndex - input, will be overwritten with output
                   ffnWorkBufferIndex,   // yqBufferIndex - work buffer
                   ffnD1BufferIndex,     // dBufferIndex - W1 output  
                   ffnD2BufferIndex,     // dqBufferIndex - work buffer for final step
                   ffnLBufferIndex,      // lBufferIndex - W3 output
                   w1Slice, w2Slice, w3Slice);
    
    // Copy FFN output to final output pipe
    segmentBuilder.addOp(
        OP_CAST, "ffn_to_output", 0,
        pointerBatchConfig(SRC_BUFFER, ffnInputBufferIndex), // buildFfnSegment writes result here
        pointerBatchConfig(SRC_PIPE, outputPipeIndex),
        size0(),
        NnCastOpCodeConfig{});

    nodeBuilder.addSegment(segmentBuilder.build());

    *netConfig = netBuilder.build();
    *nodeConfig = nodeBuilder.build();
}

void print2D(const char *name, NnUint x, NnUint y, float *w) {
    for (NnUint i = 0; i < y; i++) {
        printf("%s[%d] = ", name, i);
        for (NnUint j = 0; j < x; j++)
            printf("%f ", w[i * x + j]);
        printf("\n");
    }
}

int main() {
    initQuants();

    NnUint nThreads = 2;
    NnNetConfig netConfig;
    NnNodeConfig nodeConfig;
    buildRouterTestConfig(&netConfig, &nodeConfig);

    NnNetExecution execution(nThreads, &netConfig);
    float *x = (float *)execution.pipes[0];
    for (NnUint b = 0; b < N_BATCHES; b++) {
        for (NnUint i = 0; i < DIM; i++)
            x[b * DIM + i] = i / (float)DIM + (float)b;
    }

    print2D("x", DIM, N_BATCHES, x);

    // Create router weights (DIM x N_EXPERTS) with more variation
    float routerWeights[DIM * N_EXPERTS];
    for (NnUint expert = 0; expert < N_EXPERTS; expert++) {
        for (NnUint dim = 0; dim < DIM; dim++) {
            routerWeights[expert * DIM + dim] = 0.1f + expert * 0.2f + dim * 0.05f;
        }
    }

    NnCpuDevice *device = new NnCpuDevice(&netConfig, &nodeConfig, &execution);
    std::vector<NnExecutorDevice> devices;
    devices.push_back(NnExecutorDevice(device, -1, -1));

    NnFakeNodeSynchronizer synchronizer;
    float *expertIndices = (float *)device->buffers[0];
    float *routingWeights = (float *)device->buffers[1];
    NnExecutor executor(&netConfig, &nodeConfig, &devices, &execution, &synchronizer, false);
    executor.loadWeight("test_router", 0, sizeof(routerWeights), (NnByte *)routerWeights);
    
    // Load buildFfnSegment weights for layer 1  
    float w1Weights[DIM * DIM];
    for (NnUint i = 0; i < DIM * DIM; i++) {
        w1Weights[i] = (i % (DIM + 1) == 0) ? 1.0f : 0.1f; // Near-identity
    }
    executor.loadWeight("block_matmul_w1", 1, sizeof(w1Weights), (NnByte *)w1Weights);
    
    float w2Weights[DIM * DIM];
    for (NnUint i = 0; i < DIM * DIM; i++) {
        w2Weights[i] = (i % (DIM + 1) == 0) ? 1.0f : 0.0f; // Identity  
    }
    executor.loadWeight("block_matmul_w2", 1, sizeof(w2Weights), (NnByte *)w2Weights);
    
    float w3Weights[DIM * DIM];
    for (NnUint i = 0; i < DIM * DIM; i++) {
        w3Weights[i] = (i % (DIM + 1) == 0) ? 1.0f : 0.1f; // Near-identity
    }
    executor.loadWeight("block_matmul_w3", 1, sizeof(w3Weights), (NnByte *)w3Weights);

    execution.setBatchSize(N_BATCHES);
    executor.forward();

    // Print FFN outputs (now DIM, not N_EXPERTS)
    float *output = (float *)execution.pipes[1];
    print2D("FFN Output", DIM, N_BATCHES, output);
    
    printf("Expert selections:\n");
    for (NnUint b = 0; b < N_BATCHES; b++) {
        printf("Batch %d - experts: ", b);
        for (NnUint k = 0; k < N_ACTIVE_EXPERTS; k++) {
            printf("%d(%.3f) ", (int)expertIndices[b * N_ACTIVE_EXPERTS + k], 
                   routingWeights[b * N_ACTIVE_EXPERTS + k]);
        }
        printf("\n");
    }

    releaseNetConfig(&netConfig);
    releaseNodeConfig(&nodeConfig);
    return 0;
}