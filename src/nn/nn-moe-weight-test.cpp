#include "../llm.hpp"
#include <iostream>
#include <vector>

void testModelLoading() {
    const char* modelPath = "./converter/dllama_model_qwen3-30b-a3b-fp32_f32.m";
    
    printf("🧪 Testing MoE Model Weight Loading\n");
    printf("=====================================\n");
    
    try {
        // Load model header
        LlmHeader header = loadLlmHeader(modelPath, 0, F_32);
        
        printf("📋 Model Information:\n");
        printf("   Architecture: %s\n", (header.archType == QWEN3_MOE) ? "Qwen3MoE" : "Other");
        printf("   Dimensions: %d hidden, %d heads, %d layers\n", 
               header.dim, header.nHeads, header.nLayers);
        printf("   Experts: %d total, %d active\n", header.nExperts, header.nActiveExperts);
        printf("   Vocab Size: %d\n", header.vocabSize);
        printf("   Weight Type: %s\n", (header.weightType == F_32) ? "F32" : "Other");
        
        if (header.archType != QWEN3_MOE) {
            printf("❌ Expected Qwen3MoE architecture, got %d\n", header.archType);
            return;
        }
        
        if (header.nExperts == 0) {
            printf("❌ Expected non-zero number of experts\n");
            return;
        }
        
        if (header.weightType != F_32) {
            printf("❌ Expected F32 weights, got %d\n", header.weightType);
            return;
        }
        
        printf("✅ Model header loaded successfully!\n");
        
        // Test building the network (without loading full weights)
        printf("\n🔗 Testing Network Construction:\n");
        LlmNet net = buildLlmNet(&header, 1, 1); // 1 node, 1 batch
        
        printf("   Network pipes: %d\n", net.netConfig.nPipes);
        printf("   Node segments: %d\n", net.nodeConfigs[0].nSegments);
        printf("   Node buffers: %d\n", net.nodeConfigs[0].nBuffers);
        
        releaseLlmNet(&net);
        printf("✅ Network construction successful!\n");
        
    } catch (const std::exception& e) {
        printf("❌ Test failed: %s\n", e.what());
    }
}

int main() {
    testModelLoading();
    return 0;
}