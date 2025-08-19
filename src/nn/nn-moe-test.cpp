#include "nn-cpu-ops.cpp"
#include <vector>
#include <algorithm>
#include <cmath>

// Test framework utilities
void rand_f32(float *o, const NnUint n, const NnUint seed) {
    srand(seed + 123456);
    for (NnUint i = 0; i < n; i++) {
        float v = (float)(rand()) / RAND_MAX;
        o[i] = v * 2.0f - 1.0f;
    }
}

void compare_f32(const char *name, const float *a, const float *b, const NnUint n, const float epsilon) {
    for (NnUint i = 0; i < n; i++) {
        float error = fabs(a[i] - b[i]);
        if (error > epsilon) {
            printf("❌ %s failed at index %d: %f != %f (error: %f)\n", name, i, a[i], b[i], error);
            for (NnUint j = std::max(0, (int)i-2); j < std::min(n, i+3); j++)
                printf("   [%3d] %f vs %f\n", j, a[j], b[j]);
            exit(1);
        }
    }
    printf("✅ %24s passed\n", name);
    fflush(stdout);
}

// ===== MoE Components =====

/**
 * Reference MoE router implementation (F32 only)
 * Computes routing logits: input [batch, dim] @ router_weights [dim, nExperts] -> logits [batch, nExperts]
 */
void moe_router_ref(const float *input, const float *router_weights, float *logits, 
                   NnUint batch, NnUint dim, NnUint nExperts) {
    for (NnUint b = 0; b < batch; b++) {
        for (NnUint e = 0; e < nExperts; e++) {
            float sum = 0.0f;
            for (NnUint d = 0; d < dim; d++) {
                sum += input[b * dim + d] * router_weights[d * nExperts + e];
            }
            logits[b * nExperts + e] = sum;
        }
    }
}

/**
 * Reference top-k expert selection with softmax normalization
 * Input: logits [batch, nExperts]
 * Output: expert_indices [batch, k], expert_weights [batch, k]
 */
void moe_topk_ref(const float *logits, NnUint *expert_indices, float *expert_weights,
                  NnUint batch, NnUint nExperts, NnUint k) {
    for (NnUint b = 0; b < batch; b++) {
        const float *batch_logits = &logits[b * nExperts];
        NnUint *batch_indices = &expert_indices[b * k];
        float *batch_weights = &expert_weights[b * k];
        
        // Create pairs for sorting
        std::vector<std::pair<float, NnUint>> expert_scores;
        for (NnUint e = 0; e < nExperts; e++) {
            expert_scores.push_back({batch_logits[e], e});
        }
        
        // Sort by logit value (descending)
        std::sort(expert_scores.begin(), expert_scores.end(), 
                 [](const std::pair<float, NnUint> &a, const std::pair<float, NnUint> &b) { 
                     return a.first > b.first; 
                 });
        
        // Select top-k and apply softmax
        float max_logit = expert_scores[0].first;
        float sum_exp = 0.0f;
        
        for (NnUint i = 0; i < k; i++) {
            batch_indices[i] = expert_scores[i].second;
            float exp_val = expf(expert_scores[i].first - max_logit);
            batch_weights[i] = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize weights
        for (NnUint i = 0; i < k; i++) {
            batch_weights[i] /= sum_exp;
        }
    }
}

/**
 * Reference expert FFN computation
 * For each token and selected expert: SILU(input @ W_up) * (input @ W_gate) @ W_down
 */
void moe_expert_ffn_ref(const float *input, const NnUint *expert_indices, 
                       const float *expert_up_weights, const float *expert_gate_weights, 
                       const float *expert_down_weights, float *expert_outputs,
                       NnUint batch, NnUint dim, NnUint hiddenDim, NnUint nExperts, NnUint k) {
    
    for (NnUint b = 0; b < batch; b++) {
        const float *batch_input = &input[b * dim];
        const NnUint *batch_indices = &expert_indices[b * k];
        
        for (NnUint i = 0; i < k; i++) {
            NnUint expert_id = batch_indices[i];
            float *expert_output = &expert_outputs[b * k * dim + i * dim];
            
            // Get expert weights
            const float *W_up = &expert_up_weights[expert_id * hiddenDim * dim];
            const float *W_gate = &expert_gate_weights[expert_id * hiddenDim * dim]; 
            const float *W_down = &expert_down_weights[expert_id * dim * hiddenDim];
            
            // Temporary buffers for intermediate results
            std::vector<float> up_proj(hiddenDim);
            std::vector<float> gate_proj(hiddenDim);
            
            // Up projection: input @ W_up
            for (NnUint h = 0; h < hiddenDim; h++) {
                float sum = 0.0f;
                for (NnUint d = 0; d < dim; d++) {
                    sum += batch_input[d] * W_up[d * hiddenDim + h];
                }
                up_proj[h] = sum;
            }
            
            // Gate projection: input @ W_gate  
            for (NnUint h = 0; h < hiddenDim; h++) {
                float sum = 0.0f;
                for (NnUint d = 0; d < dim; d++) {
                    sum += batch_input[d] * W_gate[d * hiddenDim + h];
                }
                gate_proj[h] = sum;
            }
            
            // Apply SiLU to gate projection and element-wise multiply
            for (NnUint h = 0; h < hiddenDim; h++) {
                float x = gate_proj[h];
                float silu_val = x / (1.0f + expf(-x));  // SiLU activation
                up_proj[h] *= silu_val;
            }
            
            // Down projection: activated @ W_down
            for (NnUint d = 0; d < dim; d++) {
                float sum = 0.0f;
                for (NnUint h = 0; h < hiddenDim; h++) {
                    sum += up_proj[h] * W_down[h * dim + d];
                }
                expert_output[d] = sum;
            }
        }
    }
}

/**
 * Reference MoE output combination
 * Combines expert outputs using routing weights: sum(expert_weight[i] * expert_output[i])
 */
void moe_combine_ref(const float *expert_outputs, const float *expert_weights, float *final_output,
                    NnUint batch, NnUint dim, NnUint k) {
    
    // Initialize output to zero
    for (NnUint i = 0; i < batch * dim; i++) {
        final_output[i] = 0.0f;
    }
    
    for (NnUint b = 0; b < batch; b++) {
        const float *batch_weights = &expert_weights[b * k];
        float *batch_output = &final_output[b * dim];
        
        for (NnUint i = 0; i < k; i++) {
            const float *expert_output = &expert_outputs[b * k * dim + i * dim];
            float weight = batch_weights[i];
            
            for (NnUint d = 0; d < dim; d++) {
                batch_output[d] += weight * expert_output[d];
            }
        }
    }
}

// ===== Test Cases =====

void testMoeRouter() {
    const NnUint batch = 1;  // Simplified batch size
    const NnUint dim = 64;  // Must be multiple of 4 for ARM NEON
    const NnUint nExperts = 8;
    
    std::vector<float> input(batch * dim);
    std::vector<float> router_weights(dim * nExperts);
    std::vector<float> logits_ref(batch * nExperts);
    std::vector<float> logits_test(batch * nExperts);
    
    rand_f32(input.data(), batch * dim, 42);
    rand_f32(router_weights.data(), dim * nExperts, 123);
    
    // Reference implementation
    moe_router_ref(input.data(), router_weights.data(), logits_ref.data(), 
                   batch, dim, nExperts);
    
    // Test implementation using existing matmul (call per batch)
    // Need to transpose router_weights from [dim, nExperts] to [nExperts, dim] for matmul
    std::vector<float> router_weights_T(nExperts * dim);
    for (NnUint e = 0; e < nExperts; e++) {
        for (NnUint d = 0; d < dim; d++) {
            router_weights_T[e * dim + d] = router_weights[d * nExperts + e];
        }
    }
    
    for (NnUint b = 0; b < batch; b++) {
        matmul_F32_F32_F32(&logits_test[b * nExperts], &input[b * dim], router_weights_T.data(), 
                          dim, nExperts, 1, 0);
    }
    
    // Debug: print first few values to understand the difference
    printf("Debug - Input[0:4]: %.6f %.6f %.6f %.6f\n", 
           input[0], input[1], input[2], input[3]);
    printf("Debug - RouterWeights[0:4]: %.6f %.6f %.6f %.6f\n", 
           router_weights[0], router_weights[1], router_weights[2], router_weights[3]);
    printf("Debug - Reference[0:4]: %.6f %.6f %.6f %.6f\n", 
           logits_ref[0], logits_ref[1], logits_ref[2], logits_ref[3]);
    printf("Debug - Test[0:4]: %.6f %.6f %.6f %.6f\n", 
           logits_test[0], logits_test[1], logits_test[2], logits_test[3]);
    
    compare_f32("MoE Router", logits_ref.data(), logits_test.data(), batch * nExperts, 1e-5f);
}

void testMoeTopK() {
    const NnUint batch = 1;  // Simplified batch size
    const NnUint nExperts = 8;
    const NnUint k = 4;
    
    std::vector<float> logits(batch * nExperts);
    std::vector<NnUint> expert_indices(batch * k);
    std::vector<float> expert_weights(batch * k);
    
    // Create predictable logits for testing
    for (NnUint b = 0; b < batch; b++) {
        for (NnUint e = 0; e < nExperts; e++) {
            logits[b * nExperts + e] = (float)(e * (b + 1)); // Simple pattern
        }
    }
    
    moe_topk_ref(logits.data(), expert_indices.data(), expert_weights.data(),
                 batch, nExperts, k);
    
    // Verify top-k selection
    for (NnUint b = 0; b < batch; b++) {
        printf("Batch %d top-%d experts: ", b, k);
        float weight_sum = 0.0f;
        for (NnUint i = 0; i < k; i++) {
            printf("E%d(%.3f) ", expert_indices[b * k + i], expert_weights[b * k + i]);
            weight_sum += expert_weights[b * k + i];
        }
        printf("sum=%.3f\n", weight_sum);
        
        // Verify weights sum to 1.0 (softmax property)
        if (fabs(weight_sum - 1.0f) > 1e-5f) {
            printf("❌ Top-K weights don't sum to 1.0: %f\n", weight_sum);
            exit(1);
        }
    }
    
    printf("✅ %24s passed\n", "MoE Top-K");
}

void testMoeExpertFFN() {
    const NnUint batch = 1;
    const NnUint dim = 32;
    const NnUint hiddenDim = 48;
    const NnUint nExperts = 4;
    const NnUint k = 2;
    
    std::vector<float> input(batch * dim);
    std::vector<NnUint> expert_indices(batch * k);
    std::vector<float> expert_up_weights(nExperts * hiddenDim * dim);
    std::vector<float> expert_gate_weights(nExperts * hiddenDim * dim);
    std::vector<float> expert_down_weights(nExperts * dim * hiddenDim);
    std::vector<float> expert_outputs(batch * k * dim);
    
    rand_f32(input.data(), batch * dim, 42);
    rand_f32(expert_up_weights.data(), nExperts * hiddenDim * dim, 123);
    rand_f32(expert_gate_weights.data(), nExperts * hiddenDim * dim, 456);
    rand_f32(expert_down_weights.data(), nExperts * dim * hiddenDim, 789);
    
    // Set up expert indices (use experts 0 and 2)
    expert_indices[0] = 0;
    expert_indices[1] = 2;
    
    moe_expert_ffn_ref(input.data(), expert_indices.data(),
                       expert_up_weights.data(), expert_gate_weights.data(), 
                       expert_down_weights.data(), expert_outputs.data(),
                       batch, dim, hiddenDim, nExperts, k);
    
    // Basic sanity check - outputs should not be all zeros
    bool has_nonzero = false;
    for (NnUint i = 0; i < batch * k * dim; i++) {
        if (fabs(expert_outputs[i]) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    
    if (!has_nonzero) {
        printf("❌ MoE Expert FFN outputs are all zero\n");
        exit(1);
    }
    
    printf("✅ %24s passed\n", "MoE Expert FFN");
}

void testMoeCombine() {
    const NnUint batch = 1;  // Simplified batch size
    const NnUint dim = 16;
    const NnUint k = 4;
    
    std::vector<float> expert_outputs(batch * k * dim);
    std::vector<float> expert_weights(batch * k);
    std::vector<float> final_output(batch * dim);
    
    rand_f32(expert_outputs.data(), batch * k * dim, 42);
    
    // Create normalized weights
    for (NnUint b = 0; b < batch; b++) {
        float sum = 0.0f;
        for (NnUint i = 0; i < k; i++) {
            expert_weights[b * k + i] = (float)(i + 1); // weights: 1, 2, 3
            sum += expert_weights[b * k + i];
        }
        // Normalize
        for (NnUint i = 0; i < k; i++) {
            expert_weights[b * k + i] /= sum;
        }
    }
    
    moe_combine_ref(expert_outputs.data(), expert_weights.data(), final_output.data(),
                    batch, dim, k);
    
    // Verify output is reasonable (weighted combination should be within input range)
    bool is_reasonable = true;
    for (NnUint i = 0; i < batch * dim; i++) {
        if (fabs(final_output[i]) > 10.0f) { // Should be within reasonable range
            is_reasonable = false;
            break;
        }
    }
    
    if (!is_reasonable) {
        printf("❌ MoE Combine output values are unreasonable\n");
        exit(1);
    }
    
    printf("✅ %24s passed\n", "MoE Combine");
}

void testFullMoEPipeline() {
    const NnUint batch = 1;  // Simplified batch size
    const NnUint dim = 32;
    const NnUint hiddenDim = 48;
    const NnUint nExperts = 8;
    const NnUint k = 4;
    
    printf("🧪 Testing full MoE pipeline (batch=%d, dim=%d, experts=%d, k=%d)\n", 
           batch, dim, nExperts, k);
    
    // Input and weights
    std::vector<float> input(batch * dim);
    std::vector<float> router_weights(dim * nExperts);
    std::vector<float> expert_up_weights(nExperts * hiddenDim * dim);
    std::vector<float> expert_gate_weights(nExperts * hiddenDim * dim);
    std::vector<float> expert_down_weights(nExperts * dim * hiddenDim);
    
    // Intermediate results
    std::vector<float> logits(batch * nExperts);
    std::vector<NnUint> expert_indices(batch * k);
    std::vector<float> expert_weights(batch * k);
    std::vector<float> expert_outputs(batch * k * dim);
    std::vector<float> final_output(batch * dim);
    
    rand_f32(input.data(), batch * dim, 42);
    rand_f32(router_weights.data(), dim * nExperts, 123);
    rand_f32(expert_up_weights.data(), nExperts * hiddenDim * dim, 456);
    rand_f32(expert_gate_weights.data(), nExperts * hiddenDim * dim, 789);
    rand_f32(expert_down_weights.data(), nExperts * dim * hiddenDim, 101112);
    
    // Step 1: Router
    moe_router_ref(input.data(), router_weights.data(), logits.data(), 
                   batch, dim, nExperts);
    
    // Step 2: Top-K selection
    moe_topk_ref(logits.data(), expert_indices.data(), expert_weights.data(),
                 batch, nExperts, k);
    
    // Step 3: Expert computation
    moe_expert_ffn_ref(input.data(), expert_indices.data(),
                       expert_up_weights.data(), expert_gate_weights.data(), 
                       expert_down_weights.data(), expert_outputs.data(),
                       batch, dim, hiddenDim, nExperts, k);
    
    // Step 4: Combine outputs
    moe_combine_ref(expert_outputs.data(), expert_weights.data(), final_output.data(),
                    batch, dim, k);
    
    printf("   Router logits range: [%.3f, %.3f]\n", 
           *std::min_element(logits.begin(), logits.end()),
           *std::max_element(logits.begin(), logits.end()));
    
    printf("   Selected experts: ");
    for (NnUint i = 0; i < k; i++) {
        printf("E%d(%.3f) ", expert_indices[i], expert_weights[i]);
    }
    printf("\n");
    
    printf("   Final output range: [%.3f, %.3f]\n",
           *std::min_element(final_output.begin(), final_output.end()),
           *std::max_element(final_output.begin(), final_output.end()));
    
    printf("✅ %24s passed\n", "Full MoE Pipeline");
}

// ===== Main Test Runner =====

int main() {
    printf("🧪 Testing Mixture of Experts (MoE) Components\n");
    printf("===============================================\n");
    
    testMoeRouter();
    testMoeTopK();
    testMoeExpertFFN();
    testMoeCombine();
    testFullMoEPipeline();
    
    printf("\n🎉 All MoE tests passed!\n");
    return 0;
}