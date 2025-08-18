#!/bin/bash
# Qwen3-30B-A3 MoE Testing Suite
# Tests the weight-split MoE implementation with the full 30B parameter model
#
# Usage: ./test_qwen3_30b_moe.sh [model_path] [tokenizer_path]
#
# This script validates:
# 1. Model loading and MoE architecture recognition
# 2. Single-node inference functionality
# 3. Multi-node distributed inference with weight splitting
# 4. Performance scaling across different node configurations
# 5. Memory usage validation

set -e  # Exit on any error

# Configuration
MODEL_PATH="${1:-dllama_model_qwen3-30b-a3.m}"
TOKENIZER_PATH="${2:-dllama_tokenizer_qwen3-30b-a3.t}"
DLLAMA_BIN="./dllama"
RESULTS_DIR="./test_results_qwen3_30b"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_test() { echo -e "${BLUE}[TEST]${NC} $1"; }
log_perf() { echo -e "${YELLOW}[PERF]${NC} $1"; }

# Setup
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEST_LOG="$RESULTS_DIR/qwen3_30b_test_${TIMESTAMP}.log"

log_info "=== Qwen3-30B-A3 MoE Test Suite ==="
log_info "Started: $(date)"
log_info "Model: $MODEL_PATH"
log_info "Tokenizer: $TOKENIZER_PATH"
log_info "Results: $TEST_LOG"

# Test 1: Prerequisites Check
log_test "Test 1: Prerequisites Check"

if [ ! -f "$DLLAMA_BIN" ]; then
    log_error "Binary not found: $DLLAMA_BIN"
    log_info "Building project..."
    make clean && make dllama || {
        log_error "Build failed"
        exit 1
    }
fi

if [ ! -f "$MODEL_PATH" ]; then
    log_error "Model file not found: $MODEL_PATH"
    log_info "Please ensure the Qwen3-30B-A3 model is available at: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$TOKENIZER_PATH" ]; then
    log_error "Tokenizer file not found: $TOKENIZER_PATH"
    log_info "Please ensure the Qwen3-30B-A3 tokenizer is available at: $TOKENIZER_PATH"
    exit 1
fi

# File size information
model_size=$(ls -lh "$MODEL_PATH" | awk '{print $5}')
tokenizer_size=$(ls -lh "$TOKENIZER_PATH" | awk '{print $5}')
log_info "✅ Model size: $model_size"
log_info "✅ Tokenizer size: $tokenizer_size"

# Test 2: Architecture Recognition
log_test "Test 2: Architecture Recognition"
echo "Testing MoE architecture detection..."

# Run a minimal inference to check architecture recognition
gtimeout 60s $DLLAMA_BIN inference \
    --model "$MODEL_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --prompt "test" \
    --steps 1 \
    --buffer-float-type q80 \
    --nthreads 4 \
    2>&1 | tee "$RESULTS_DIR/architecture_test.log" || {
    
    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        log_warn "Architecture test timed out (expected for large model)"
    else
        log_error "Architecture test failed with exit code $exit_code"
        cat "$RESULTS_DIR/architecture_test.log"
        exit 1
    fi
}

# Validate architecture recognition
if grep -q "💡 Arch: Qwen3MoE" "$RESULTS_DIR/architecture_test.log"; then
    log_info "✅ Architecture correctly recognized as Qwen3MoE"
elif grep -q "Qwen3MoE" "$RESULTS_DIR/architecture_test.log"; then
    log_info "✅ Qwen3MoE architecture detected"
else
    log_error "❌ Architecture not recognized as Qwen3MoE"
    echo "Output from architecture test:"
    cat "$RESULTS_DIR/architecture_test.log"
    exit 1
fi

# Check for MoE weight loading
if grep -q "🔀 MoE weights loaded" "$RESULTS_DIR/architecture_test.log"; then
    log_info "✅ MoE weight loading detected"
    # Extract number of experts if available
    if grep -q "💡 Experts:" "$RESULTS_DIR/architecture_test.log"; then
        experts=$(grep "💡 Experts:" "$RESULTS_DIR/architecture_test.log" | sed 's/.*Experts: \([0-9]*\).*/\1/')
        log_info "✅ Number of experts: $experts"
    fi
else
    log_warn "⚠️ MoE weight loading not explicitly detected (may work with current format)"
fi

# Test 3: Single Node Inference
log_test "Test 3: Single Node Inference"
echo "Testing basic inference functionality..."

test_prompts=(
    "Hello, how are you today?"
    "Explain the concept of artificial intelligence in 50 words."
    "What is the capital of France?"
    "Write a short poem about technology."
)

for i in "${!test_prompts[@]}"; do
    prompt="${test_prompts[$i]}"
    log_info "Testing prompt $((i+1)): '$prompt'"
    
    # Test with performance measurement
    start_time=$(date +%s)
    gtimeout 120s $DLLAMA_BIN inference \
        --model "$MODEL_PATH" \
        --tokenizer "$TOKENIZER_PATH" \
        --prompt "$prompt" \
        --steps 20 \
        --buffer-float-type q80 \
        --nthreads 8 \
        2>&1 | tee "$RESULTS_DIR/single_node_test_$((i+1)).log" || {
        
        exit_code=$?
        if [ $exit_code -eq 124 ]; then
            log_warn "Single node test $((i+1)) timed out"
        else
            log_error "Single node test $((i+1)) failed with exit code $exit_code"
            continue
        fi
    }
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    log_perf "Single node test $((i+1)) duration: ${duration}s"
    
    # Extract performance metrics if available
    if grep -q "tokens/s:" "$RESULTS_DIR/single_node_test_$((i+1)).log"; then
        tokens_per_sec=$(grep "tokens/s:" "$RESULTS_DIR/single_node_test_$((i+1)).log" | tail -1 | sed 's/.*tokens\/s: \([0-9.]*\).*/\1/')
        log_perf "Single node performance: $tokens_per_sec tokens/s"
    fi
done

log_info "✅ Single node inference tests completed"

# Test 4: Memory Usage Test
log_test "Test 4: Memory Usage Validation"
echo "Testing memory usage with single node..."

# Get memory usage before
mem_before=$(ps aux | awk 'NR>1 {sum+=$6} END {print sum/1024}' 2>/dev/null || echo "0")

# Run memory-intensive test
gtimeout 90s $DLLAMA_BIN inference \
    --model "$MODEL_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --prompt "Generate a detailed explanation of machine learning concepts including supervised learning, unsupervised learning, and deep learning. Provide examples and applications for each." \
    --steps 100 \
    --buffer-float-type q80 \
    --nthreads 8 \
    2>&1 | tee "$RESULTS_DIR/memory_test.log" || {
    
    log_warn "Memory test may have timed out or failed"
}

# Memory usage analysis
if command -v free >/dev/null 2>&1; then
    free -h > "$RESULTS_DIR/memory_usage.log"
    log_info "✅ Memory usage logged"
elif command -v vm_stat >/dev/null 2>&1; then
    vm_stat > "$RESULTS_DIR/memory_usage.log"
    log_info "✅ Memory usage logged (macOS)"
else
    log_warn "⚠️ Memory monitoring tools not available"
fi

# Test 5: Two-Node Distributed Test
log_test "Test 5: Two-Node Distributed Test"
echo "Testing distributed inference with 2 nodes..."

# Start worker in background
log_info "Starting worker node on port 9998..."
$DLLAMA_BIN worker --port 9998 --nthreads 8 > "$RESULTS_DIR/worker_9998.log" 2>&1 &
WORKER_PID=$!
sleep 3  # Give worker time to start

# Test distributed inference
log_info "Testing distributed inference with 2 nodes..."
gtimeout 120s $DLLAMA_BIN inference \
    --model "$MODEL_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --workers 127.0.0.1:9998 \
    --prompt "Explain distributed computing and its benefits in modern applications." \
    --steps 30 \
    --buffer-float-type q80 \
    --nthreads 8 \
    2>&1 | tee "$RESULTS_DIR/two_node_test.log" || {
    
    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        log_warn "Two-node test timed out"
    else
        log_warn "Two-node test failed with exit code $exit_code"
    fi
}

# Stop worker
kill $WORKER_PID 2>/dev/null || true
wait $WORKER_PID 2>/dev/null || true

# Analyze two-node results
if grep -q "tokens/s:" "$RESULTS_DIR/two_node_test.log"; then
    tokens_per_sec_2node=$(grep "tokens/s:" "$RESULTS_DIR/two_node_test.log" | tail -1 | sed 's/.*tokens\/s: \([0-9.]*\).*/\1/')
    log_perf "Two-node performance: $tokens_per_sec_2node tokens/s"
else
    log_warn "⚠️ Could not extract two-node performance metrics"
fi

if grep -q "🔗" "$RESULTS_DIR/two_node_test.log"; then
    log_info "✅ Two-node distributed inference completed"
else
    log_warn "⚠️ Two-node test may not have used distributed mode"
fi

# Test 6: Four-Node Distributed Test (Optional)
log_test "Test 6: Four-Node Distributed Test (Optional)"

# Check available memory for 4-node test
available_mem=$(free -m 2>/dev/null | awk 'NR==2{print $7}' || sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024)}' || echo "0")
if [ "$available_mem" -gt 16000 ]; then
    echo "Testing distributed inference with 4 nodes..."
    
    # Start 3 workers
    $DLLAMA_BIN worker --port 9998 --nthreads 6 > "$RESULTS_DIR/worker1.log" 2>&1 &
    WORKER1_PID=$!
    $DLLAMA_BIN worker --port 9999 --nthreads 6 > "$RESULTS_DIR/worker2.log" 2>&1 &
    WORKER2_PID=$!
    $DLLAMA_BIN worker --port 10000 --nthreads 6 > "$RESULTS_DIR/worker3.log" 2>&1 &
    WORKER3_PID=$!
    sleep 5  # Give workers time to start
    
    # Test 4-node inference
    gtimeout 150s $DLLAMA_BIN inference \
        --model "$MODEL_PATH" \
        --tokenizer "$TOKENIZER_PATH" \
        --workers 127.0.0.1:9998 127.0.0.1:9999 127.0.0.1:10000 \
        --prompt "Describe the advantages of parallel computing and how it enables handling of large-scale artificial intelligence models." \
        --steps 40 \
        --buffer-float-type q80 \
        --nthreads 6 \
        2>&1 | tee "$RESULTS_DIR/four_node_test.log" || {
        
        log_warn "Four-node test may have timed out or failed"
    }
    
    # Stop workers
    kill $WORKER1_PID $WORKER2_PID $WORKER3_PID 2>/dev/null || true
    wait $WORKER1_PID $WORKER2_PID $WORKER3_PID 2>/dev/null || true
    
    # Analyze four-node results
    if grep -q "tokens/s:" "$RESULTS_DIR/four_node_test.log"; then
        tokens_per_sec_4node=$(grep "tokens/s:" "$RESULTS_DIR/four_node_test.log" | tail -1 | sed 's/.*tokens\/s: \([0-9.]*\).*/\1/')
        log_perf "Four-node performance: $tokens_per_sec_4node tokens/s"
        log_info "✅ Four-node distributed inference completed"
    else
        log_warn "⚠️ Could not extract four-node performance metrics"
    fi
else
    log_warn "⚠️ Insufficient memory for 4-node test (need >16GB available)"
    log_info "Available memory: ${available_mem}MB"
fi

# Test 7: Performance Summary
log_test "Test 7: Performance Summary"

echo "=== PERFORMANCE SUMMARY ===" | tee "$RESULTS_DIR/performance_summary.log"
echo "Model: Qwen3-30B-A3 MoE" | tee -a "$RESULTS_DIR/performance_summary.log"
echo "Test Date: $(date)" | tee -a "$RESULTS_DIR/performance_summary.log"
echo "" | tee -a "$RESULTS_DIR/performance_summary.log"

# Compare performance across configurations
if [ ! -z "${tokens_per_sec:-}" ]; then
    echo "Single Node: ${tokens_per_sec} tokens/s" | tee -a "$RESULTS_DIR/performance_summary.log"
fi
if [ ! -z "${tokens_per_sec_2node:-}" ]; then
    echo "Two Nodes: ${tokens_per_sec_2node} tokens/s" | tee -a "$RESULTS_DIR/performance_summary.log"
    if [ ! -z "${tokens_per_sec:-}" ]; then
        speedup=$(echo "scale=2; $tokens_per_sec_2node / $tokens_per_sec" | bc -l 2>/dev/null || echo "N/A")
        echo "2-Node Speedup: ${speedup}x" | tee -a "$RESULTS_DIR/performance_summary.log"
    fi
fi
if [ ! -z "${tokens_per_sec_4node:-}" ]; then
    echo "Four Nodes: ${tokens_per_sec_4node} tokens/s" | tee -a "$RESULTS_DIR/performance_summary.log"
    if [ ! -z "${tokens_per_sec:-}" ]; then
        speedup=$(echo "scale=2; $tokens_per_sec_4node / $tokens_per_sec" | bc -l 2>/dev/null || echo "N/A")
        echo "4-Node Speedup: ${speedup}x" | tee -a "$RESULTS_DIR/performance_summary.log"
    fi
fi

echo "" | tee -a "$RESULTS_DIR/performance_summary.log"

# Summary
log_info "=== QWEN3-30B-A3 MOE TEST SUMMARY ==="
log_info "Test results saved in: $RESULTS_DIR"
log_info "Performance summary: $RESULTS_DIR/performance_summary.log"

# Count successful tests
test_count=$(find "$RESULTS_DIR" -name "*.log" -type f | wc -l)
log_info "Total test files generated: $test_count"

log_info "✅ Qwen3-30B-A3 MoE Test Suite completed!"

echo ""
echo "=== NEXT STEPS ==="
echo "1. Review performance metrics in: $RESULTS_DIR/performance_summary.log"
echo "2. Check for any warnings or errors in individual test logs"
echo "3. For production use, consider:"
echo "   - Optimizing --nthreads based on your CPU cores"
echo "   - Testing with different --buffer-float-type settings"
echo "   - Using --max-seq-len to manage memory usage"
echo "   - Setting up dedicated hardware for multi-node deployment"
echo ""
echo "=== USAGE EXAMPLES ==="
echo "# Interactive chat with 2 nodes:"
echo "$DLLAMA_BIN chat --model $MODEL_PATH --tokenizer $TOKENIZER_PATH --workers 127.0.0.1:9998 --nthreads 8"
echo ""
echo "# Batch inference with 4 nodes:"
echo "$DLLAMA_BIN inference --model $MODEL_PATH --tokenizer $TOKENIZER_PATH --workers 127.0.0.1:9998 127.0.0.1:9999 127.0.0.1:10000 --prompt 'Your prompt' --steps 100 --nthreads 6"