# Qwen3-30B-A3 MoE Testing Guide

## Quick Start

### Prerequisites
```bash
# Ensure you have the model files:
ls -la dllama_model_qwen3-30b-a3.m     # ~20-30GB
ls -la dllama_tokenizer_qwen3-30b-a3.t # ~2MB

# Build the project
make clean && make dllama
```

### Automated Testing
```bash
# Run comprehensive test suite
./test_qwen3_30b_moe.sh

# Or specify custom paths
./test_qwen3_30b_moe.sh /path/to/model.m /path/to/tokenizer.t
```

## Manual Testing Commands

### 1. Single Node Testing

#### Basic Inference Test
```bash
./dllama inference \
  --model dllama_model_qwen3-30b-a3.m \
  --tokenizer dllama_tokenizer_qwen3-30b-a3.t \
  --prompt "Explain quantum computing" \
  --steps 30 \
  --buffer-float-type q80 \
  --nthreads 8
```

#### Interactive Chat Test
```bash
./dllama chat \
  --model dllama_model_qwen3-30b-a3.m \
  --tokenizer dllama_tokenizer_qwen3-30b-a3.t \
  --buffer-float-type q80 \
  --nthreads 8
```

### 2. Two-Node Distributed Testing

#### Terminal 1 (Worker)
```bash
./dllama worker --port 9998 --nthreads 8
```

#### Terminal 2 (Root)
```bash
./dllama chat \
  --model dllama_model_qwen3-30b-a3.m \
  --tokenizer dllama_tokenizer_qwen3-30b-a3.t \
  --workers 127.0.0.1:9998 \
  --buffer-float-type q80 \
  --nthreads 8
```

### 3. Four-Node Distributed Testing

#### Terminals 1-3 (Workers)
```bash
# Terminal 1
./dllama worker --port 9998 --nthreads 6

# Terminal 2  
./dllama worker --port 9999 --nthreads 6

# Terminal 3
./dllama worker --port 10000 --nthreads 6
```

#### Terminal 4 (Root)
```bash
./dllama chat \
  --model dllama_model_qwen3-30b-a3.m \
  --tokenizer dllama_tokenizer_qwen3-30b-a3.t \
  --workers 127.0.0.1:9998 127.0.0.1:9999 127.0.0.1:10000 \
  --buffer-float-type q80 \
  --nthreads 6
```

### 4. Network Distributed Testing

#### Multiple Machines Setup
```bash
# Machine 1 (IP: 10.0.0.1)
./dllama worker --port 9998 --nthreads 8

# Machine 2 (IP: 10.0.0.2)
./dllama worker --port 9998 --nthreads 8

# Machine 3 (IP: 10.0.0.3)  
./dllama worker --port 9998 --nthreads 8

# Root Machine (IP: 10.0.0.0)
./dllama chat \
  --model dllama_model_qwen3-30b-a3.m \
  --tokenizer dllama_tokenizer_qwen3-30b-a3.t \
  --workers 10.0.0.1:9998 10.0.0.2:9998 10.0.0.3:9998 \
  --buffer-float-type q80 \
  --nthreads 8
```

## Expected Outputs

### Architecture Recognition
Look for these key indicators in the output:
```
💡 Arch: Qwen3MoE
💡 Experts: 128
💡 ActiveExperts: 8
💡 HiddenDim: 768
💡 Dim: 2048
🔀 MoE weights loaded: router + 128 experts with tensor parallelism (X nodes)
```

### Performance Metrics
```
📊 Performance: X.X tokens/s
💾 Memory: X.X GB per node
🔗 Nodes: X active workers + 1 root
⚡ Speedup: X.Xx vs single node
```

### Memory Usage Expectations
| Configuration | Memory per Node | Total Memory |
|---------------|-----------------|--------------|
| 1 node        | ~20GB           | ~20GB        |
| 2 nodes       | ~10GB           | ~20GB        |
| 4 nodes       | ~5GB            | ~20GB        |
| 8 nodes       | ~2.5GB          | ~20GB        |

## Troubleshooting

### Model Loading Issues
```bash
# Check file integrity
ls -la dllama_model_qwen3-30b-a3.m

# Verify architecture
head -c 1024 dllama_model_qwen3-30b-a3.m | hexdump -C
```

### Memory Issues
```bash
# Reduce sequence length
--max-seq-len 2048

# Monitor memory usage
htop
# or on macOS
vm_stat
```

### Network Issues
```bash
# Test worker connectivity
telnet 127.0.0.1 9998

# Check network performance
ping 10.0.0.1
iperf3 -c 10.0.0.1
```

### Performance Issues
```bash
# Optimize threads per CPU
--nthreads $(nproc)

# Try different buffer types
--buffer-float-type f32  # Higher memory, potentially faster
--buffer-float-type q80  # Lower memory, standard
```

## Validation Checklist

### ✅ Successful Test Indicators
- [ ] Architecture correctly recognized as Qwen3MoE
- [ ] MoE weight loading message appears
- [ ] Single node inference produces coherent text
- [ ] Multi-node setup connects successfully
- [ ] Performance scales with number of nodes
- [ ] Memory usage decreases per node with more nodes
- [ ] No segmentation faults or crashes

### ⚠️ Warning Signs
- [ ] Architecture recognized as Qwen3 instead of Qwen3MoE
- [ ] No MoE weight loading messages
- [ ] Performance doesn't scale with nodes
- [ ] High memory usage per node in multi-node setup
- [ ] Network timeouts or connection errors

## Performance Benchmarks

### Test Prompts for Consistent Measurement
```bash
# Short prompt (quick test)
"Explain artificial intelligence in 30 words."

# Medium prompt (standard test)  
"Describe the benefits of distributed computing for machine learning applications."

# Long prompt (stress test)
"Write a comprehensive explanation of how neural networks learn, including backpropagation, gradient descent, and optimization techniques. Provide examples and discuss modern architectures."
```

### Measurement Commands
```bash
# Time measurement
time ./dllama inference --model ... --prompt "..." --steps 50

# Memory monitoring during inference
# Terminal 1: Run inference
# Terminal 2: watch -n 1 'ps aux | grep dllama'
```

## Configuration Optimization

### Thread Optimization
```bash
# Get CPU info
nproc                    # Linux
sysctl -n hw.ncpu       # macOS

# Recommended thread counts:
# Single node: nproc
# Multi-node: nproc / nodes_per_machine
```

### Memory Optimization
```bash
# For limited memory systems:
--max-seq-len 1024      # Reduce context length
--buffer-float-type q80 # Use quantized buffers

# For high-memory systems:
--max-seq-len 4096      # Full context
--buffer-float-type f32 # Full precision
```

This testing framework validates the weight-split MoE implementation and provides clear metrics for performance scaling and resource utilization across different node configurations.