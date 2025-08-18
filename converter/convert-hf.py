import gc
import json
import sys
import os
from writer import parseFloatType, writeTensor, writeHeader, FloatType
from safetensors import safe_open

class ArchType:
    """Architecture type constants for different model architectures.
    
    Each architecture has specific handling for weights, attention mechanisms,
    and layer organizations.
    """
    LLAMA = 0xABCD00      # Standard transformer (Llama, Mistral)
    QWEN3 = 0xABCD01      # Qwen3 dense model with Q/K normalization
    QWEN3_MOE = 0xABCD02  # Qwen3 Mixture of Experts model

def permute(tensor, nHeads: int, nKvHeads: int):
    """Permute tensor for attention weight compatibility.
    
    Applies head-specific permutation for Q/K tensors in Llama architectures.
    For grouped query attention, uses nKvHeads instead of nHeads.
    
    Args:
        tensor: Input weight tensor to permute
        nHeads: Number of query/attention heads
        nKvHeads: Number of key-value heads (for GQA)
        
    Returns:
        Permuted tensor with correct head organization
    """
    if nHeads != nKvHeads:
        nHeads = nKvHeads
    return (tensor.reshape(nHeads, 2, tensor.shape[0] // nHeads // 2, *tensor.shape[1:]).swapaxes(1, 2).reshape(tensor.shape))

class Processor:
    """Processes model weights and converts them to distributed-llama format.
    
    Handles weight extraction, transformation, and binary serialization for
    different model architectures including standard transformers and MoE models.
    """
    
    def __init__(self, config):
        """Initialize processor with model configuration.
        
        Args:
            config: Dictionary containing model configuration including
                   architecture type, dimensions, expert counts, etc.
        """
        self.config = config
        self.archType = config['arch_type']
        self.currentModelIndex = None
        self.currentModel = None
        self.currentModelKeys = None
        self.layerMap = {}  # Maps layer names to file indices
        self.plan = []      # Ordered list of weights to extract
        
        # General debugging support
        self.debug = config.get('debug', False)
        if self.debug:
            arch_name = {
                ArchType.LLAMA: "Llama",
                ArchType.QWEN3: "Qwen3", 
                ArchType.QWEN3_MOE: "Qwen3MoE"
            }.get(self.archType, "Unknown")
            print(f"🔍 [DEBUG] Processor initialized for {arch_name} architecture")
            if self._is_moe_model():
                print(f"🔍 [DEBUG] MoE model: {config['n_experts']} experts, {config['n_active_experts']} active")

    def _is_moe_model(self):
        """Check if the current model is a Mixture of Experts model.
        
        Returns:
            bool: True if model has experts (n_experts > 0)
        """
        return self.config.get('n_experts', 0) > 0
        
    def __unloadModel(self):
        """Unload current model file and free memory.
        
        Performs garbage collection to ensure memory is freed properly.
        """
        if self.currentModel:
            del self.currentModel
            self.currentModel = None
            gc.collect()
        self.currentModelIndex = None

    def __loadModel(self, index: int):
        """Load a model file and cache its weight names.
        
        Args:
            index: Index of the file to load from config['files']
        """
        if (self.currentModelIndex == index):
            return
        self.__unloadModel()
        filePath = self.config['files'][index]
        fileName = os.path.basename(filePath)
        print(f'💿 Loading file {fileName}...')
        self.currentModel = safe_open(filePath, framework='pt', device='cpu')
        self.currentModelKeys = list(self.currentModel.keys())
        
        # Build layer mapping for efficient weight lookup
        for key in self.currentModelKeys:
            self.layerMap[key] = index
            
        print(f'Found {len(self.currentModelKeys)} layers')
        
        # Debug weight key detection
        if self.debug:
            print(f"🔍 [DEBUG] Loaded {len(self.currentModelKeys)} weight keys from {fileName}")
            if self._is_moe_model():
                # Look for MoE keys - Qwen3MoE uses mlp.gate.weight and mlp.experts.X patterns
                router_keys = [k for k in self.currentModelKeys if 'mlp.gate.weight' in k]
                expert_keys = [k for k in self.currentModelKeys if 'mlp.experts.' in k]
                if router_keys or expert_keys:
                    print(f"🔍 [DEBUG] Found MoE weights - Router: {len(router_keys)}, Expert: {len(expert_keys)}")
                    if router_keys:
                        print(f"🔍 [DEBUG] Sample router key: {router_keys[0]}")
                    if expert_keys:
                        print(f"🔍 [DEBUG] Sample expert key: {expert_keys[0]}")
                else:
                    print(f"🔍 [DEBUG] Warning: No MoE weights found in this file")
            # Show sample of weight keys for debugging
            sample_keys = self.currentModelKeys[:5]
            print(f"🔍 [DEBUG] Sample weight keys: {sample_keys}")
                
        self.currentModelIndex = index

    def __transformQ(self, tensor):
        if self.archType == ArchType.LLAMA:
            return permute(tensor, self.config['n_heads'], self.config['n_heads'])
        return tensor

    def __transformK(self, tensor):
        if self.archType == ArchType.LLAMA:
            return permute(tensor, self.config['n_heads'], self.config['n_kv_heads'])
        return tensor

    def __preparePlan(self):
        """Prepare the weight extraction plan based on model architecture.
        
        Creates an ordered list of weights to extract, handling different
        architectures and MoE vs dense models appropriately.
        """
        wt = self.config['weights_float_type']
        p = self.plan
        
        # Embedding layer (always F32 for numerical stability)
        p.append([FloatType.F32, 'model.embed_tokens.weight'])
        
        if self.debug:
            print(f"🔍 [DEBUG] Preparing weight extraction plan for {self.config['n_layers']} layers")
            if self._is_moe_model():
                print(f"🔍 [DEBUG] MoE model: will extract router + expert weights per layer")
            
        for l in range(0, self.config['n_layers']):
            # Attention projection weights
            p.append([wt, self.__transformQ,
                f'model.layers.{l}.self_attn.q_proj.weight'])
            p.append([wt, self.__transformK,
                f'model.layers.{l}.self_attn.k_proj.weight'])
            p.append([wt,
                f'model.layers.{l}.self_attn.v_proj.weight'])
            p.append([wt,
                f'model.layers.{l}.self_attn.o_proj.weight'])

            # MoE layers: Extract router + expert weights
            if (self.config['n_experts'] > 0):
                # Router/gating network (critical for expert selection)
                # Using F32 for router weights to maintain selection precision
                # Qwen3MoE uses model.layers.X.mlp.gate.weight for router
                p.append([FloatType.F32,
                    f'model.layers.{l}.mlp.gate.weight'])
                    
                if self.debug:
                    print(f"🔍 [DEBUG] Layer {l}: Adding router + {self.config['n_experts']} expert FFNs")
                    
                # Expert feed-forward weights 
                # Qwen3MoE uses model.layers.X.mlp.experts.Y.{gate_proj,up_proj,down_proj}.weight
                for e in range(self.config['n_experts']):
                    p.append([wt,
                        f'model.layers.{l}.mlp.experts.{e}.up_proj.weight']) # up
                    p.append([wt,
                        f'model.layers.{l}.mlp.experts.{e}.gate_proj.weight']) # gate 
                    p.append([wt,
                        f'model.layers.{l}.mlp.experts.{e}.down_proj.weight']) # down
            # Dense FFN layers
            else:
                p.append([wt,
                    f'model.layers.{l}.mlp.gate_proj.weight']) # gate
                p.append([wt,
                    f'model.layers.{l}.mlp.down_proj.weight']) # down
                p.append([wt,
                    f'model.layers.{l}.mlp.up_proj.weight']) # up

            # Qwen3 and Qwen3MoE both use Q/K normalization
            if (self.archType == ArchType.QWEN3 or self.archType == ArchType.QWEN3_MOE):
                p.append([FloatType.F32,
                    f'model.layers.{l}.self_attn.q_norm.weight'])
                p.append([FloatType.F32,
                    f'model.layers.{l}.self_attn.k_norm.weight'])

            p.append([FloatType.F32,
                f'model.layers.{l}.input_layernorm.weight'])
            p.append([FloatType.F32,
                f'model.layers.{l}.post_attention_layernorm.weight'])
        p.append([FloatType.F32,
            'model.norm.weight'])
        p.append([wt,
            'lm_head.weight', 'model.embed_tokens.weight'])

    def write(self, outputFile: str):
        self.__preparePlan()

        # Loading the last model file to get the layer names
        self.__loadModel(len(self.config['files']) - 1)
        self.__unloadModel()

        for planItem in self.plan:
            lookup = planItem[1:]
            transform = None
            if (callable(lookup[0])):
                transform = lookup[0]
                lookup = lookup[1:]

            if (self.currentModelIndex == None):
                modelIndex = 0
            else:
                modelIndex = None
                for layerName in lookup:
                    if (layerName in self.layerMap):
                        modelIndex = self.layerMap[layerName]
                        break
                if (modelIndex is None):
                    modelIndex = self.currentModelIndex + 1
            self.__loadModel(modelIndex)

            tensor = None
            for layerName in lookup:
                if (layerName in self.currentModelKeys):
                    tensor = self.currentModel.get_tensor(layerName)
                    break
            if tensor is None:
                raise Exception(f'Layer {lookup[0]} not found')
            print(f'🔶 Writing tensor {layerName} {tensor.shape}...')

            floatType = planItem[0]
            if (transform):
                tensor = transform(tensor)
            writeTensor(outputFile, tensor, floatType)

def parseArchType(type: str, has_experts: bool = False):
    """Parse model architecture type from config.
    
    Args:
        type: Model type string from config.json
        has_experts: Whether model has MoE experts (for validation)
        
    Returns:
        ArchType constant for the detected architecture
        
    Raises:
        Exception: If architecture type is not supported
    """
    archType = {
        'llama': ArchType.LLAMA,
        'mistral': ArchType.LLAMA,
        'qwen3': ArchType.QWEN3,
        'qwen3_moe': ArchType.QWEN3_MOE,  # Direct MoE architecture type
    }.get(type)
    
    if (archType is None):
        raise Exception(f'Unsupported arch type: {type}')
    return archType

def parseHiddenAct(act: str):
    hiddenAct = {
        'gelu': 0,
        'silu': 1
    }.get(act)
    if (hiddenAct is None):
        raise Exception(f'Unsupported hidden act: {act}')
    return hiddenAct

def parseRopeType(rt: str):
    ropeType = {
        'llama3': 2, # LLAMA3_1
    }.get(rt)
    if (ropeType is None):
        raise Exception(f'Unsupported rope type: {ropeType}')
    return ropeType

def parseRmsNormEpsilon(epsilon: float):
    if (epsilon == 1e-05):
        return 5
    elif (epsilon == 1e-06):
        return 6
    raise Exception(f'Unsupported epsilon: {epsilon}')

def loadConfig(folderPath: str, weightsFloatType: int, debug: bool = False):
    """Load and parse model configuration from HuggingFace format.
    
    Args:
        folderPath: Path to directory containing config.json and model files
        weightsFloatType: Target quantization type for weights
        debug: Enable debug output for MoE detection
        
    Returns:
        Dictionary containing parsed model configuration
        
    Raises:
        Exception: If config is invalid or model files not found
    """
    allFiles = os.listdir(folderPath)
    allFiles.sort()
    
    with open(os.path.join(folderPath, 'config.json')) as fc:
        config = json.load(fc)
    
    # Collect safetensor files
    files = []
    for fileName in allFiles:
        if fileName.endswith('.safetensors') and not fileName.startswith('.'):
            files.append(os.path.join(folderPath, fileName))
    if (len(files) == 0):
        raise Exception('Not found any model file')

    # Parse expert configuration first for architecture detection
    # Handle different field names used by different MoE models
    nExperts = config.get('num_local_experts') or config.get('num_experts')
    nActiveExperts = (
        config.get('num_active_local_experts') or 
        config.get('num_experts_per_tok') or
        config.get('num_experts_per_token')
    )
    num_experts = int(nExperts) if nExperts is not None else 0
    num_active_experts = int(nActiveExperts) if nActiveExperts is not None else 0
    
    has_experts = num_experts > 0
    
    if debug and has_experts:
        print(f"🔍 [DEBUG] Detected MoE model: {num_experts} experts, {num_active_experts} active")
        print(f"🔍 [DEBUG] Model type: {config['model_type']}")

    result = {
        'version': 0,
        'arch_type': parseArchType(config['model_type'], has_experts),
        'hidden_act': parseHiddenAct(config['hidden_act']),
        'dim': config['hidden_size'],
        'hidden_dim': config['intermediate_size'],
        'n_layers': config['num_hidden_layers'],
        'n_heads': config['num_attention_heads'],
        'n_kv_heads': config['num_key_value_heads'],
        'weights_float_type': weightsFloatType,
        'max_seq_len': config['max_position_embeddings'],
        'vocab_size': config['vocab_size'],
        'files': files,
        'debug': debug,  # Pass debug flag to processor
    }

    # Add MoE configuration
    result['n_experts'] = num_experts
    result['n_active_experts'] = num_active_experts

    ropeTheta = config.get('rope_theta')
    if (ropeTheta is not None):
        result['rope_theta'] = int(ropeTheta)

    ropeScaling = config.get('rope_scaling')
    if (ropeScaling is not None):
        result['rope_scaling_factor'] = int(ropeScaling['factor'])
        result['rope_scaling_low_freq_factor'] = int(ropeScaling['low_freq_factor'])
        result['rope_scaling_high_freq_factory'] = int(ropeScaling['high_freq_factor'])
        result['rope_scaling_orig_max_seq_len'] = int(ropeScaling['original_max_position_embeddings'])
        result['rope_type'] = parseRopeType(ropeScaling['rope_type'])

    headDim = config.get('head_dim')
    if (headDim is not None):
        result['head_dim'] = headDim

    rmsNormEps = config.get('rms_norm_eps')
    if (rmsNormEps is not None):
        result['norm_epsilon'] = parseRmsNormEpsilon(rmsNormEps)
    return result

def printUsage():
    """Print usage information for the converter script."""
    print('Usage: python convert-hf.py <sourceFolderPath> <weightsFloatType> <name> [--debug]')
    print()
    print('Options:')
    print('  <sourceFolderPath> The path to the folder containing the model files')
    print('  <weightsFloatType> The float type of the weights (e.g. "q40")')
    print('  <name>             The name of the model (e.g. "llama3")')
    print('  --debug            Enable verbose debug output for conversion process')
    print()
    print('Performance notes:')
    print('  - Large models may take several hours to convert')
    print('  - Use F32 for better quality but larger files')
    print('  - Use Q40 for smaller files with minimal quality loss')
    print()
    print('Supported architectures:')
    print('  - Llama (standard transformer)')
    print('  - Qwen3 (dense model with Q/K normalization)')
    print('  - Qwen3MoE (Mixture of Experts model)')

if __name__ == '__main__':
    """Main conversion script entry point."""
    
    # Handle help flag
    if '--help' in sys.argv or '-h' in sys.argv:
        printUsage()
        exit(0)
        
    if (len(sys.argv) < 4):
        printUsage()
        exit(1)

    sourceFolderPath = sys.argv[1]
    weightsFloatType = parseFloatType(sys.argv[2])
    name = sys.argv[3]
    
    # Check for debug flag
    debug = '--debug' in sys.argv
    
    outputFileName = f'dllama_model_{name}_{sys.argv[2]}.m'

    print(f'🚀 Starting conversion...')
    print(f'📁 Source: {sourceFolderPath}')
    print(f'⚖️  Weight type: {sys.argv[2]}')
    print(f'📄 Output file: {outputFileName}')
    if debug:
        print(f'🔍 Debug mode: enabled')
    print()

    try:
        config = loadConfig(sourceFolderPath, weightsFloatType, debug)
        
        # Print architecture information
        arch_name = {
            ArchType.LLAMA: "Llama",
            ArchType.QWEN3: "Qwen3", 
            ArchType.QWEN3_MOE: "Qwen3MoE"
        }.get(config['arch_type'], "Unknown")
        
        print(f'🏗️  Architecture: {arch_name}')
        if config['n_experts'] > 0:
            print(f'👥 MoE: {config["n_experts"]} experts, {config["n_active_experts"]} active per token')
        print()

        with open(outputFileName, 'wb') as outputFile:
            writeHeader(outputFile, config)
            processor = Processor(config)
            processor.write(outputFile)

        print(f'✅ {outputFileName} created successfully')
        
    except Exception as e:
        print(f'❌ Conversion failed: {e}')
        if debug:
            import traceback
            traceback.print_exc()
        exit(1)