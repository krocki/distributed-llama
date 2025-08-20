CXX = g++
CXXFLAGS = -g -std=c++11 -Werror -Wformat -Werror=format-security 

ifndef TERMUX_VERSION
	CXXFLAGS += -march=native -mtune=native
endif

#ifdef DEBUG
	CXXFLAGS += -g -fsanitize=address
#else
#	CXXFLAGS += -O3
#endif

ifdef WVLA
	CXXFLAGS += -Wvla-extension
endif

ifdef DLLAMA_VULKAN
	CGLSLC = glslc

ifeq ($(OS),Windows_NT)
	LIBS += -L$(VK_SDK_PATH)\lib -lvulkan-1
	CXXFLAGS += -DDLLAMA_VULKAN -I$(VK_SDK_PATH)\include
else
	LIBS += -lvulkan
	CXXFLAGS += -DDLLAMA_VULKAN
endif

	DEPS += nn-vulkan.o
endif

ifeq ($(OS),Windows_NT)
    LIBS += -lws2_32
	DELETE_CMD = del /f
else
    LIBS += -lpthread
    DELETE_CMD = rm -fv
endif

.PHONY: clean dllama

clean:
	$(DELETE_CMD) *.o dllama dllama-* socket-benchmark mmap-buffer-* *-test *.exe

# nn
nn-quants.o: src/nn/nn-quants.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-core.o: src/nn/nn-core.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-executor.o: src/nn/nn-executor.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-network.o: src/nn/nn-network.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
llamafile-sgemm.o: src/nn/llamafile/sgemm.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-cpu-ops.o: src/nn/nn-cpu-ops.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-cpu.o: src/nn/nn-cpu.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-cpu-test: src/nn/nn-cpu-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-cpu-ops-test: src/nn/nn-cpu-ops-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-moe-test: src/nn/nn-moe-test.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o llm.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-simple-test: src/nn/nn-simple-test.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-moe-simple-test: src/nn/nn-moe-simple-test.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o llm.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-moe-minimal-test: src/nn/nn-moe-minimal-test.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-basic-test: src/nn/nn-basic-test.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o llm.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-minimal-crash: src/nn/nn-minimal-crash.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-moe-pipeline-test: src/nn/nn-moe-pipeline-test.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o llm.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-router-ffn-test: src/nn/nn-router-ffn-test.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o llm.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-incremental-moe-test: src/nn/nn-incremental-moe-test.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o llm.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-manual-ffn-test: src/nn/nn-manual-ffn-test.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-working-integration: src/nn/nn-working-integration.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o llm.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-simple-router-ffn: src/nn/nn-simple-router-ffn.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-moe-final-test: src/nn/nn-moe-final-test.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o llm.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-manual-moe-working: src/nn/nn-manual-moe-working.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o llm.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-test-buildmoe: src/nn/nn-test-buildmoe.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o llm.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-expert-selection-test: src/nn/nn-expert-selection-test.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o llm.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-vulkan.o: src/nn/nn-vulkan.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

ifdef DLLAMA_VULKAN
VULKAN_SHADER_SRCS := $(wildcard src/nn/vulkan/*.comp)
VULKAN_SHADER_BINS := $(VULKAN_SHADER_SRCS:.comp=.spv)
DEPS += $(VULKAN_SHADER_BINS)

%.spv: %.comp
	$(CGLSLC) -c $< -o $@ --target-env=vulkan1.2
nn-vulkan-test: src/nn/nn-vulkan-test.cpp nn-quants.o nn-core.o nn-executor.o nn-vulkan.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv, $^) -o $@ $(LIBS)
endif

# llm
tokenizer.o: src/tokenizer.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
llm.o: src/llm.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
app.o: src/app.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
tokenizer-test: src/tokenizer-test.cpp nn-quants.o nn-core.o llamafile-sgemm.o nn-cpu-ops.o tokenizer.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
dllama: src/dllama.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o tokenizer.o llm.o app.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv, $^) -o $@ $(LIBS)
dllama-api: src/dllama-api.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o tokenizer.o llm.o app.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv, $^) -o $@ $(LIBS)
