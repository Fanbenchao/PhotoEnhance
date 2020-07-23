TF_CFLAGS ?= `python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'`
TF_LFLAGS ?= `python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'`

# TF_INC ?= /usr/local/lib/python2.7/dist-packages/tensorflow/include
CUDA_HOME ?= /usr/local/cuda/lib64 # Replace with your cuda home

SRC_DIR = ops

BUILD_DIR = build
LIB_DIR = lib

CC = c++ -std=c++11
NVCC = nvcc -std c++11
CFLAGS = -fPIC -O2 $(TF_CFLAGS)
LDFLAGS = -L$(CUDA_HOME) -lcudart $(TF_LFLAGS)
NVFLAGS = -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $(TF_CFLAGS) \
					-expt-relaxed-constexpr -Wno-deprecated-gpu-targets -ftz=true


SRC = bilateral_slice.cc
CUDA_SRC = bilateral_slice.cu.cc
CUDA_OBJ = $(addprefix $(BUILD_DIR)/,$(CUDA_SRC:.cc=.o))
SRCS = $(addprefix $(SRC_DIR)/, $(SRC))

all: $(LIB_DIR)/hdrnet_ops.so

# Main library
$(LIB_DIR)/hdrnet_ops.so: $(CUDA_OBJ) $(LIB_DIR) $(SRCS)
	$(CC) -shared -o $@ $(SRCS) $(CUDA_OBJ) $(CFLAGS) $(LDFLAGS) 

# Cuda kernels
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc $(BUILD_DIR)
	$(NVCC) -c  $< -o $@ $(NVFLAGS)

$(BUILD_DIR):
	mkdir -p $@


$(LIB_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)
