ARCH ?= sm_80
BUILD ?= release
NVCC := nvcc
INC := -Iinclude
OUT := bin
BIN := $(OUT)/matmul

COMMON := -std=c++17 $(INC) -Xcompiler -fPIC
ifeq ($(BUILD),debug)
  CFLAGS := -g -G -O0
else
  CFLAGS := -O3
endif

MATRIX_TXT := data/matrix_data.txt
MATRIX_HEADER := include/matrix_data.h

NVFLAGS := -arch=$(ARCH) $(COMMON) $(CFLAGS) -Xptxas -v \
            -DMATRIX_FILE=\"matrix_data.h\"

SRC := src/kernels.cu src/main.cu
OBJ := $(patsubst src/%.cpp, build/%.o, $(filter %.cpp,$(SRC))) \
       $(patsubst src/%.cu,  build/%.o, $(filter %.cu,$(SRC)))

.PHONY: all clean run

all: $(BIN)

$(MATRIX_HEADER): $(MATRIX_TXT) tools/gen_matrix_header.py
	@echo "🧮 Generating $@ ..."
	@python3 tools/gen_matrix_header.py $(MATRIX_TXT) $(MATRIX_HEADER)

$(OUT):
	@mkdir -p $(OUT)
build:
	@mkdir -p build

build/%.o: src/%.cpp | build $(MATRIX_HEADER)
	@$(NVCC) $(NVFLAGS) -c $< -o $@

build/%.o: src/%.cu | build $(MATRIX_HEADER)
	@$(NVCC) $(NVFLAGS) -c $< -o $@

$(BIN): $(OUT) $(OBJ)
	@$(NVCC) $(NVFLAGS) $(OBJ) -o $(BIN)
	@echo "✅ Build complete: $(BIN)"

run: all
	@./$(BIN)

clean:
	@rm -rf build $(OUT)
	@echo "🧹 Clean done."
