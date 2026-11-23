NVCC := nvcc
TARGET := gpu_tictactoe
SOURCE := gpu_tictactoe.cu

NVCCFLAGS := -std=c++17 -O2

GAMES ?= 1

.PHONY: all build run clean

all: build

build: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) $(SOURCE) -o $(TARGET)

run: $(TARGET)
	./$(TARGET) -g $(GAMES)

clean:
	rm -f $(TARGET)
