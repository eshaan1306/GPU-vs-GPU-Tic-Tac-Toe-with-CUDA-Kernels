# GPU vs GPU Tic-Tac-Toe with CUDA Kernels

This project implements a GPU-based Tic-Tac-Toe simulation using NVIDIA CUDA.

Two computer-controlled players evaluate possible Tic-Tac-Toe moves using CUDA kernels. The CPU maintains the board and controls the game loop, while the GPU evaluates legal moves and assigns scores to them in parallel.

## GPU Computation

The project uses two CUDA kernels:

- `ChooseMovePlayerA`
- `ChooseMovePlayerB`

For each turn, the current board is copied from host memory to GPU memory using `cudaMemcpy`.

A CUDA kernel is then launched with one GPU thread for each of the nine board positions. Each thread independently evaluates one possible move and produces a score.

The scores are copied back to the CPU, where the highest-scoring legal move is selected.

The project therefore performs the move-evaluation portion of the game using GPU computation rather than CPU-only evaluation.

CUDA features used include:

- CUDA kernels
- GPU threads
- `cudaMalloc`
- host-to-device memory transfer
- device-to-host memory transfer
- `cudaDeviceSynchronize`
- CUDA constant memory

## Player Strategies

GPU Player A uses the following priorities:

1. Make an immediate winning move.
2. Block an immediate opponent win.
3. Prefer the center.
4. Prefer corners.
5. Prefer edges.

GPU Player B uses the same basic win/block logic but applies a different positional preference so that the two GPU players can make different decisions.

## Requirements

The project requires:

- NVIDIA CUDA-capable GPU
- NVIDIA CUDA Toolkit
- `nvcc`
- GNU Make

No external C++ libraries are required.

## Build

Compile the project with:

```bash
make
```

or:

```bash
make build
```

This produces:

```text
gpu_tictactoe
```

## Run

Run one game:

```bash
./gpu_tictactoe
```

or:

```bash
make run
```

Run multiple games using the command-line argument:

```bash
./gpu_tictactoe -g 5
```

or:

```bash
make run GAMES=5
```

The `-g` / `--games` option determines how many GPU-vs-GPU games are executed.

Display command-line help with:

```bash
./gpu_tictactoe --help
```

## Program Flow

For each turn:

1. The CPU stores the current 3x3 board.
2. The board is copied to GPU memory.
3. A CUDA kernel launches nine GPU threads.
4. Each thread evaluates one board position.
5. The GPU writes a score for every possible move.
6. Scores are transferred back to the CPU.
7. The CPU selects the best legal move.
8. The board is updated and checked for a win or draw.

After all requested games finish, the program prints the number of wins for each GPU player and the number of draws.

## Example

```bash
make clean
make
./gpu_tictactoe -g 3
```

Example output includes the board state, GPU player selections, move scores, game result, and final summary.

## Execution Environment

The project can be compiled and executed on any Linux environment with an NVIDIA CUDA-capable GPU and the CUDA Toolkit installed.

For the course submission, the program is executed using an NVIDIA T4 GPU through Google Colab because the development machine does not contain a CUDA-capable GPU.

## What I Learned

This project demonstrates how CUDA kernels can be used for parallel decision evaluation rather than only traditional numerical workloads.

The project provided practical experience with CUDA kernel launches, device memory allocation, host/device memory transfers, synchronization, constant memory, GPU thread indexing, and integrating GPU computation into a CPU-controlled application.
