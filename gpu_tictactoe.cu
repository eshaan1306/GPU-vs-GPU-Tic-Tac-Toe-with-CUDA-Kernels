#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

constexpr int kBoardSize = 9;

__device__ __constant__ int kWinLines[8][3] = {
    {0, 1, 2}, {3, 4, 5}, {6, 7, 8},
    {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
    {0, 4, 8}, {2, 4, 6}
};

__device__ int DeviceCheckWin(const int* board) {
    for (int line = 0; line < 8; ++line) {
        const int a = kWinLines[line][0];
        const int b = kWinLines[line][1];
        const int c = kWinLines[line][2];

        const int sum = board[a] + board[b] + board[c];

        if (sum == 3) {
            return 1;
        }

        if (sum == -3) {
            return -1;
        }
    }

    return 0;
}

__device__ int ScoreMove(const int* board, int move_index, int player) {
    if (board[move_index] != 0) {
        return -1000;
    }

    int temp[kBoardSize];

    #pragma unroll
    for (int i = 0; i < kBoardSize; ++i) {
        temp[i] = board[i];
    }

    // Highest priority: make an immediate winning move.
    temp[move_index] = player;

    if (DeviceCheckWin(temp) == player) {
        return 100;
    }

    // Second priority: block an immediate opponent win.
    const int opponent = -player;

    #pragma unroll
    for (int i = 0; i < kBoardSize; ++i) {
        temp[i] = board[i];
    }

    temp[move_index] = opponent;

    if (DeviceCheckWin(temp) == opponent) {
        return 80;
    }

    // Positional preference.
    if (move_index == 4) {
        return 10;
    }

    if (move_index == 0 || move_index == 2 ||
        move_index == 6 || move_index == 8) {
        return 5;
    }

    return 3;
}

__global__ void ChooseMovePlayerA(
    const int* board,
    int player,
    int* scores) {

    const int index = threadIdx.x;

    if (index >= kBoardSize) {
        return;
    }

    scores[index] = ScoreMove(board, index, player);
}

__global__ void ChooseMovePlayerB(
    const int* board,
    int player,
    int* scores) {

    const int index = threadIdx.x;

    if (index >= kBoardSize) {
        return;
    }

    int score = ScoreMove(board, index, player);

    if (score <= -1000) {
        scores[index] = score;
        return;
    }

    // Player B uses a slightly different positional strategy.
    if (index == 0 || index == 2 ||
        index == 6 || index == 8) {
        score -= 2;
    } else if (index != 4) {
        score += 4;
    }

    scores[index] = score;
}

void CheckCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(
            stderr,
            "CUDA error during %s: %s\n",
            operation,
            cudaGetErrorString(error));

        std::exit(EXIT_FAILURE);
    }
}

void PrintBoard(const int* board) {
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            const int value = board[row * 3 + column];

            char symbol = '.';

            if (value == 1) {
                symbol = 'X';
            } else if (value == -1) {
                symbol = 'O';
            }

            std::printf("%c ", symbol);
        }

        std::printf("\n");
    }

    std::printf("\n");
}

int HostCheckWin(const int* board) {
    static const int kHostWinLines[8][3] = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8},
        {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
        {0, 4, 8}, {2, 4, 6}
    };

    for (int line = 0; line < 8; ++line) {
        const int a = kHostWinLines[line][0];
        const int b = kHostWinLines[line][1];
        const int c = kHostWinLines[line][2];

        const int sum = board[a] + board[b] + board[c];

        if (sum == 3) {
            return 1;
        }

        if (sum == -3) {
            return -1;
        }
    }

    return 0;
}

bool BoardFull(const int* board) {
    for (int i = 0; i < kBoardSize; ++i) {
        if (board[i] == 0) {
            return false;
        }
    }

    return true;
}

int PlayGame(int game_number, int* device_board, int* device_scores) {
    int board[kBoardSize] = {0};

    int current_player = 1;

    std::printf("\n============================\n");
    std::printf("Game %d\n", game_number);
    std::printf("============================\n\n");

    while (true) {
        std::printf("Current board:\n");
        PrintBoard(board);

        CheckCuda(
            cudaMemcpy(
                device_board,
                board,
                kBoardSize * sizeof(int),
                cudaMemcpyHostToDevice),
            "copying board to GPU");

        if (current_player == 1) {
            std::printf("GPU Player A (X) thinking...\n");

            ChooseMovePlayerA<<<1, kBoardSize>>>(
                device_board,
                current_player,
                device_scores);
        } else {
            std::printf("GPU Player B (O) thinking...\n");

            ChooseMovePlayerB<<<1, kBoardSize>>>(
                device_board,
                current_player,
                device_scores);
        }

        CheckCuda(
            cudaGetLastError(),
            "launching CUDA kernel");

        CheckCuda(
            cudaDeviceSynchronize(),
            "synchronizing CUDA device");

        int scores[kBoardSize];

        CheckCuda(
            cudaMemcpy(
                scores,
                device_scores,
                kBoardSize * sizeof(int),
                cudaMemcpyDeviceToHost),
            "copying move scores from GPU");

        int best_index = -1;
        int best_score = -100000;

        for (int i = 0; i < kBoardSize; ++i) {
            if (board[i] == 0 && scores[i] > best_score) {
                best_score = scores[i];
                best_index = i;
            }
        }

        if (best_index == -1) {
            std::printf("No legal move found.\n");
            return 0;
        }

        board[best_index] = current_player;

        std::printf(
            "GPU Player %c selected position %d with score %d\n\n",
            current_player == 1 ? 'X' : 'O',
            best_index,
            best_score);

        const int winner = HostCheckWin(board);

        if (winner == 1) {
            PrintBoard(board);
            std::printf("GPU Player A (X) wins!\n");
            return 1;
        }

        if (winner == -1) {
            PrintBoard(board);
            std::printf("GPU Player B (O) wins!\n");
            return -1;
        }

        if (BoardFull(board)) {
            PrintBoard(board);
            std::printf("Game is a draw.\n");
            return 0;
        }

        current_player = -current_player;
    }
}

void PrintUsage(const char* program_name) {
    std::printf(
        "Usage: %s [-g number_of_games]\n",
        program_name);

    std::printf(
        "Example: %s -g 5\n",
        program_name);
}

int main(int argc, char** argv) {
    int number_of_games = 1;

    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-g") == 0 ||
             std::strcmp(argv[i], "--games") == 0) &&
            i + 1 < argc) {

            number_of_games = std::atoi(argv[++i]);

            if (number_of_games <= 0) {
                std::fprintf(
                    stderr,
                    "Number of games must be greater than zero.\n");

                return EXIT_FAILURE;
            }
        } else if (
            std::strcmp(argv[i], "-h") == 0 ||
            std::strcmp(argv[i], "--help") == 0) {

            PrintUsage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            std::fprintf(
                stderr,
                "Unknown command-line argument: %s\n",
                argv[i]);

            PrintUsage(argv[0]);

            return EXIT_FAILURE;
        }
    }

    int* device_board = nullptr;
    int* device_scores = nullptr;

    CheckCuda(
        cudaMalloc(
            &device_board,
            kBoardSize * sizeof(int)),
        "allocating GPU board memory");

    CheckCuda(
        cudaMalloc(
            &device_scores,
            kBoardSize * sizeof(int)),
        "allocating GPU score memory");

    std::printf("GPU vs GPU Tic-Tac-Toe using CUDA\n");
    std::printf("Games requested: %d\n", number_of_games);

    int player_a_wins = 0;
    int player_b_wins = 0;
    int draws = 0;

    for (int game = 1; game <= number_of_games; ++game) {
        const int result =
            PlayGame(game, device_board, device_scores);

        if (result == 1) {
            ++player_a_wins;
        } else if (result == -1) {
            ++player_b_wins;
        } else {
            ++draws;
        }
    }

    std::printf("\n============================\n");
    std::printf("Final Results\n");
    std::printf("============================\n");
    std::printf("Games: %d\n", number_of_games);
    std::printf("GPU Player A wins: %d\n", player_a_wins);
    std::printf("GPU Player B wins: %d\n", player_b_wins);
    std::printf("Draws: %d\n", draws);

    CheckCuda(
        cudaFree(device_board),
        "freeing GPU board memory");

    CheckCuda(
        cudaFree(device_scores),
        "freeing GPU score memory");

    CheckCuda(
        cudaDeviceReset(),
        "resetting CUDA device");

    return EXIT_SUCCESS;
}
