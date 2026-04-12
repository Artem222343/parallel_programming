#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>

using namespace std;

// CUDA Ядро: выполняется на видеокарте. Каждый поток считает 1 элемент матрицы C
__global__ void matrixMulKernel(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

bool readMatrix(const string& filename, vector<double>& matrix, int& N) {
    ifstream file(filename);
    if (!file.is_open()) return false;
    file >> N;
    matrix.resize(N * N);
    for (int i = 0; i < N * N; ++i) {
        file >> matrix[i];
    }
    return true;
}

void writeMatrix(const string& filename, const vector<double>& matrix, int N) {
    ofstream file(filename);
    file << N << "\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            file << matrix[i * N + j] << " ";
        }
        file << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 5) {
        cerr << "Использование: " << argv[0] << " <matA.txt> <matB.txt> <matC_out.txt> [block_size]\n";
        return 1;
    }

    string fileA = argv[1];
    string fileB = argv[2];
    string fileC = argv[3];

    // Размер блока потоков по умолчанию 16x16
    int block_size = (argc == 5) ? atoi(argv[4]) : 16;

    vector<double> h_A, h_B, h_C;
    int N_A, N_B;

    if (!readMatrix(fileA, h_A, N_A) || !readMatrix(fileB, h_B, N_B) || N_A != N_B) {
        cerr << "Ошибка чтения файлов или несовпадение размеров!\n";
        return 1;
    }

    int N = N_A;
    h_C.assign(N * N, 0.0);
    size_t bytes = N * N * sizeof(double);

    // Указатели на память устройства (видеокарты)
    double *d_A, *d_B, *d_C;

    // Выделение памяти на видеокарте
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Настройка замера времени CUDA (для точного измерения без учета I/O файлов)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Копирование данных из ОЗУ в видеопамять (Host -> Device)
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    // Конфигурация сетки (Grid) и блоков (Block)
    dim3 threadsPerBlock(block_size, block_size);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Запуск ядра на видеокарте
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    // Синхронизация для ожидания завершения GPU
    cudaDeviceSynchronize();

    // Копирование результата обратно (Device -> Host)
    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Очистка видеопамяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    writeMatrix(fileC, h_C, N);

    cout << "Объем задачи (N): " << N << "\n";
    cout << "Размер блока: " << block_size << "x" << block_size << "\n";
    cout << "Время выполнения: " << milliseconds / 1000.0 << " сек\n";

    return 0;
}