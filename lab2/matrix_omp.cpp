#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <cstdlib>

using namespace std;

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
    // Добавлен 4-й опциональный аргумент для потоков
    if (argc < 4 || argc > 5) {
        cerr << "Использование: " << argv[0] << " <matA.txt> <matB.txt> <matC_out.txt> [num_threads]\n";
        return 1;
    }

    string fileA = argv[1];
    string fileB = argv[2];
    string fileC = argv[3];

    int num_threads = omp_get_max_threads();
    if (argc == 5) {
        num_threads = atoi(argv[4]);
        omp_set_num_threads(num_threads);
    }

    vector<double> A, B, C;
    int N_A, N_B;

    if (!readMatrix(fileA, A, N_A) || !readMatrix(fileB, B, N_B)) {
        cerr << "Ошибка чтения файлов!\n";
        return 1;
    }

    if (N_A != N_B) {
        cerr << "Размеры матриц не совпадают!\n";
        return 1;
    }

    int N = N_A;
    C.assign(N * N, 0.0);

    auto start_time = chrono::high_resolution_clock::now();

    // Распараллеливаем внешний цикл по строкам матрицы
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            double r = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += r * B[k * N + j];
            }
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    writeMatrix(fileC, C, N);

    cout << "Объем задачи (N): " << N << "\n";
    cout << "Потоков: " << num_threads << "\n";
    cout << "Время выполнения: " << elapsed.count() << " сек\n";

    return 0;
}