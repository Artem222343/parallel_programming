#include <iostream>
#include <vector>
#include <fstream>
#include <mpi.h>

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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) cerr << "Использование: mpiexec -n <cores> " << argv[0] << " <matA.txt> <matB.txt> <matC_out.txt>\n";
        MPI_Finalize();
        return 1;
    }

    string fileA = argv[1];
    string fileB = argv[2];
    string fileC = argv[3];

    int N = 0;
    vector<double> A, B, C;

    if (rank == 0) {
        int N_A, N_B;
        if (!readMatrix(fileA, A, N_A) || !readMatrix(fileB, B, N_B) || N_A != N_B) {
            cerr << "Ошибка чтения файлов или несовпадение размеров!\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        N = N_A;
        C.resize(N * N, 0.0);
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        B.resize(N * N);
    }
    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<int> sendcounts(size), displs(size);
    int offset = 0;
    for (int i = 0; i < size; ++i) {
        int rows = N / size + (i < (N % size) ? 1 : 0); 
        sendcounts[i] = rows * N;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int local_rows = sendcounts[rank] / N;
    vector<double> local_A(local_rows * N);
    vector<double> local_C(local_rows * N, 0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = 0.0;
    if (rank == 0) start_time = MPI_Wtime();

    MPI_Scatterv(rank == 0 ? A.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_A.data(), local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_rows; ++i) {
        for (int k = 0; k < N; ++k) {
            double r = local_A[i * N + k];
            for (int j = 0; j < N; ++j) {
                local_C[i * N + j] += r * B[k * N + j];
            }
        }
    }

    MPI_Gatherv(local_C.data(), local_rows * N, MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double elapsed = MPI_Wtime() - start_time;
        writeMatrix(fileC, C, N);
        cout << "Объем задачи (N): " << N << "\n";
        cout << "Процессов MPI: " << size << "\n";
        cout << "Время выполнения: " << elapsed << " сек\n";
    }

    MPI_Finalize();
    return 0;
}