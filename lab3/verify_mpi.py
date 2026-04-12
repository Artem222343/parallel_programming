import numpy as np
import subprocess
import sys
import os
import re

def generate_matrix(filename, N):
    mat = np.random.rand(N, N)
    with open(filename, 'w') as f:
        f.write(f"{N}\n")
        np.savetxt(f, mat, fmt='%.6f')
    return mat

def read_matrix(filename):
    with open(filename, 'r') as f:
        N = int(f.readline().strip())
        mat = np.loadtxt(f)
    return mat

def run_experiment(N, procs, exe_path, verify=False):
    fileA = "A.txt"
    fileB = "B.txt"
    fileC = "C_out.txt"

    matA = generate_matrix(fileA, N)
    matB = generate_matrix(fileB, N)

    cmd = ["mpiexec", "-n", str(procs), exe_path, fileA, fileB, fileC]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    
    time_match = re.search(r"Время выполнения:\s+([0-9.]+)", result.stdout)
    exec_time = float(time_match.group(1)) if time_match else -1

    if verify:
        matC_python = np.dot(matA, matB)
        matC_cpp = read_matrix(fileC)
        is_correct = np.allclose(matC_python, matC_cpp, atol=1e-5)
        if not is_correct:
            print(f"ОШИБКА ВЕРИФИКАЦИИ на N={N}, procs={procs}!")
            sys.exit(1)

    # Очистка файлов, игнорируя ошибки, если файл не создался
    for f in [fileA, fileB, fileC]:
        if os.path.exists(f): os.remove(f)
    
    return exec_time

if __name__ == "__main__":
    exe_name = "matrix_mpi" if os.name != 'nt' else "matrix_mpi.exe"
    cpp_executable = os.path.join(".", exe_name)
    
    if not os.path.exists(cpp_executable):
        print(f"Ошибка: Исполняемый файл '{cpp_executable}' не найден.")
        print("Скомпилируйте: mpicxx -O3 matrix_mpi.cpp -o matrix_mpi")
        sys.exit(1)

    sizes_to_test = [200, 400, 800, 1200, 1600, 2000]
    procs_to_test = [1, 2, 4, 6, 8, 12]

    print("Проведение быстрой верификации для N=400 (MPI procs=4)...")
    run_experiment(400, 4, cpp_executable, verify=True)
    print("Верификация успешно пройдена. Начинаем бенчмарк...\n")

    header = f"| N \\ Процессы | " + " | ".join([str(p) for p in procs_to_test]) + " |"
    print(header)
    print("|---" + "|---" * len(procs_to_test) + "|")

    for size in sizes_to_test:
        row_str = f"| {size} |"
        for procs in procs_to_test:
            exec_time = run_experiment(size, procs, cpp_executable, verify=False)
            row_str += f" {exec_time:.5f} |"
        print(row_str)