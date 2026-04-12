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

def run_experiment(N, threads, exe_path, verify=False):
    fileA = "A.txt"
    fileB = "B.txt"
    fileC = "C_out.txt"

    matA = generate_matrix(fileA, N)
    matB = generate_matrix(fileB, N)

    result = subprocess.run([exe_path, fileA, fileB, fileC, str(threads)], capture_output=True, text=True, encoding='utf-8')
    
    # Парсим время выполнения из вывода C++
    time_match = re.search(r"Время выполнения:\s+([0-9.]+)", result.stdout)
    exec_time = float(time_match.group(1)) if time_match else -1

    if verify:
        matC_python = np.dot(matA, matB)
        matC_cpp = read_matrix(fileC)
        is_correct = np.allclose(matC_python, matC_cpp, atol=1e-5)
        if not is_correct:
            print(f"ОШИБКА ВЕРИФИКАЦИИ на N={N}, threads={threads}!")
            sys.exit(1)

    os.remove(fileA)
    os.remove(fileB)
    os.remove(fileC)
    
    return exec_time

if __name__ == "__main__":
    cpp_executable = "./matrix_omp" if os.name != 'nt' else "matrix_omp.exe"
    
    if not os.path.exists(cpp_executable):
        print(f"Ошибка: Исполняемый файл '{cpp_executable}' не найден.")
        print("Скомпилируйте: g++ -O3 -fopenmp matrix_omp.cpp -o matrix_omp")
        sys.exit(1)

    sizes_to_test = [200, 400, 800, 1200, 1600, 2000]
    threads_to_test = [1, 2, 4, 6, 8, 12]
    
    results = {n: {} for n in sizes_to_test}

    print("Проведение быстрой верификации для N=400...")
    run_experiment(400, 4, cpp_executable, verify=True)
    print("Верификация успешно пройдена. Начинаем бенчмарк...\n")

    # Печать заголовка таблицы
    header = f"| N \ Потоки | " + " | ".join([str(t) for t in threads_to_test]) + " |"
    print(header)
    print("|---" + "|---" * len(threads_to_test) + "|")

    for size in sizes_to_test:
        row_str = f"| {size} |"
        for threads in threads_to_test:
            exec_time = run_experiment(size, threads, cpp_executable, verify=False)
            results[size][threads] = exec_time
            row_str += f" {exec_time:.5f} |"
        print(row_str)