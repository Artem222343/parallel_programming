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

def run_experiment(N, block_size, exe_path, verify=False):
    fileA = "A.txt"
    fileB = "B.txt"
    fileC = "C_out.txt"

    matA = generate_matrix(fileA, N)
    matB = generate_matrix(fileB, N)

    result = subprocess.run([exe_path, fileA, fileB, fileC, str(block_size)], capture_output=True, text=True, encoding='utf-8')
    
    time_match = re.search(r"Время выполнения:\s+([0-9.]+)", result.stdout)
    exec_time = float(time_match.group(1)) if time_match else -1

    if verify:
        matC_python = np.dot(matA, matB)
        matC_cpp = read_matrix(fileC)
        is_correct = np.allclose(matC_python, matC_cpp, atol=1e-5)
        if not is_correct:
            print(f"ОШИБКА ВЕРИФИКАЦИИ на N={N}, block={block_size}x{block_size}!")
            sys.exit(1)

    for f in [fileA, fileB, fileC]:
        if os.path.exists(f): os.remove(f)
    
    return exec_time

if __name__ == "__main__":
    exe_name = "matrix_cuda" if os.name != 'nt' else "matrix_cuda.exe"
    cuda_executable = os.path.join(".", exe_name)
    
    if not os.path.exists(cuda_executable):
        print(f"Ошибка: Исполняемый файл '{cuda_executable}' не найден.")
        print("Скомпилируйте: nvcc matrix_cuda.cu -o matrix_cuda")
        sys.exit(1)

    sizes_to_test = [200, 400, 800, 1200, 1600, 2000]
    blocks_to_test = [8, 16, 32]

    print("Проведение быстрой верификации для N=400 (Block 16x16)...")
    run_experiment(400, 16, cuda_executable, verify=True)
    print("Верификация успешно пройдена. Начинаем бенчмарк...\n")

    header = f"| N \\ Размер блока (2D) | " + " | ".join([f"{b}x{b}" for b in blocks_to_test]) + " |"
    print(header)
    print("|---" + "|---" * len(blocks_to_test) + "|")

    for size in sizes_to_test:
        row_str = f"| {size} |"
        for b_size in blocks_to_test:
            exec_time = run_experiment(size, b_size, cuda_executable, verify=False)
            row_str += f" {exec_time:.5f} |"
        print(row_str)