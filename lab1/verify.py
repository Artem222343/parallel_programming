import numpy as np
import subprocess
import sys
import os

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

def run_verification(N, exe_path):
    print(f"--- Тестирование для N = {N} ---")
    fileA = "A.txt"
    fileB = "B.txt"
    fileC = "C_out.txt"

    print("Генерация матриц...")
    matA = generate_matrix(fileA, N)
    matB = generate_matrix(fileB, N)

    print("Запуск C++ программы...")
    result = subprocess.run([exe_path, fileA, fileB, fileC], capture_output=True, text=True, encoding='utf-8')
    print(result.stdout)

    print("Проверка результатов с помощью Numpy...")
    matC_python = np.dot(matA, matB)
    
    matC_cpp = read_matrix(fileC)

    is_correct = np.allclose(matC_python, matC_cpp, atol=1e-5)

    if is_correct:
        print("Верификация пройдена: Результат C++ совпадает с Python/Numpy.\n")
    else:
        print("Ошибка: Результаты не совпадают!\n")

    os.remove(fileA)
    os.remove(fileB)
    os.remove(fileC)

if __name__ == "__main__":
    cpp_executable = "./matrix" if os.name != 'nt' else "matrix.exe"
    
    if not os.path.exists(cpp_executable):
        print(f"Ошибка: Исполняемый файл '{cpp_executable}' не найден. Скомпилируйте C++ код сначала.")
        sys.exit(1)

    sizes_to_test = [100, 250, 500, 750, 1000]
    
    for size in sizes_to_test:
        run_verification(size, cpp_executable)
