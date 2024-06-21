import numpy as np
import scipy.linalg
import time

def generate_matrix_and_vector(n):
    A = np.random.rand(n, n)
    b = np.random.rand(n)
    return A, b

def lu_decomposition(A, b):
    start = time.time()
    P, L, U = scipy.linalg.lu(A)
    x = scipy.linalg.solve_triangular(U, scipy.linalg.solve_triangular(L, P.T @ b, lower=True))
    end = time.time()
    return x, end - start

def matrix_inversion(A, b):
    start = time.time()
    A_inv = np.linalg.inv(A)
    x = A_inv @ b
    end = time.time()
    identity_check = np.allclose(A @ A_inv, np.eye(A.shape[0])) and np.allclose(A_inv @ A, np.eye(A.shape[0]))
    return x, identity_check, end - start

def qr_decomposition(A, b):
    start = time.time()
    Q, R = scipy.linalg.qr(A)
    x = scipy.linalg.solve_triangular(R, Q.T @ b)
    end = time.time()
    return x, end - start

def check_solution(A, b, x):
    return np.allclose(A @ x, b)

def main(n):
    A, b = generate_matrix_and_vector(n)

    x_lu, time_lu = lu_decomposition(A, b)
    print("LU Decomposition - correct?:", check_solution(A, b, x_lu), ", Time taken:", time_lu)

    x_inv, identity_check, time_inv = matrix_inversion(A, b)
    print("Matrix Inversion - correct?:", check_solution(A, b, x_inv), ", Time taken:", time_inv, ", identity check?:", identity_check)

    x_qr, time_qr = qr_decomposition(A, b)
    print("QR Decomposition - correct?:", check_solution(A, b, x_qr), ", Time taken:", time_qr)

# if __name__ == "__main__":
#     n = int(input(">>"))
#     main(n)

#testy
def tests():
    for n in [10, 100, 1000, 5000]:
        for i in range(3):
            A, b = generate_matrix_and_vector(n)
            print("n =", n, ", test", i+1, ":")

            x_lu, time_lu = lu_decomposition(A, b)
            print("LU Decomposition - correct?:", check_solution(A, b, x_lu), ", Time taken:", "{:.2f}".format(time_lu))

            x_inv, identity_check, time_inv = matrix_inversion(A, b)
            print("Matrix Inversion - correct?:", check_solution(A, b, x_inv), ", Time taken:", "{:.2f}".format(time_inv), ", identity check?:", identity_check)

            x_qr, time_qr = qr_decomposition(A, b)
            print("QR Decomposition - correct?:", check_solution(A, b, x_qr), ", Time taken:", "{:.2f}".format(time_qr))

            print()


# tests()
import matplotlib.pyplot as plt

def test1():
    sizes = np.linspace(10, 1000, 5, dtype=int)
    lu_times = []
    inv_times = []
    qr_times = []

    # Pomiar czasu dla każdej metody
    for n in sizes:
        A, b = generate_matrix_and_vector(n)
        
        _, lu_time = lu_decomposition(A, b)
        lu_times.append(lu_time)
        
        _, _, inv_time = matrix_inversion(A, b)
        inv_times.append(inv_time)
        
        _, qr_time = qr_decomposition(A, b)
        qr_times.append(qr_time)

    # Rysowanie wykresów
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, lu_times, marker='o', label='LU Decomposition')
    plt.plot(sizes, inv_times, marker='o', label='Matrix Inversion')
    plt.plot(sizes, qr_times, marker='o', label='QR Decomposition')
    plt.xlabel('Rozmiar macierzy n')
    plt.ylabel('Czas wykonania [s]')
    plt.title('Czas rozwiązywania układu równań dla różnych metod')
    plt.legend()
    plt.grid(True)
    plt.show()


def test2():
    # Parametry symulacji
    sizes = np.linspace(10, 1000, 5, dtype=int)
    num_runs = 50  # Liczba uruchomień dla każdego rozmiaru macierzy
    lu_times = {size: [] for size in sizes}
    inv_times = {size: [] for size in sizes}
    qr_times = {size: [] for size in sizes}

    # Pomiar czasu dla każdej metody
    for _ in range(num_runs):
        for n in sizes:
            A, b = generate_matrix_and_vector(n)
            
            _, lu_time = lu_decomposition(A, b)
            lu_times[n].append(lu_time)
            
            _, _, inv_time = matrix_inversion(A, b)
            inv_times[n].append(inv_time)
            
            _, qr_time = qr_decomposition(A, b)
            qr_times[n].append(qr_time)

    # Obliczenie średnich czasów
    avg_lu_times = [np.mean(lu_times[size]) for size in sizes]
    avg_inv_times = [np.mean(inv_times[size]) for size in sizes]
    avg_qr_times = [np.mean(qr_times[size]) for size in sizes]

    # Rysowanie wykresów
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, avg_lu_times, marker='o', label='LU Decomposition')
    plt.plot(sizes, avg_inv_times, marker='o', label='Matrix Inversion')
    plt.plot(sizes, avg_qr_times, marker='o', label='QR Decomposition')
    plt.xlabel('Rozmiar macierzy n')
    plt.ylabel('Średni czas wykonania [s]')
    plt.title('Średni czas rozwiązywania układu równań dla różnych metod (50 uruchomień)')
    plt.legend()
    plt.grid(True)
    plt.show()

def test3():
    # Parametry symulacji
    sizes = np.linspace(10, 1000, 5, dtype=int)
    num_runs = 50  # Liczba uruchomień dla każdego rozmiaru macierzy
    lu_times = {size: [] for size in sizes}
    inv_times = {size: [] for size in sizes}
    qr_times = {size: [] for size in sizes}

    # Pomiar czasu dla każdej metody
    for _ in range(num_runs):
        for n in sizes:
            A, b = generate_matrix_and_vector(n)
            
            _, lu_time = lu_decomposition(A, b)
            lu_times[n].append(lu_time)
            
            _, _, inv_time = matrix_inversion(A, b)
            inv_times[n].append(inv_time)
            
            _, qr_time = qr_decomposition(A, b)
            qr_times[n].append(qr_time)

    # Obliczenie średnich czasów i odchylenia standardowego
    avg_lu_times = [np.mean(lu_times[size]) for size in sizes]
    std_lu_times = [np.std(lu_times[size]) for size in sizes]

    avg_inv_times = [np.mean(inv_times[size]) for size in sizes]
    std_inv_times = [np.std(inv_times[size]) for size in sizes]

    avg_qr_times = [np.mean(qr_times[size]) for size in sizes]
    std_qr_times = [np.std(qr_times[size]) for size in sizes]

    # Rysowanie wykresów z odchyleniem standardowym
    plt.figure(figsize=(10, 6))
    plt.errorbar(sizes, avg_lu_times, yerr=std_lu_times, marker='o', label='LU Decomposition', capsize=5)
    plt.errorbar(sizes, avg_inv_times, yerr=std_inv_times, marker='o', label='Matrix Inversion', capsize=5)
    plt.errorbar(sizes, avg_qr_times, yerr=std_qr_times, marker='o', label='QR Decomposition', capsize=5)
    plt.xlabel('Rozmiar macierzy n')
    plt.ylabel('Średni czas wykonania [s]')
    plt.title('Średni czas rozwiązywania układu równań z odchyleniem standardowym (50 uruchomień)')
    plt.legend()
    plt.grid(True)
    plt.show()


test3()