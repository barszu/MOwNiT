import numpy as np

def generate_vector_x(n):
    return np.random.choice([-1, 0], size=n)

def generate_matrix_A(n):
    # Przykładowa macierz, proszę dostosować do swojej specyfikacji
    return np.random.rand(n, n)

def jacobi_method(A, b, x0, max_iterations=100, tolerance=1e-10):
    n = A.shape[0]
    x = x0.copy()
    D = np.diag(np.diag(A))
    R = A - D
    D_inv = np.diag(1 / np.diag(D))
    for _ in range(max_iterations):
        x_new = D_inv @ (b - R @ x)
        if np.linalg.norm(x_new - x, np.inf) < tolerance:
            return x_new, _
        x = x_new
    return x, max_iterations

def main(n):
    A = generate_matrix_A(n)
    x_true = generate_vector_x(n)
    b = A @ x_true

    x0 = np.zeros(n)  # Początkowe przybliżenie

    # Metoda Jacobiego
    x_jacobi, iterations_jacobi = jacobi_method(A, b, x0)
    print(f"Jacobi solution: {x_jacobi}, iterations: {iterations_jacobi}")

    # Metoda Czebyszewa będzie wymagała bardziej skomplikowanej implementacji
    # x_chebyshev, iterations_chebyshev = chebyshev_method(A, b, x0)
    # print(f"Chebyshev solution: {x_chebyshev}, iterations: {iterations_chebyshev}")

if __name__ == "__main__":
    main(10)
