import numpy as np

def generate_A(n):
    # Przykładowa macierz diagonalnie dominująca
    A = np.random.rand(n, n)
    np.fill_diagonal(A, sum(A) + 1)
    return A

def generate_b(A, x):
    return np.dot(A, x)

def generate_x(n):
    return np.random.choice([-1, 0], size=n)


def create_matrix(n:int):
    A = [[0 for _ in range(n)] for __ in range(n)]
    # przekatna glowna
    for i in range(n):
        if i in [0,n-1]:
            A[i][i] = 1
        else: A[i][i] = 2

    #przekatne po bokach
    for i in range(n):
        try:
            A[i-1][i] = 1/(i+2)
            A[i+1][i] = 1/(i+3)
        except:
            None

    return A

def generate_b(A, x):
    return np.dot(A, x)

def jacobi_method(A, b, q, max_iterations=100): #-> x_new , iter
    n = len(A)
    x = np.zeros(n)
    D = np.diag(np.diag(A))
    R = A - D
    D_inv = np.linalg.inv(np.diag(np.diag(D)))
    
    for iter in range(max_iterations):
        x_new = np.dot(D_inv, b - np.dot(R, x))
        if np.linalg.norm(x_new - x) < q or np.linalg.norm(b - np.dot(A, x_new)) / np.linalg.norm(b) < q:
            return x_new , iter
        x = x_new
    return x , iter




def chebyshev_method(A, b, q, alpha, beta, max_iterations=100): #-> x_new , iter
    n = len(A)
    x = np.zeros(n)
    D = np.diag(np.diag(A))
    R = A - D
    D_inv = np.linalg.inv(np.diag(np.diag(D)))
    
    for iter in range(max_iterations):
        x_new = x + beta * np.dot(D_inv, b - np.dot(A, x))
        if np.linalg.norm(x_new - x) < q or np.linalg.norm(b - np.dot(A, x_new)) / np.linalg.norm(b) < q:
            return x_new, iter
        x = alpha * x_new + (1 - alpha) * x
    return x, iter


def chebyshev_method(A, b, q, max_iterations=100):
    eigenvalues = np.linalg.eigvals(A)
    lambda_min = np.min(eigenvalues)
    lambda_max = np.max(eigenvalues)
    alpha = (lambda_max - lambda_min) / (lambda_max + lambda_min)
    beta = 2 / (lambda_max + lambda_min)

    n = len(A)
    x = np.zeros(n)
    D = np.diag(np.diag(A))
    R = A - D
    D_inv = np.linalg.inv(np.diag(np.diag(D)))
    
    for iter in range(max_iterations):
        x_new = x + beta * np.dot(D_inv, b - np.dot(A, x))
        if np.linalg.norm(x_new - x) < q or np.linalg.norm(b - np.dot(A, x_new)) / np.linalg.norm(b) < q:
            return x_new , iter
        x = alpha * x_new + (1 - alpha) * x
    return x , iter


import numpy as np

def chebyshev_method(A: np.ndarray, b: np.ndarray, precision: float):
    x_prior = np.zeros(len(A[0])).reshape(-1, 1)
    t = []
    results = []
    eig_vals = np.linalg.eig(A)[0]
    p, q = np.min(np.abs(eig_vals)), np.max(np.abs(eig_vals))
    r = b - A @ x_prior
    x_posterior = x_prior + 2 * r / (p + q)
    r = b - A @ x_posterior
    t.append(1)
    t.append(-(p + q) / (q - p))
    beta = -4 / (q - p)
    i = 1
    norm_one = 2
    norm_two = 2
    norm_b = np.linalg.norm(b)
    
    while norm_one > precision or norm_two > precision:
        norm_one = np.linalg.norm(abs(x_posterior - x_prior))
        norm_two = np.linalg.norm(A @ x_posterior - b) / norm_b
        results.append((i, norm_one, norm_two))
        i += 1
        t.append(2 * t[1] * t[-1] - t[-2])
        alpha = t[-3] / t[-1]
        old_prior, old_posterior = x_prior, x_posterior
        x_prior = old_posterior
        x_posterior = (1 + alpha) * old_posterior - alpha * old_prior + (beta * t[-2] / t[-1]) * r
        r = b - A @ x_posterior
    
    return x_posterior, results



n = 4  # Przykładowy wymiar
A = create_matrix(n)
# x = np.random.choice([-1, 0], size=n)  # Permutacja ze zbioru {-1, 0}
x = [-1,0,0,-1]
b = generate_b(A, x)
q = 1e-6


x_jacobi , iter_jacobi = jacobi_method(A, b, q)


# Ustawienie q, alpha, beta i uruchomienie metody Czebyszewa
alpha, beta = 0.5, 0.5
# x_chebyshev, iter_chebyshev = chebyshev_method(A, b, q, alpha, beta)

# Uruchomienie metody Czebyszewa z obliczonymi parametrami
x_chebyshev, iter_chebyshev = chebyshev_method(A, b, q)

print(x)

print("x_jacobi:")
print(x_jacobi)
print(f"po iteracjach: {iter_jacobi}")
print("x_chebyshev:")
print(x_chebyshev)
print(f"po iteracjach: {iter_chebyshev}")

print('######################')
for t in iter_chebyshev :
    print( t [ 0 ] , "&" , t [ 1 ] , "&" , t [ 2 ] , " \\\\ " )
