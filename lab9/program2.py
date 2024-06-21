import numpy as np

def generate_A(n):
    # Przykładowa macierz diagonalnie dominująca
    A = np.random.rand(n, n)
    np.fill_diagonal(A, sum(A) + 1)
    return A

def generate_b(A, x):
    return np.dot(A, x)

def generate_random_x(n):
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
        
        # distance = np.linalg.norm(x - x_new)
        # print("Odległość między x i new_x:", distance)
        x = x_new
    return x , iter

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
        # old_x = x
        x = alpha * x_new + (1 - alpha) * x
        # distance = np.linalg.norm(old_x - x_new)
        # print("Odległość między x i new_x:", distance)
    return x , iter







def main1(): #deprecated
    n = 4  # Przykładowy wymiar
    A = create_matrix(n)
    # x = np.random.choice([-1, 0], size=n)  # Permutacja ze zbioru {-1, 0}
    x = [-1,0,0,-1]
    b = generate_b(A, x)
    q = 1e-6


    x_jacobi , iter_jacobi = jacobi_method(A, b, q)
    x_chebyshev, iter_chebyshev = chebyshev_method(A, b, q)

    print(x)

    print("x_jacobi:")
    print(x_jacobi)
    print(f"po iteracjach: {iter_jacobi}")
    print("x_chebyshev:")
    print(x_chebyshev)
    print(f"po iteracjach: {iter_chebyshev}")

    print('######################')
    # for t in iter_chebyshev :
    #     print( t [ 0 ] , "&" , t [ 1 ] , "&" , t [ 2 ] , " \\\\ " )

def main2(n):
    import matplotlib.pyplot as plt
    # n = 100
    A = create_matrix(n)
    # x = [-1,-1,-1,-1,-1]
    x = generate_random_x(n)
    b = generate_b(A, x)

    res_jacobi = []
    res_chebyshev = []

    for q in [1e-2, 1e-4, 1e-6, 1e-10]:
        print(f"q = {q}")
        x_jacobi , iter_jacobi = jacobi_method(A, b, q)
        x_chebyshev, iter_chebyshev = chebyshev_method(A, b, q)

        print("x_jacobi:")
        print(x_jacobi)
        print(f"po iteracjach: {iter_jacobi}")
        print("x_chebyshev:")
        print(x_chebyshev)
        print(f"po iteracjach: {iter_chebyshev}")
        print('\n')

        res_jacobi.append((q, iter_jacobi))
        res_chebyshev.append((q, iter_chebyshev))

    import matplotlib.pyplot as plt
    qs_jacobi, iterations_jacobi = zip(*res_jacobi)

    # Rozpakowanie danych dla metody Czebyszewa
    qs_chebyshev, iterations_chebyshev = zip(*res_chebyshev)

    # Tworzenie wykresów
    plt.figure(figsize=(10, 5))

    # Wykres dla metody Jacobiego
    plt.subplot(1, 2, 1)
    plt.plot(qs_jacobi, iterations_jacobi, marker='o', label='Metoda Jacobiego')
    plt.xscale('linear')
    plt.yscale('linear')  # Zmiana na skalę liniową
    plt.title(f'Metoda Jacobiego (n={n})')
    plt.xlabel('q (kryterium zbieżności)')
    plt.ylabel('Liczba iteracji')
    # plt.xlim(qs_jacobi[-1], qs_jacobi[0])
    plt.ylim(0, max(iterations_jacobi) + 5)  # Ustawienie granic dla osi Y
    plt.gca().invert_xaxis()  # Odwrócenie osi Y
    plt.legend()

    # Wykres dla metody Czebyszewa
    plt.subplot(1, 2, 2)
    plt.plot(qs_chebyshev, iterations_chebyshev, marker='o', label='Metoda Czebyszewa')
    plt.xscale('linear')
    plt.yscale('linear')  # Zmiana na skalę liniową
    plt.title(f'Metoda Czebyszewa (n={n})')
    plt.xlabel('q (kryterium zbieżności)')
    plt.ylabel('Liczba iteracji')
    # plt.xlim(qs_chebyshev[-1], qs_chebyshev[0])
    plt.ylim(0, max(iterations_chebyshev) + 5)  # Ustawienie granic dla osi Y
    plt.gca().invert_xaxis()  # Odwrócenie osi Y
    plt.legend()

    plt.tight_layout()
    plt.show()


def main3():
    # Macierz współczynników
    A = np.array([
        [10, -1, 2, -3],
        [1, 10, -1, 2],
        [2, 3, 20, -1],
        [3, 2, 1, 20]
    ])

    # Wektor wyrazów wolnych
    b = np.array([0, 5, -10, 15])

    # Wektor niewiadomych
    x = np.array([0, 0, 0, 0])  # Zakładam początkowe przybliżenie x

    for q in [1e-3, 1e-4, 1e-5]:
        x_jacobi, iter_jacobi = jacobi_method(A, b, q)
        print(f"q = {q:.0e}")
        print(f"x: {x_jacobi}")
        print(f"po iteracjach: {iter_jacobi}")
        print('\n')





if __name__ == '__main__':
    # main1()
    # main2()
    # for n in [10, 100, 1000]:
    #     main2(n)

    main3()

    

       