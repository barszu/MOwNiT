import math

# Obliczenie wartości pierwiastka z dwóch
sqrt2 = math.sqrt(2)

def hermite_nodes_and_weights(n):
    # Inicjalizacja list dla miejsc zerowych i wag
    nodes = []
    weights = []

    # Początkowa wartość dla wielomianu H_0(x) i jego pochodnej
    H_prev = 1
    H_prime_prev = 0

    # Obliczanie miejsc zerowych i wag
    for i in range(1, n + 1):
        # Pierwiastek wielomianu H_n(x) jest zerem wielomianu H_{n+1}(x)
        node = math.sqrt((i - 1) / 2)

        # Obliczenie wagi
        weight = sqrt2 / (H_prev * H_prime_prev)

        # Dodanie miejsca zerowego i wagi do list
        nodes.append(node)
        weights.append(weight)

        # Aktualizacja wartości dla kolejnego wielomianu
        H_curr = node * H_prev
        H_prime_curr = H_prev + node * H_prime_prev
        H_prev = H_curr
        H_prime_prev = H_prime_curr

    return nodes, weights

# Przykładowe użycie
# n = 5  # Liczba węzłów
# nodes, weights = hermite_nodes_and_weights(n)

# print("Miejsca zerowe wielomianów Hermite'a:", nodes)
# print("Wagi wielomianów Hermite'a:", weights)


import numpy as np

# Funkcja do obliczania miejsc zerowych i wag wielomianów Hermite'a za pomocą NumPy
def hermite_nodes_and_weights(n):
    # Obliczanie węzłów i wag dla wielomianów Hermite'a
    nodes, weights = np.polynomial.hermite.hermgauss(n)
    return nodes, weights

# Funkcja, którą całkujemy
def f(x):
    return np.cos(x)

# Obliczenie całki używając kwadratury Gaussa-Hermite'a
def gauss_hermite_integration(f, n):
    # Obliczenie miejsc zerowych i wag wielomianów Hermite'a
    nodes, weights = hermite_nodes_and_weights(n)
    print(nodes)
    
    # Obliczenie całki jako sumy iloczynów wag i wartości funkcji w miejscach zerowych
    integral = sum(weights[i] * f(nodes[i]) for i in range(n))
    
    return integral


# for n in [3,4,5,6,7,10]:
#     result = gauss_hermite_integration(f, n)
#     print("Wynik całki dla n =", n, ":", result)

n = 7
result = gauss_hermite_integration(f, n)
print("Wynik całki dla n =", n, ":", result)


