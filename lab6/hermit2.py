import numpy as np
import math

hermit_coefficients = [
    [1],
    [2, 0],  
    [4, 0, -2],
    [8, 0, -12, 0],
    [16, 0, -48, 0, 12],
    [32, 0, -160, 0, 120, 0],
    [64, 0, -480, 0, 720, 0, -120],
    [128, 0, -1344, 0, 3360, 0, -1680, 0]
]

hermit_functions = [np.poly1d(coefficients) for coefficients in hermit_coefficients]
hermit_roots = [function.roots for function in hermit_functions]

def hermite_weight(n,x):
    return (np.sqrt(np.pi) * math.factorial(n) * (2 ** (n - 1))) /  ( (n**2) * np.polyval(hermit_functions[n-1], x)**2 )

# Funkcja, którą całkujemy
def f(x):
    return np.cos(x)

def gauss_hermite_integration(f, n):
    # Obliczenie miejsc zerowych i wag wielomianów Hermite'a
    nodes = hermit_roots[n]
    weights = [hermite_weight(n, node) for node in nodes]
    
    # Obliczenie całki jako sumy iloczynów wag i wartości funkcji w miejscach zerowych
    integral = sum(weights[i] * f(nodes[i]) for i in range(n))
    
    return integral

#uzycie
for n in [2,3,4,5,6,7]:
    result = gauss_hermite_integration(f, n)
    print("Wynik całki dla n =", n, ":", result)

