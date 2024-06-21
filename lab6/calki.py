import numpy as np

# Definicja funkcji, którą całkujemy
def f(x):
    return np.exp(-x**2) * np.cos(x)

# Metoda prostokątów
def rectangle_method(f, a, b, n):
    h = (b - a) / n
    result = 0
    for i in range(n):
        x = a + i * h + h / 2  # środek przedziału
        result += f(x)
    result *= h
    return result

# Metoda trapezów
def trapezoidal_method(f, a, b, n):
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    result *= h
    return result

# Metoda Simpsona
def simpsons_method(f, a, b, n):
    h = (b - a) / n
    result = f(a) + f(b)
    for i in range(1, n):
        if i % 2 == 0:
            result += 2 * f(a + i * h)
        else:
            result += 4 * f(a + i * h)
    result *= h / 3
    return result

# Wyniki dla różnych metod
for n in [10, 100, 1000]:
    result_rectangle = rectangle_method(f, -10, 10, n)
    result_trapezoidal = trapezoidal_method(f, -10, 10, n)
    result_simpsons = simpsons_method(f, -10, 10, n)

    print("Wyniki dla n=", n)
    print("Metoda prostokątów:", result_rectangle)
    print("Metoda trapezów:", result_trapezoidal)
    print("Metoda Simpsona:", result_simpsons)
    print()

# //////////////////////////////////////////////////////////////////////////////////////////

def adaptive_simpsons(f, a, b, eps, whole_area=None):
    if whole_area is None:
        whole_area = (b - a) * (f(a) + 4 * f((a + b) / 2) + f(b)) / 6
    
    c = (a + b) / 2
    left_area = (c - a) * (f(a) + 4 * f((a + c) / 2) + f(c)) / 6
    right_area = (b - c) * (f(c) + 4 * f((c + b) / 2) + f(b)) / 6
    approx_area = left_area + right_area
    
    if abs(whole_area - approx_area) <= 15 * eps:
        return approx_area + (approx_area - whole_area) / 15
    
    return adaptive_simpsons(f, a, c, eps / 2, left_area) + adaptive_simpsons(f, c, b, eps / 2, right_area)

# Definicja funkcji, którą całkujemy
def f(x):
    return np.exp(-x**2) * np.cos(x)

# Obliczenie całki adaptacyjnie
result_adaptive = adaptive_simpsons(f, -10, 10, 1e-6)

print("Wynik (adaptacyjne):", result_adaptive)

