import numpy as np

# Definicja funkcji
def F(X):
    x , y = X
    return np.array([
        x**2 + x*y**3 - 9,
        3*x**2*y - y**3 - 4
    ])

# Definicja Jacobianu
def jacobian(X):
    x , y = X
    return np.array([
        [2*x + y**3, 3*x*y**2],
        [6*x*y, 3*x**2 - 3*y**2]
    ])

# Metoda Newtona
def newton_method(F, jacobian, X_0, tol=1e-10, max_iter=100):
    X_n = np.array(X_0)
    for i in range(max_iter):
        J = jacobian(X_n)
        Fx = F(X_n)
        try:
            # Obliczenie kroku metody Newtona
            delta_x = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            print("Błąd macierzy Jacobiego. Może być osobliwa.")
            return None
        X_n = X_n + delta_x
        if np.linalg.norm(delta_x) < tol:
            print(f'Znaleziono rozwiązanie po {i+1} iteracjach.')
            return X_n
    print('Nie znaleziono rozwiązania w dopuszczalnej liczbie iteracji.')
    return None

starting_points = [
    [-3.5,-0.5],
    [3.5,-0.5],
    [-1.5,-2.5],
    [1,1.5]
]

for X in starting_points:
    print(f"Punkt startowy: {X}")
    solution = newton_method(F, jacobian, X)
    print("Rozwiązanie:", solution)
    print('\n')

# # Punkt startowy
# x0 = [1.0, 1.0] # Dobierz odpowiedni punkt startowy

# # Wywołanie metody Newtona
# solution = newton_method(F, jacobian, x0)
# print("Rozwiązanie:", solution)
