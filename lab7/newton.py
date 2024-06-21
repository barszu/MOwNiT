def newton_method(f, df, x0, tol=1e-10, max_iter=1000):
    x = x0
    for n in range(max_iter):
        f_x = f(x)
        if abs(f_x) < tol:
            print(f'Znaleziono rozwiązanie po {n+1} iteracjach.')
            return x
        df_x = df(x)
        if df_x == 0:
            print('ERROR Zero derivative. No solution found.')
            return None
        x = x - f_x / df_x
    print('WARNING No solution found within tolerance.')
    return x

def find_root_in_interval(f, df, interval):
    print(f"Przedział: {interval}")
    for initial_guess in interval:
        print(f"Przybliżenie początkowe: {initial_guess}")
        root = newton_method(f, df, initial_guess)
        print(f"Miejsce zerowe:", root)
    print('\n')

#zad1
import numpy as np

def f(x):
    return x * np.cos(x) - 1

def df(x):
    return -x * np.sin(x) + np.cos(x)

def ddf(x):
    return -2 * np.sin(x) - x * np.cos(x)

x_intervals = [[-5, -4], [-3, -1], [4, 6]]
for interval in x_intervals:
    find_root_in_interval(f, df, interval)



# zad2 
def f(x):
    return x**3 - 5*x - 6

def df(x):
    return 3*x**2 - 5

find_root_in_interval(f, df, [2, 4])

#zad3
import numpy as np

def f(x):
    return np.exp(-x) - x**2 + 1

def df(x):
    return -np.exp(-x) - 2*x

def ddf(x):
    return np.exp(-x) - 2

find_root_in_interval(f, df, [0, 2])

