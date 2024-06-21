import numpy as np

arr = []

# Tworzymy pierwsze dwa wielomiany Legendre'a
arr.append(np.poly1d([1]))
arr.append(np.poly1d([1, 0]))

# Definiujemy zmienną x
x = np.poly1d([1, 0])

# Generujemy kolejne wielomiany Legendre'a
for n in range(1, 6):
    new_poly = ((2 * n + 1) / (n + 1)) * arr[n] * x - (n / (n + 1)) * arr[n - 1]
    arr.append(new_poly)

# Wyświetlamy ostatni wygenerowany wielomian Legendre'a
for i , el in enumerate(arr):
    print(f"Legendre_{i} =")
    print(el)
