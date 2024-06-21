import math

def exp_series(x, epsilon=1e-10):
    result = 0
    term = 1
    n = 0
    while abs(term) > epsilon:
        result += term
        n += 1
        term = x**n / math.factorial(n)
    return result

# Testowanie dla różnych wartości x
x_values = [1, -1, 5, -5, 10, -10]
for x in x_values:
    a = exp_series(x)
    b = math.exp(x)
    print(f"x = {x}:")
    print("moja implementacja:", a)
    print("math.exp(x):", b)
    print("roznica: ", abs(a-b))
    print()
