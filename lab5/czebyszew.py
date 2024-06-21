import math

def czebyszew_weights_and_roots(n):
    weights = [math.pi / n] * n
    roots = [math.cos((2*k + 1) * math.pi / (2 * n)) for k in range(0, n)]
    return weights, roots

def f(x):
    # return 1 / (1 + x**2)
    return math.sqrt(1 - x**2) / (1 + x**2)

def approximate_integral(n):
    weights, roots = czebyszew_weights_and_roots(n)
    integral = sum(w * f(x) for w, x in zip(weights, roots))
    return integral

n = 9
approximation = approximate_integral(n)
print("Przybliżona wartość całki:", approximation)
print("Rzeczywista wartość całki:", math.pi / 2)
print("Błąd bezwzględny:", abs(approximation - math.pi / 2))








