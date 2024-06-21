import random
import math

def hit_and_miss_integration(func, a, b, num_points, y_max):
    count_under_curve = 0

    for _ in range(num_points):
        x = random.uniform(a, b)
        y = random.uniform(0, y_max)
        
        if y <= func(x):
            count_under_curve += 1

    area_rectangle = (b - a) * y_max
    integral = area_rectangle * (count_under_curve / num_points)
    return integral


for n in [10**6]:
    print("n = ", n)
    print()

    def f(x): return x**2 + x + 1
    res = hit_and_miss_integration(f, 0, 1, n, f(1))
    print("x^2 + x + 1: ", res)
    print("error: ", abs(res - 11/6))
    print()


    def f(x): return math.sqrt(1 - x**2)
    res = hit_and_miss_integration(f, 0, 1, n, f(0))
    print("sqrt(1 - x^2): ", res)
    print("error: ", abs(res - math.pi/4))
    print()


    def f(x): return 1/math.sqrt(x)
    # aby nie dzielić przez zero, zaczynamy od 0.0001
    res = hit_and_miss_integration(f, 0.0001, 1, n, f(0.0001))
    print("1/sqrt(x): ", res)
    print("error: ", abs(res - 2))
    print()


import matplotlib.pyplot as plt

errors_func1 = []
errors_func2 = []
errors_func3 = []
n_values = [10**i for i in range(1, 7)]

for n in n_values:
    print("n = ", n)
    print()

    def f(x): return x**2 + x + 1
    res = hit_and_miss_integration(f, 0, 1, n, f(1))
    error = abs(res - 11/6)
    errors_func1.append(error)

    def f(x): return math.sqrt(1 - x**2)
    res = hit_and_miss_integration(f, 0, 1, n, f(0))
    error = abs(res - math.pi/4)
    errors_func2.append(error)

    def f(x): return 1/math.sqrt(x)
    res = hit_and_miss_integration(f, 0.01, 1, n, f(0.01))
    error = abs(res - 2)
    errors_func3.append(error)

plt.plot(n_values, errors_func1, label='x^2 + x + 1')
plt.plot(n_values, errors_func2, label='sqrt(1 - x^2)')
plt.plot(n_values, errors_func3, label='1/sqrt(x)')
plt.xlabel('n')
plt.ylabel('Error')
plt.xscale('log')  # Add this line to set logarithmic scale for x-axis
plt.legend()
plt.show()

