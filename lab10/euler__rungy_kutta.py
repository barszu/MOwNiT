from math import sin, cos, e

def f(x, y):
    return sin(x) * cos(x) - y * cos(x)

def exact_solution(x):
    return e ** (-sin(x)) + sin(x) - 1

def Runge_Kutta(iterations, h, x_0, y_0, f):
    x = x_0
    y = y_0
    for _ in range(iterations):
        k1 = h * f(x, y)
        k2 = h * f(x + h / 2, y + k1 / 2)
        k3 = h * f(x + h / 2, y + k2 / 2)
        k4 = h * f(x + h, y + k3)
        x += h
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x, y

def Euler(iterations, h, x_0, y_0, f):
    x = x_0
    y = y_0
    for _ in range(iterations):
        k1 = h * f(x, y)
        x += h
        y += k1
    return x, y


import matplotlib.pyplot as plt

n_values = [100, 1000, 10000, 100000, 1000000]
# Plot for x = 1
plt.subplot(1, 2, 1)
plt.title('x = 1')  # Add title for the plot
errors_rk_x1 = []
errors_euler_x1 = []
for n in n_values:
    x, y = Runge_Kutta(n, 1 / n, 0, 0, f)
    error_rk = abs(y - exact_solution(x))
    errors_rk_x1.append(error_rk)
    
    x, y = Euler(n, 1 / n, 0, 0, f)
    error_euler = abs(y - exact_solution(x))
    errors_euler_x1.append(error_euler)
    
plt.plot(n_values, errors_rk_x1, label="Runge-Kutta")
plt.plot(n_values, errors_euler_x1, label="Euler")
plt.xlabel('n')
plt.ylabel('Error')
plt.xscale('log')
plt.yscale('log')
plt.legend()

# Plot for x = 2
plt.subplot(1, 2, 2)
plt.title('x = 2')  # Add title for the plot
errors_rk_x2 = []
errors_euler_x2 = []
for n in n_values:
    x, y = Runge_Kutta(n, 2 / n, 0, 0, f)
    error_rk = abs(y - exact_solution(x))
    errors_rk_x2.append(error_rk)
    
    x, y = Euler(n, 2 / n, 0, 0, f)
    error_euler = abs(y - exact_solution(x))
    errors_euler_x2.append(error_euler)
    
plt.plot(n_values, errors_rk_x2, label="Runge-Kutta")
plt.plot(n_values, errors_euler_x2, label="Euler")
plt.xlabel('n')
plt.ylabel('Error')
plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.show()