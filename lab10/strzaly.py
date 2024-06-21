from math import sin, cos, pi
import numpy as np
import numpy as np

def f(x, y, y_prim):
    return x - y

def exact_solution(x):
    return cos(x) - sin(x) + x

def Runge_Kutta_for_hit(iterations, h, x_0, y_0, a, func):
    x = x_0
    y = y_0
    
    for _ in range(iterations):
        k1 = h * func(x, y, a)
        k2 = h * func(x + h / 2, y + k1 / 2, a)
        k3 = h * func(x + h / 2, y + k2 / 2, a)
        k4 = h * func(x + h, y + k3, a)
        delta_a = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        k1 = h * a
        k2 = h * (a + h * func(x + h / 2, y + k1 / 2, a))
        k3 = h * (a + h * func(x + h / 2, y + k2 / 2, a))
        k4 = h * (a + h * func(x + h, y + k3, a))
        delta_y = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        x += h
        y += delta_y
        a += delta_a
    
    return x, y

def hit_method(final_iterations, a_0, a_1, x_0, y_0, x_1, y_1, func, h, epsilon):
    bisect_iterations = int((x_1 - x_0) / h)
    a = (a_0 + a_1) / 2
    y = Runge_Kutta_for_hit(bisect_iterations, h, x_0, y_0, a, func)[1]
    i = 0
    
    while abs(y - y_1) > epsilon:
        if (y - y_1) * (Runge_Kutta_for_hit(bisect_iterations, h, x_0, y_0, a_0, func)[1] - y_1) > 0:
            a_0 = a
        else:
            a_1 = a
        a = (a_0 + a_1) / 2
        y = Runge_Kutta_for_hit(bisect_iterations, h, x_0, y_0, a, func)[1]
        i += 1
    
    return Runge_Kutta_for_hit(final_iterations, h, x_0, y_0, a, func)


def generate_graph():
    import matplotlib.pyplot as plt
    n = 100
    x_vals = np.linspace(0.5, 2, n)
    y_vals_hit = [hit_method(n, -100, 100, 0, 1, pi / 2, pi / 2 - 1, f, x / n, 1e-3)[1] for x in x_vals]
    y_vals_exact = [exact_solution(x) for x in x_vals]

    # Narysuj wykres
    plt.plot(x_vals, y_vals_hit, label='Approximation')
    plt.plot(x_vals, y_vals_exact, label='Exact Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of the differential equation')
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_err_graph():
    import matplotlib.pyplot as plt

    x_vals = [0.5, 1, 2]
    n_vals = [100, 1000, 10000, 100000, 1000000]
    for i, x_val in enumerate(x_vals):
        plt.subplot(1, 3, i + 1)
        plt.title(f'x = {x_val}')
        y_graph_values = []
        for n in n_vals:
            y_val = exact_solution(x_val)
            y_approx = hit_method(n, -100, 100, 0, 1, pi / 2, pi / 2 - 1, f, x_val / n, 1e-3)[1]
            y_graph_values.append(abs(y_val - y_approx))

        plt.plot(n_vals, y_graph_values, label='Error')
        plt.xlabel('n')
        plt.ylabel('Error')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
    plt.show()


if __name__ == '__main__':
    generate_graph()
    generate_err_graph()
