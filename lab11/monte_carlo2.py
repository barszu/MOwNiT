import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt

def rectangle_integration(func, a, b, epsilon):
    n = int((b - a) / epsilon)
    total_area = 0.0
    for i in range(n):
        x = a + i * epsilon
        total_area += func(x) * epsilon
    return total_area

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

def measure_time(method, *args):
    start_time = time.time()
    result = method(*args)
    elapsed_time = time.time() - start_time
    return result, elapsed_time

def f1(x): return x**2 + x + 1
def f2(x): return math.sqrt(1 - x**2)
def f3(x): return 1/math.sqrt(x)



for integrand , y_max, f_name in zip([f1, f2, f3], [f1(1) , f2(0) , f3(0.0001)] , ["x^2 + x + 1", "sqrt(1 - x^2)", "1/sqrt(x)"]):

    # Define integration limits
    a = 0.01  # To avoid division by zero
    b = 1

    # Define accuracy levels
    epsilons = [1e-3, 1e-4, 1e-5, 1e-6]
    num_points_list = [int(1 / epsilon) for epsilon in epsilons]

    # Store results
    rect_results = []
    monte_carlo_results = []

    # Perform calculations and measure times
    for epsilon, num_points in zip(epsilons, num_points_list):
        rect_result, rect_time = measure_time(rectangle_integration, integrand, a, b, epsilon)
        monte_carlo_result, monte_carlo_time = measure_time(hit_and_miss_integration, integrand, a, b, num_points, y_max)
        
        rect_results.append((epsilon, rect_result, rect_time))
        monte_carlo_results.append((epsilon, monte_carlo_result, monte_carlo_time))

    # Print results
    print("Rectangle Method Results:")
    for epsilon, result, time_taken in rect_results:
        print(f"Epsilon: {epsilon}, Integral: {result}, Time: {time_taken}s")

    print("\nMonte Carlo Method Results:")
    for epsilon, result, time_taken in monte_carlo_results:
        print(f"Epsilon: {epsilon}, Integral: {result}, Time: {time_taken}s")

    # Plot results
    epsilons_log = [math.log10(eps) for eps in epsilons]
    rect_times = [res[2] for res in rect_results]
    monte_carlo_times = [res[2] for res in monte_carlo_results]

    plt.plot(epsilons_log, rect_times, label="Rectangle Method", marker='o')
    plt.plot(epsilons_log, monte_carlo_times, label="Monte Carlo Method", marker='o')

    plt.xlabel("log10(Epsilon)")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.title(f"Time vs Epsilon for {f_name}")
    plt.show()
