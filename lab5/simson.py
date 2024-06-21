def adaptive_integration(f, a, b, tol):
    def simpson_rule(f, a, b):
        return (b - a) / 6 * (f(a) + 4 * f((a + b) / 2) + f(b))

    def adaptive_integration_helper(f, a, b, tol, whole_interval):
        c = (a + b) / 2
        left_subinterval = simpson_rule(f, a, c)
        right_subinterval = simpson_rule(f, c, b)
        total_subinterval = left_subinterval + right_subinterval

        if abs(total_subinterval - whole_interval) <= 15 * tol:
            return total_subinterval + (total_subinterval - whole_interval) / 15

        return (adaptive_integration_helper(f, a, c, tol / 2, left_subinterval) +
                adaptive_integration_helper(f, c, b, tol / 2, right_subinterval))

    whole_interval = simpson_rule(f, a, b)
    return adaptive_integration_helper(f, a, b, tol, whole_interval)

# Funkcja, którą chcemy całkować
def function(x):
    return 1 / (1 + x**2)

# Granice całkowania
a = 0
b = 1

# Tolerancja
tolerance = 1e-6

# Obliczenie całki
integral = adaptive_integration(function, a, b, tolerance)
print("Przybliżona wartość całki:", integral)
