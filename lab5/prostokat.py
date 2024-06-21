def function(x):
    return 1 / (1 + x**2)

a = 0
b = 1
h = 0.1

# Obliczanie całki za pomocą wzoru prostokątów
integral = 0
x_i = a + h/2  # Środek pierwszego podprzedziału
while x_i < b:
    integral += function(x_i)
    x_i += h

integral *= h

print("Przybliżona wartość całki:", integral)
