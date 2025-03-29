import numpy as np

# Función f(x) = x * sin(x)
def f(x):
    return x * np.sin(x)

# Derivada real de f(x) = x * sin(x)
def f_deriv_real(x):
    return np.sin(x) + x * np.cos(x)

# Aproximación de la derivada centrada
def D(f, h, x):
    return (f(x + h) - f(x - h)) / (2 * h)

# Cálculo del error relativo porcentual
def error_relativo(Dh, deriv_real):
    return np.abs((deriv_real - Dh) / deriv_real) * 100

# Valores de h1, h2 y el punto donde evaluaremos
h1 = 0.5
h2 = 0.25
x = 1.0

# Calcular D_{h1} y D_{h2}
Dh1 = D(f, h1, x)
Dh2 = D(f, h2, x)

# Calcular derivada real
deriv_real = f_deriv_real(x)

# Calcular errores relativos porcentuales
error_h1 = error_relativo(Dh1, deriv_real)
error_h2 = error_relativo(Dh2, deriv_real)

# Calcular derivada mejorada usando Richardson
DR = (4/3) * Dh2 - (1/3) * Dh1

# Resultados
print(f"Derivada de f(x) = x*sin(x) en x={x}:")
print(f"  D_{h1} = {Dh1}")
print(f"  D_{h2} = {Dh2}")
print(f"  Derivada real: {deriv_real}")
print(f"  Error relativo de D_{h1}: {error_h1}%")
print(f"  Error relativo de D_{h2}: {error_h2}%")
print(f"  Derivada mejorada (Richardson) D_R: {DR}")
