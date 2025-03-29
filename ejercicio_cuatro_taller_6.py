import numpy as np

def F(x, y):
    """Primera ecuación del sistema: F(x, y) = y - (x^2 - 4x)"""
    return y - (x**2 - 4*x)

def G(x, y):
    """Segunda ecuación del sistema: G(x, y) = (x - 2)^2 + (y + 4) - 16"""
    return (x - 2)**2 + (y + 4) - 16

def jacobiano(x, y):
    """Calcula la matriz Jacobiana evaluada en (x, y)."""
    J = np.array([
        [-2*x + 4, 1],  # Derivadas parciales de F respecto a x e y
        [2*(x - 2), 1]   # Derivadas parciales de G respecto a x e y
    ])
    return J

def newton_raphson_3_iteraciones(x0, y0):
    """Método de Newton-Raphson limitado a 3 iteraciones."""
    x, y = x0, y0  # Punto inicial

    print(f"Iteración 0: x = {x}, y = {y}")

    for i in range(3):  # Ejecuta exactamente 3 iteraciones
        # Evaluar las funciones en el punto actual
        F_val = F(x, y)
        G_val = G(x, y)

        # Construir el vector de funciones
        F_vector = np.array([F_val, G_val])

        # Calcular el Jacobiano en el punto actual
        J = jacobiano(x, y)

        # Resolver el sistema lineal J * delta = -F_vector
        delta = np.linalg.solve(J, -F_vector)

        # Actualizar las soluciones
        x = x + delta[0]
        y = y + delta[1]

        print(f"Iteración {i + 1}: x = {x}, y = {y}")

    return x, y

# Punto inicial
x0, y0 = 0, 0

# Ejecutar el método de Newton-Raphson por 3 iteraciones
sol_x, sol_y = newton_raphson_3_iteraciones(x0, y0)

print(f"Solución aproximada después de 3 iteraciones: x = {sol_x}, y = {sol_y}")
