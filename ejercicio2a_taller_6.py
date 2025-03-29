import numpy as np

# Función para el método de Jacobi con cálculo de error relativo
def jacobi(A, b, x0, tol, max_iter):
    n = len(b)
    x = np.zeros_like(x0)
    print("Método de Jacobi:")
    for k in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i][j] * x0[j] for j in range(n) if i != j)
            x_new[i] = (b[i] - s) / A[i][i]
        
        # Calcular el error relativo
        error_relativo = np.linalg.norm(x_new - x0, np.inf) / np.linalg.norm(x_new, np.inf)
        
        # Mostrar resultados de la iteración y el error relativo
        print(f"Iteración {k + 1}: {x_new}, Error relativo: {error_relativo}")
        
        # Criterio de parada
        if error_relativo < tol:
            break
        
        x0 = x_new
    return x_new, k + 1

# Función para el método de Gauss-Seidel con cálculo de error relativo
def gauss_seidel(A, b, x0, tol, max_iter):
    n = len(b)
    x = np.copy(x0)
    print("\nMétodo de Gauss-Seidel:")
    for k in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s = sum(A[i][j] * x_new[j] for j in range(i)) + sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s) / A[i][i]
        
        # Calcular el error relativo
        error_relativo = np.linalg.norm(x_new - x, np.inf) / np.linalg.norm(x_new, np.inf)
        
        # Mostrar resultados de la iteración y el error relativo
        print(f"Iteración {k + 1}: {x_new}, Error relativo: {error_relativo}")
        
        # Criterio de parada
        if error_relativo < tol:
            break
        
        x = x_new
    return x_new, k + 1

# Sistema de ecuaciones
A = np.array([[10, -1, 2, 0], 
              [-1, 11, -1, 3], 
              [2, -1, 10, -1], 
              [0, 3, -1, 8]])
b = np.array([6, 25, -11, 15])

# Valores iniciales
x0 = np.zeros_like(b)

# Tolerancia y número máximo de iteraciones
tol = 1e-3
max_iter = 10

# Ejecutar Jacobi
sol_jacobi, iter_jacobi = jacobi(A, b, x0, tol, max_iter)
print(f"\nSolución final de Jacobi: {sol_jacobi} en {iter_jacobi} iteraciones")

# Ejecutar Gauss-Seidel
sol_gauss_seidel, iter_gauss_seidel = gauss_seidel(A, b, x0, tol, max_iter)
print(f"\nSolución final de Gauss-Seidel: {sol_gauss_seidel} en {iter_gauss_seidel} iteraciones")
