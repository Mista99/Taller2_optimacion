import numpy as np
import matplotlib.pyplot as plt

# Definir la función y sus derivadas
def f(x):
    return 3 + 6 * x + 5 * x**2 + 3 * x**3 + 4 * x**4

def df(x):
    return 6 + 10 * x + 9 * x**2 + 16 * x**3  # Derivada de f(x)

def d2f(x):
    return 10 + 18 * x + 48 * x**2  # Segunda derivada de f(x)

# Método de Bisección
def biseccion(xl, xu, tol=0.01, max_iter=100,):
    if df(xl) * df(xu) >= 0:
        print("No se cumple la condición de signo opuesto en los extremos.")
        return None, []  # Devuelve None y una lista vacía si no se cumple la condición
    
    errores = []
    for _ in range(max_iter):
        xm = (xl + xu) / 2
        error = abs(f(xm))
        errores.append(error)
        
        if df(xl) * df(xm) < 0:
            xu = xm
        else:
            xl = xm
            
        if error < tol:
            break
    
    return xm, errores

# Método de Newton-Raphson
def newton_raphson(x0, tol=0.01, max_iter=100):
    errores = []
    for _ in range(max_iter):
        x_new = x0 - df(x0) / d2f(x0)
        error = abs(x_new - x0)
        errores.append(error)
        
        if error < tol:
            return x_new, errores
        x0 = x_new
        
    return x0, errores

# Método de Interpolación Cuadrática
def interpolacion_cuadratica(x0, x1, x2, tol=0.01, max_iter=700):
    errores = []
    for _ in range(max_iter):
        f0, f1, f2 = f(x0), f(x1), f(x2)
        
        # Coeficientes de la interpolación cuadrática
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        a = (f0 * (x1 - x2) + f1 * (x2 - x0) + f2 * (x0 - x1)) / denom
        b = (f0 * (x1**2 - x2**2) + f1 * (x2**2 - x0**2) + f2 * (x0**2 - x1**2)) / (2 * denom)

        # Encontrar el mínimo usando la fórmula de la parábola
        x_min = -b / (2 * a)

        # Calcular error
        error = abs(x_min - x1)
        errores.append(error)

        # Actualizar puntos
        x0, x1, x2 = x1, x2, x_min

        if error < tol:
            break
    
    return x_min, errores

# Método de Sección Dorada
def secion_dorada(a, b, tol=0.01, max_iter=100):
    errores = []
    phi = (1 + np.sqrt(5)) / 2  # Número áureo
    for _ in range(max_iter):
        d = (b - a) / phi
        x1 = b - d
        x2 = a + d

        if f(x1) < f(x2):
            b = x2
        else:
            a = x1
        
        error = abs(b - a)
        errores.append(error)

        if error < tol:
            break
            
    return (a + b) / 2, errores

# Definir los puntos iniciales
xl = -2
xu = 1
x0 = -1
tol = 0.0000001

# Usar los métodos
min_biseccion, errores_biseccion = biseccion(xl, xu, tol=tol, max_iter=100)
min_newton, errores_newton = newton_raphson(x0, tol=tol)
min_interpolacion, errores_interpolacion = interpolacion_cuadratica(-1.5, -0.3, 2, tol=tol)
min_seccion, errores_seccion = secion_dorada(xl, xu, tol=tol)

# Imprimir resultados
if min_biseccion is not None:
    print(f"Bisección: mínimo = {min_biseccion:.6f}, iteraciones = {len(errores_biseccion)}")
else:
    print("Bisección: no se encontró un mínimo válido.")

print(f"Newton-Raphson: mínimo = {min_newton:.6f}, iteraciones = {len(errores_newton)}")
print(f"Interpolación Cuadrática: mínimo = {min_interpolacion:.6f}, iteraciones = {len(errores_interpolacion)}")
print(f"Sección Dorada: mínimo = {min_seccion:.6f}, iteraciones = {len(errores_seccion)}")

# Gráficar errores
plt.plot(range(1, len(errores_biseccion) + 1), errores_biseccion, label='Bisección', marker='o')
plt.plot(range(1, len(errores_newton) + 1), errores_newton, label='Newton-Raphson', marker='x')
plt.plot(range(1, len(errores_interpolacion) + 1), errores_interpolacion, label='Interpolación Cuadrática', marker='s')
plt.plot(range(1, len(errores_seccion) + 1), errores_seccion, label='Sección Dorada', marker='^')

plt.title(f'Comparación de errores en diferentes métodos con tol = {tol}')
plt.xlabel('Iteración')
plt.ylabel('Error')
plt.yscale('log')  # Usar escala logarítmica para mejor visualización
plt.legend()
plt.grid(True)
plt.show()
