import matplotlib.pyplot as plt
import numpy as np

# Definir la función f(x) y su derivada f'(x) para Newton-Raphson
def f(x):
    return x**3 + 2*x**2 - 4*x + 5

def df(x):
    return 3*x**2 + 4*x - 4

# Método de la bisección
def biseccion(f, a, b, tol=1e-6, max_iter=100):
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos")
    
    errores = []
    iteraciones = 0
    while (b - a) / 2.0 > tol and iteraciones < max_iter:
        c = (a + b) / 2.0
        fc = f(c)
        error = abs((b - a) / 2.0)
        errores.append(error)
        
        if fc == 0:
            return c, iteraciones, errores
        elif f(a) * fc < 0:
            b = c
        else:
            a = c
        
        iteraciones += 1
    
    return (a + b) / 2.0, iteraciones, errores

# Método de falsa posición
def falsa_posicion(f, a, b, tol=1e-6, max_iter=100):
    errores = []
    iteraciones = 0
    while iteraciones < max_iter:
        fa, fb = f(a), f(b)
        c = b - fb * (b - a) / (fb - fa)
        fc = f(c)
        error = abs(fc)
        errores.append(error)
        
        if error < tol:
            return c, iteraciones, errores
        elif fa * fc < 0:
            b = c
        else:
            a = c
        
        iteraciones += 1
    
    return c, iteraciones, errores

# Método de Newton-Raphson
def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    errores = []
    iteraciones = 0
    x = x0
    while iteraciones < max_iter:
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-12:  # Evitar división por cero
            raise ValueError("La derivada es muy cercana a cero.")
        
        x_new = x - fx / dfx
        error = abs(x_new - x)
        errores.append(error)
        
        if error < tol:
            return x_new, iteraciones, errores
        x = x_new
        iteraciones += 1
    
    return x, iteraciones, errores

# Método de punto fijo (g(x) derivada de f(x) == 0)
def punto_fijo(g, x0, tol=1e-6, max_iter=100):
    errores = []
    iteraciones = 0
    x = x0
    while iteraciones < max_iter:
        x_new = g(x)
        error = abs(x_new - x)
        errores.append(error)
        
        if error < tol:
            return x_new, iteraciones, errores
        x = x_new
        iteraciones += 1
    
    return x, iteraciones, errores

# Definir una función g(x) para el método de punto fijo
def g(x):
    value = (x**3 + 2*x**2 + 5)/4
    if value < 0:
        return np.nan  # o puedes retornar un valor específico
    return np.sqrt(value)

# Intervalo y punto inicial
a, b = -4, -3
x0 = -3

# Bisección
raiz_biseccion, iter_biseccion, errores_biseccion = biseccion(f, a, b)

# Falsa Posición
raiz_falsa_pos, iter_falsa_pos, errores_falsa_pos = falsa_posicion(f, a, b)

# Newton-Raphson
raiz_newton, iter_newton, errores_newton = newton_raphson(f, df, x0)

# Punto Fijo
raiz_punto_fijo, iter_punto_fijo, errores_punto_fijo = punto_fijo(g, x0)

# Imprimir resultados
print(f"Bisección: raíz = {raiz_biseccion:.6f}, iteraciones = {iter_biseccion}")
print(f"Falsa Posición: raíz = {raiz_falsa_pos:.6f}, iteraciones = {iter_falsa_pos}")
print(f"Newton-Raphson: raíz = {raiz_newton:.6f}, iteraciones = {iter_newton}")
print(f"Punto Fijo: raíz = {raiz_punto_fijo:.6f}, iteraciones = {iter_punto_fijo}")

# Graficar los errores de cada método
plt.plot(range(1, len(errores_biseccion) + 1), errores_biseccion, label='Bisección', marker='o')
plt.plot(range(1, len(errores_falsa_pos) + 1), errores_falsa_pos, label='Falsa Posición', marker='x')
plt.plot(range(1, len(errores_newton) + 1), errores_newton, label='Newton-Raphson', marker='s')
plt.plot(range(1, len(errores_punto_fijo) + 1), errores_punto_fijo, label='Punto Fijo', marker='d')

plt.title('Comparación de errores en diferentes métodos')
plt.xlabel('Iteración')
plt.ylabel('Error')
plt.yscale('log')  # Usar escala logarítmica para mejor visualización
plt.legend()
plt.grid(True)
plt.show()
