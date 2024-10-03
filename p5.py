import numpy as np
import matplotlib.pyplot as plt

# Definir la función f(x) y su derivada f'(x) para Newton-Raphson usando tangente hiperbólica
def f(x):
    return np.tanh(x**2 - 9)

def df(x):
    return 2 * x * (1 - np.tanh(x**2 - 9)**2)

# La derivada de tanh(u) es sech^2(u), y usando la identidad hiperbólica sech^2(u) = 1 - tanh^2(u),
# podemos reescribir la derivada de tanh(x^2 - 9) de la siguiente forma:
#
# Derivada: 2x * (1 - tanh^2(x^2 - 9))
# Esto es equivalente a 2x * (1 - np.tanh(x^2 - 9)**2) en Python utilizando la función tanh de NumPy.


# Método de la Bisección
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

# Método de Falsa Posición
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

# Método de Punto Fijo
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

# Definir la función g(x) para el método de punto fijo
def g(x):
    # Se elige una reformulación adecuada para que x = g(x)
    # Un ejemplo simple podría ser aproximar x usando una función derivada de f(x)
    return np.sqrt(9 + np.arctanh(x))

# Definir los valores iniciales y ejecutar los métodos
a, b = 2.82, 3.2
x0_newton = 3.18
x0_punto_fijo = 1  # Puedes ajustar este valor inicial

# Bisección
raiz_biseccion, iter_biseccion, errores_biseccion = biseccion(f, a, b)

# Falsa Posición
raiz_falsa_pos, iter_falsa_pos, errores_falsa_pos = falsa_posicion(f, a, b)

# Newton-Raphson
raiz_newton, iter_newton, errores_newton = newton_raphson(f, df, x0_newton)

# Punto Fijo
raiz_punto_fijo, iter_punto_fijo, errores_punto_fijo = punto_fijo(g, x0_punto_fijo)

# Imprimir resultados
print(f"Bisección: raíz = {raiz_biseccion:.6f}, iteraciones = {iter_biseccion}")
print(f"Falsa Posición: raíz = {raiz_falsa_pos:.6f}, iteraciones = {iter_falsa_pos}")
print(f"Newton-Raphson (x0={x0_newton}): raíz = {raiz_newton:.6f}, iteraciones = {iter_newton}")
print(f"Punto Fijo: raíz = {raiz_punto_fijo:.6f}, iteraciones = {iter_punto_fijo}")

# Graficar los errores de cada método
plt.plot(range(1, len(errores_biseccion) + 1), errores_biseccion, label='Bisección', marker='o')
plt.plot(range(1, len(errores_falsa_pos) + 1), errores_falsa_pos, label='Falsa Posición', marker='x')
plt.plot(range(1, len(errores_newton) + 1), errores_newton, label='Newton-Raphson', marker='s')
plt.plot(range(1, len(errores_punto_fijo) + 1), errores_punto_fijo, label='Punto Fijo', marker='^')

plt.title('Comparación de errores en diferentes métodos (Tangente Hiperbólica)')
plt.xlabel('Iteración')
plt.ylabel('Error')
plt.yscale('log')  # Usar escala logarítmica para mejor visualización
plt.legend()
plt.grid(True)
plt.show()
