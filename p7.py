import numpy as np
import matplotlib.pyplot as plt

# Definir la función y sus derivadas
def f(x):
    return 2 * x + 3 / x

def df(x):
    return 2 - 3 / (x ** 2)

def d2f(x):
    return 6 / (x ** 3)  # Derivada de df(x)

# Método de Interpolación Cuadrática
def interpolacion_cuadratica(x0, x1, x2, tol=1e-6, max_iter=100):
    errores = []
    # Inicializar una lista para almacenar los resultados de la tabla
    resultados = []
    
    for i in range(max_iter):
        # Evaluar los puntos
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

        # Almacenar los resultados en la tabla
        resultados.append((i + 1, x0, x1, x2, x_min, error))

        # Actualizar puntos
        x0, x1, x2 = x1, x2, x_min

        if error < tol:
            break

    return x_min, errores, resultados

# Método de Newton-Raphson para mínimos
def newton_raphson_minimos(x0, tol=1e-6, max_iter=100):
    errores = []
    for _ in range(max_iter):

        x_new = x0 - df(x0) / d2f(x0)
        error = abs(x_new - x0)
        errores.append(error)
        
        if error < tol:
            return x_new, errores
        x0 = x_new
    return x0, errores

# Método de Sección Dorada
def secion_dorada(a, b, tol=1e-6, max_iter=100):
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

# Definir los puntos iniciales para interpolación cuadrática
# Definir los puntos iniciales para interpolación cuadrática
x0 = -0.1  # Cambiar a un punto positivo
x1 = 0.5 # También cercano al mínimo
x2 = 5.0 # Mantener el último punto


# Usar los métodos
min_interpolacion, errores_interpolacion, resultados_interpolacion = interpolacion_cuadratica(x0, x1, x2)
min_newton, errores_newton = newton_raphson_minimos(x1)
min_seccion, errores_seccion = secion_dorada(x0, x2)

# Imprimir resultados
print(f"Interpolación Cuadrática: mínimo = {min_interpolacion:.6f}, iteraciones = {len(errores_interpolacion)}")
print(f"Newton-Raphson: mínimo = {min_newton:.6f}, iteraciones = {len(errores_newton)}")
print(f"Sección Dorada: mínimo = {min_seccion:.6f}, iteraciones = {len(errores_seccion)}")

# Mostrar tabla de evolución de valores en la interpolación cuadrática
# print("\nEvolución de la Interpolación Cuadrática:")
# print(f"{'Iteración':<12}{'x0':<12}{'x1':<12}{'x2':<12}{'x_min':<12}{'Error':<12}")
# for i, x0_val, x1_val, x2_val, x_min_val, error_val in resultados_interpolacion:
#     print(f"{i:<12}{x0_val:<12.6f}{x1_val:<12.6f}{x2_val:<12.6f}{x_min_val:<12.6f}{error_val:<12.6f}")

# Graficar errores
plt.plot(range(1, len(errores_interpolacion) + 1), errores_interpolacion, label='Interpolación Cuadrática', marker='o')
plt.plot(range(1, len(errores_newton) + 1), errores_newton, label='Newton-Raphson', marker='x')
plt.plot(range(1, len(errores_seccion) + 1), errores_seccion, label='Sección Dorada', marker='s')

plt.title('Comparación de errores en diferentes métodos')
plt.xlabel('Iteración')
plt.ylabel('Error')
plt.yscale('log')  # Usar escala logarítmica para mejor visualización
plt.legend()
plt.grid(True)
plt.show()
