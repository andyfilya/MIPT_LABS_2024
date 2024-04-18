import numpy as np
import matplotlib.pyplot as plt

def solve_heat_equation(phi, a, k, h, tau, num_steps):
    # Количество узлов по x
    N = int(a / h)

    # Коэффициенты для метода Рунге-Кутты
    c1, c2 = 1/3, 1
    a11, a21 = 1/3, 0
    a22 = 1
    b1, b2 = 3/4, 1/3

    # Инициализация матрицы решения
    u = np.zeros((N+1, num_steps+1))
    x = np.linspace(0, a, N+1)
    u[:, 0] = phi(x, a)

    # Цикл по временным слоям
    for j in range(num_steps):
        # Вспомогательные векторы
        u1 = np.zeros(N+1)
        u2 = np.zeros(N+1)

        # Первый шаг Рунге-Кутты
        A = np.diag(-2 * np.ones(N-1)) + np.diag(np.ones(N-2), 1) + np.diag(np.ones(N-2), -1)
        b = -u[1:-1, j]
        u1[1:-1] = np.linalg.solve(np.eye(N-1) - a11 * tau * k / h**2 * A, b)

        # Второй шаг Рунге-Кутты
        b = -u[1:-1, j] - a21 * tau * k / h**2 * A @ u1[1:-1]
        u2[1:-1] = np.linalg.solve(np.eye(N-1) - a22 * tau * k / h**2 * A, b)

        # Обновление решения
        u[1:-1, j+1] = u[1:-1, j] + b1 * u1[1:-1] + b2 * u2[1:-1]

    return u

# Начальное условие
def phi(x, a):
    return np.where((0 <= x) & (x <= a/2), x * (a - x) / 2, 0)


a = 3
k = 10
h = 0.01
tau = 0.01
num_steps = 1000

# Вычисление решения
u = solve_heat_equation(phi, a, k, h, tau, num_steps)

# Визуализация решения (исправленная)
x = np.linspace(0, a, u.shape[0])
t = np.linspace(0, tau * num_steps, u.shape[1])
T, X = np.meshgrid(t, x)  # Изменен порядок присваивания

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, T, u, cmap='coolwarm')  
ax.set_xlabel('x')  # Возвращаем x и t на свои места
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')
ax.set_title('Численное решение уравнения теплопроводности')
plt.show()

# Параметры задачи
a = 3
k = 1
h = 0.1
tau = 0.01
num_steps = 100

# Вычисление решения
u = solve_heat_equation(phi, a, k, h, tau, num_steps)

# Визуализация решения
x = np.linspace(0, a, u.shape[0])
t = np.linspace(0, tau * num_steps, u.shape[1])
X, T = np.meshgrid(t, x)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, T, u.T, cmap='coolwarm')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('u(x,t)')
ax.set_title('Численное решение уравнения теплопроводности')
plt.show()
