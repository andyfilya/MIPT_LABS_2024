import math
from scipy.special import gamma, factorial
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from numpy.linalg import eigvals

t_d = 24 * 60 * 60
k_1 = []
k_2 = 10**5
k_3 = 10**(-16)
delta_t = 2 * t_d / 10000

def delta_t_generate(n : int) -> float:
  return 2 * t_d / n

def k_1_gen(real_t : float):
  k_1.append(10**(-2) * max(0, math.sin(2 * math.pi * real_t / t_d)))

c_1_real, c_2_real, c_3_real, c_4_real = [0], [0], [5 * 10 ** 11], [8 * 10 ** 11]

def c_1_tmp(delta_t : float, real_t : float) -> float:
  tmp = c_1_real[-1] + delta_t * (k_1[-1] * c_3_real[-1] - k_2 * c_1_real[-1])
  return tmp
def c_2_tmp(delta_t : float, real_t : float) -> float:
  tmp = c_2_real[-1] + delta_t * (k_1[-1] * c_3_real[-1] - k_3 * c_2_real[-1] * c_4_real[-1])
  return tmp
def c_3_tmp(delta_t : float, real_t : float) -> float:
  tmp = c_3_real[-1] + delta_t * (k_3 * c_2_real[-1] * c_4_real[-1] - k_1[-1] * c_3_real[-1])
  return tmp
def c_4_tmp(delta_t : float, real_t : float) -> float:
  tmp = c_4_real[-1] + delta_t * (k_2 * c_1_real[-1] - k_3 * c_2_real[-1] * c_4_real[-1])
  return tmp

def init_steps(vars):
  x, y, z, l = vars
  
  f1 = x - c_1_real[-1] - delta_t / 2 * (k_1[-1] * z - k_2 * x + k_1[-2] * c_3_real[-1]  - k_2 * c_1_real[-1])
  f2 = y - c_2_real[-1] - delta_t / 2 * (k_1[-1] * z - k_3 * y * l + k_1[-2] * c_3_real[-1] - k_3 * c_2_real[-1] * c_4_real[-1])
  f3 = z - c_3_real[-1] - delta_t / 2 * (k_3 * y * l - k_1[-1] * z + k_3 * c_2_real[-1] * c_4_real[-1] - k_1[-2] * c_3_real[-1])
  f4 = l - c_4_real[-1] - delta_t / 2 * (k_2 * x - k_3 * y * l  + k_2 * c_1_real[-1] - k_3 * c_2_real[-1] * c_4_real[-1])

  return [f1, f2, f3, f4]

def first_step(k : int, delta_t : float, real_t : float, net : list[float]) -> list[float]:
  init_ptr = [t_d, t_d, t_d, t_d]
  for _ in range(k):
    k_1_gen(real_t)
    
    init_ptr = fsolve(init_steps, init_ptr)
    c_1_real.append(init_ptr[0])
    c_2_real.append(init_ptr[1])
    c_3_real.append(init_ptr[2])
    c_4_real.append(init_ptr[3])

    net.append(real_t)
  return init_ptr




def draw_graph_c1(c_1_x : list[float], net : list[float]):
  plt.plot(net, c_1_x, color = "blue", label = r"c_1(t)")


  plt.xlabel("t")
  plt.ylabel(r'c_1')

  plt.grid()
  plt.legend()
  plt.show()
def draw_graph_c2(c_1_x : list[float], net : list[float]):
  plt.plot(net, c_1_x, color = "blue", label = r"c_2(t)")

  plt.xlabel("t")
  plt.ylabel(r'c_2')

  plt.grid()
  plt.legend()
  plt.show()
def draw_graph_c3(c_1_x : list[float], net : list[float]):
  plt.plot(net, c_1_x, color = "blue", label = r'c_3(t)')

  plt.xlabel("t")
  plt.ylabel(r'c_3')

  plt.grid()
  plt.legend()
  plt.show()
def draw_graph_c4(c_1_x : list[float], net : list[float]):
  plt.plot(net, c_1_x, color = "blue", label = r'c_4(t)')

  plt.xlabel("t")
  plt.ylabel(r'c_4')
  plt.grid()
  plt.legend()
  plt.show()
def net_generate(real_t : float) -> list[float]:
  return [real_t]

def to_solve(vars):
  x, y, z, l = vars

  f1 = 3 / 2 * x - 2 * c_1_real[-1] + 1 / 2 * c_1_real[-2] - delta_t * (k_1[-1] * z - k_2 * x)
  f2 = 3 / 2 * y - 2 * c_2_real[-1] + 1 / 2 * c_2_real[-2] - delta_t * (k_1[-1] * z - k_3 * y * l)
  f3 = 3 / 2 * z - 2 * c_3_real[-1] + 1 / 2 * c_3_real[-2] - delta_t * (k_3 * y * l - k_1[-1] * z)
  f4 = 3 / 2 * l - 2 * c_4_real[-1] + 1 / 2 * c_4_real[-2] - delta_t * (k_2 * x - k_3 * y * l)

  return [f1, f2, f3, f4]

def solve_eq(real_t : float, net : list[float], ptr : list[float]):
  iter_num = 1
  while real_t < 2 * t_d:
    k_1_gen(real_t)

    print(f'ptr : {ptr} \n\n')
    ptr = fsolve(to_solve, ptr)

    c_1_real.append(ptr[0])
    c_2_real.append(ptr[1])
    c_3_real.append(ptr[2])
    c_4_real.append(ptr[3])


    net.append(real_t)
    real_t += delta_t
    iter_num += 1 
    print(f' \n iter_num : {iter_num} \n ')

def jacobian(c1, c2, c3, c4, k1):
  """
  Calculate the Jacobian matrix of the system.
  """
  dc1_dc1 = -k_2
  dc1_dc2 = 0
  dc1_dc3 = k1
  dc1_dc4 = 0

  dc2_dc1 = 0
  dc2_dc2 = -k_3 * c4
  dc2_dc3 = k1
  dc2_dc4 = -k_3 * c2

  dc3_dc1 = 0
  dc3_dc2 = k_3 * c4
  dc3_dc3 = -k1
  dc3_dc4 = k_3 * c2

  dc4_dc1 = k_2
  dc4_dc2 = -k_3 * c4
  dc4_dc3 = 0
  dc4_dc4 = -k_3 * c2

  jacobian_matrix = np.array([
      [dc1_dc1, dc1_dc2, dc1_dc3, dc1_dc4],
      [dc2_dc1, dc2_dc2, dc2_dc3, dc2_dc4],
      [dc3_dc1, dc3_dc2, dc3_dc3, dc3_dc4],
      [dc4_dc1, dc4_dc2, dc4_dc3, dc4_dc4]
  ])
  return jacobian_matrix
def analyze_stability(delta_t, c1_values, c2_values, c3_values, c4_values, k1_values):
  """
  Analyze the stability of the explicit Euler method.
  """
  max_eigenvalue_product = 0
  for i in range(len(c1_values)):
    J = jacobian(c1_values[i], c2_values[i], c3_values[i], c4_values[i], k1_values[i])
    eigenvalues = np.abs(eigvals(J))
    max_eigenvalue_product = max(max_eigenvalue_product, np.max(eigenvalues * delta_t))
  
  if max_eigenvalue_product < 2:
    print("Explicit Euler method is stable for this step size.")
  else:
    print("Explicit Euler method is unstable for this step size.")

def experiment_max_step(c1_0, c2_0, c3_0, c4_0):
  """
  Experimentally determine the maximum stable step size.
  """
  max_step = delta_t
  threshold = 1e6  # Порог для определения расходимости 
  step_increase_factor = 1.1  # Фактор увеличения шага

  while True:
    c1_vals, c2_vals, c3_vals, c4_vals = [c1_0], [c2_0], [c3_0], [c4_0]
    net = [0]
    real_t = 0

    # Решение системы методом Эйлера с текущим шагом
    while real_t < 2 * t_d:
      # ... (Логика решения, аналогичная first_step и solve_eq) ...

      # Проверка на расходимость
      if any(abs(c) > threshold for c in [c1_vals[-1], c2_vals[-1], c3_vals[-1], c4_vals[-1]]):
        break

      real_t += max_step
      net.append(real_t)
      
    # Если решение не разошлось, увеличить шаг и продолжить
    if real_t >= 2 * t_d:
      max_step *= step_increase_factor
    else:
      break

  print("Максимальный устойчивый шаг (приблизительно):", max_step)

def visualize_max_step_dependence(k1_values, c1_0, c2_0, c3_0, c4_0):
  max_steps = []
  for k1 in k1_values:
    max_step = experiment_max_step(c1_0, c2_0, c3_0, c4_0)
    max_steps.append(max_step)
  
  plt.plot(k1_values, max_steps)
  plt.xlabel("k1")
  plt.ylabel("Максимальный устойчивый шаг")
  plt.show()

def main():
  real_t = 0
  k_1_gen(real_t)

  net = net_generate(real_t)
  real_t += delta_t
  ptr = first_step(10, delta_t, real_t, net) # eyler for first 10 nums
  solve_eq(real_t, net, ptr)
  print(c_1_real, c_2_real, c_3_real, c_4_real)
  print(net)

  analyze_stability(delta_t, c_1_real, c_2_real, c_3_real, c_4_real, k_1)
  experiment_max_step(c_1_real[0], c_2_real[0], c_3_real[0], c_4_real[0])
  visualize_max_step_dependence([0.1, 0.2, 0.3, 0.4, 0.5], c_1_real[0], c_2_real[0], c_3_real[0], c_4_real[0])
  draw_graph_c1(c_1_real, net)
  draw_graph_c2(c_2_real, net)
  draw_graph_c3(c_3_real, net)
  draw_graph_c4(c_4_real, net)




if __name__ == '__main__':
  main()