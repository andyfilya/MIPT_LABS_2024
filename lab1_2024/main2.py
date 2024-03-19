import math
from scipy.special import gamma, factorial
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

t_d = 24 * 60 * 60
k_1 = []
k_2 = 10**5
k_3 = 10**(-16)
delta_t = 2 * t_d / 10

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

def first_step(k : int, delta_t : float, real_t : float, net : list[float]):
  for _ in range(k):
    k_1_gen(real_t)
    tmp_1 = c_1_tmp(delta_t, real_t)
    tmp_2 = c_2_tmp(delta_t, real_t)
    tmp_3 = c_3_tmp(delta_t, real_t)
    tmp_4 = c_4_tmp(delta_t, real_t)
    real_t += delta_t
    net.append(real_t)
    c_1_real.append(tmp_1)
    c_2_real.append(tmp_2)
    c_3_real.append(tmp_3)
    c_4_real.append(tmp_4)
    


def draw_graph_c1(c_1_x : list[float], net : list[float]):
  plt.plot(net, c_1_x, color = "blue", label = "c_1(t)")

  plt.legend()
  plt.show()

def draw_graph_c2(c_1_x : list[float], net : list[float]):
  plt.plot(net, c_1_x, color = "blue", label = "c_2(t)")

  plt.legend()
  plt.show()
def draw_graph_c3(c_1_x : list[float], net : list[float]):
  plt.plot(net, c_1_x, color = "blue", label = "c_3(t)")

  plt.legend()
  plt.show()
def draw_graph_c4(c_1_x : list[float], net : list[float]):
  plt.plot(net, c_1_x, color = "blue", label = "c_4(t)")

  plt.legend()
  plt.show()
def net_generate(real_t : float) -> list[float]:
  return [real_t]

def to_solve(vars):
  x, y, z, l = vars

  f1 = 3 / 2 * x - 2 * c_1_real[-1] + 1 / 2 * c_1_real[-2] - delta_t * (k_1[-1] * z - k_2 * x)
  f2 = 3 / 2 * y - 2 * c_2_real[-1] + 1 / 2 * c_2_real[-2] - delta_t * (k_1[-1] * z - k_2 * y * l)
  f3 = 3 / 2 * x - 2 * c_3_real[-1] + 1 / 2 * c_3_real[-2] - delta_t * (k_3 * y * l - k_1[-1] * z)
  f4 = 3 / 2 * x - 2 * c_4_real[-1] + 1 / 2 * c_4_real[-2] - delta_t * (k_2 * x - k_3 * y * l)

  return [f1, f2, f3, f4]

def solve_eq(real_t : float, net : list[float]):
  iter_num = 1
  ptr = [t_d, t_d, t_d, t_d]
  while real_t < 2 * t_d:
    k_1_gen(real_t + delta_t)

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


def main():
  real_t = 0
  net = net_generate(real_t)
  real_t += delta_t
  first_step(2, delta_t, real_t, net) # eyler for first 3 nums
  solve_eq(real_t, net)
  print(c_1_real, c_2_real, c_3_real, c_4_real)
  print(net)
  draw_graph_c1(c_1_real, net)
  draw_graph_c2(c_2_real, net)
  draw_graph_c3(c_3_real, net)
  draw_graph_c4(c_4_real, net)



if __name__ == '__main__':
  main()