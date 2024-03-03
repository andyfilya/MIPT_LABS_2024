import math
from scipy.special import gamma, factorial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T = (math.sqrt(32 * math.pi) * (gamma(5/4 + 0j) / gamma(3/4 + 0j))).real
delta_t = list()
u_real = [1]
v_real = [0]
x = 0

def generate_e(a : float, b : float) -> float:
  return math.fabs(a - b)

def generate_delta_t(n : int) -> list[float]: # // 10 -> for example if you want 150 pieces -> n = 10
  arr = list()
  for _ in range(n):
    arr.append(T / (n * 10))
  return arr



def first_eyler(u_real : list[float], v_real : list[float], x : float, delta_t : float):
  n, iter_num = 0, 2
  while n < iter_num:
    if x >= T:
      print("err while first_eyler")
      break
    u_next = u_real[n] + delta_t * v_real[n]
    v_next = v_real[n] - delta_t * (u_real[n] ** 3)
    u_real.append(u_next)
    v_real.append(v_next)
    n += 1
    x += delta_t
def second_adams(u_real : list[float], v_real : list[float], x : float, delta_t : float):
  n = 2
  while x < T:
    u_next = u_real[n] + delta_t / 12 * (23 * v_real[n] - 16 * v_real[n-1] + 5 * v_real[n-2])
    v_next = v_real[n] + delta_t / 12 * (-23 * (u_real[n] ** 3) + 16 * (u_real[n-1] ** 3) - 5 * (u_real[n-2] ** 3))
    u_real.append(u_next)
    v_real.append(v_next)
    x += delta_t
    n += 1
def net_func(delta_t : float) -> list[float]:
  net = []
  tmp = 0
  while tmp < T:
    net.append(tmp)
    tmp += delta_t
  return net
def draw_graph(u_real : list[float], v_real : list[float], net : list[float]):

  plt.plot(net, u_real, color = "blue", label = "u(t)")
  plt.plot(net, v_real, color = "red", label = "v(t)")

  plt.legend()
  plt.show()
def draw_error(u_tmp : list[float], v_tmp : list[float], delta_t : list[float]):
  plt.plot(delta_t, u_tmp, color = "blue", label = "||u(T) - u(0)||")
  plt.plot(delta_t, v_tmp, color = "red", label = "||v(T) - v(0)||")

  plt.legend()

  plt.xlabel("\Delta t")
  plt.ylabel("\Psi (x)")
  
  plt.show()

def main():
  u_tmp, v_tmp = list(), list()
  p = int(input())
  u_real = [1]
  v_real = [0]  
  delta_t = [T / (10 * k) for k in range (1,p)]
  for m in range(p-1):
    print("started ", m+1, "iteration...")
    x = 0
    first_eyler(u_real, v_real, x, delta_t[m]) 
    x += 3 * delta_t[m]
    second_adams(u_real, v_real, x, delta_t[m])

    u_tmp.append(generate_e(u_real[0], u_real[-1])) # for draw graph e(delta_t)
    v_tmp.append(generate_e(v_real[0], v_real[-1])) # for draw graph e(delta_t)

    net = net_func(delta_t[m])
    print(len(u_real), len(v_real), len(net))
    draw_graph(u_real, v_real, net)
    u_real = [1]
    v_real = [0]
  draw_error(u_tmp, v_tmp, delta_t)

if __name__ == "__main__":
  main()

