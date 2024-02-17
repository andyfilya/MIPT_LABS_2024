import math
from scipy.special import gamma, factorial
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T = (math.sqrt(32 * math.pi) * (gamma(5/4 + 0j) / gamma(3/4 + 0j))).real
N = 1000

delta_t = T / N

u_real = [1]
v_real = [0]
x = 0
def first_eyler(u_real, v_real, x):
  n, iter_num = 0, 10
  while n < iter_num:
    if x >= T:
      print("err while first_eyler")
      break
    u_next = u_real[n] + delta_t * v_real[n]
    v_next = v_real[n] - delta_t * (u_next ** 3)
    u_real.append(u_next)
    v_real.append(v_next)
    n += 1
    x += delta_t
def second_adams(u_real, v_real, x):
  n = 10
  const_param = 1 - (23 ** 2) / (12 ** 2) * (delta_t ** 2)
  while x < T:
    u_next = u_real[n] + delta_t / 12 * (23 * (v_real[n] + delta_t / 12 * (-16 * u_real[n-1] + 5 * u_real[n-2]))) - 16 * v_real[n-1] + 5 * v_real[n-2]
    u_next = u_next / const_param
    v_next = v_real[n] + delta_t / 12 * (23 * u_next - 16 * u_real[n-1] + 5 * u_real[n-2])
    u_real.append(u_next)
    v_real.append(v_next)
    x += delta_t
    n += 1
def net_func():
  net = []
  tmp = 0
  while tmp < T:
    net.append(tmp)
    tmp += delta_t
  return net
def draw_graph(u_real, v_real, net):

  plt.plot(net, u_real, color = "blue", label = "u(t)")
  plt.plot(net, v_real, color = "red", label = "v(t)")

  plt.legend()
  plt.show()
def main():
  x = 0
  first_eyler(u_real, v_real, x)
  x += 11 * delta_t
  second_adams(u_real, v_real, x)
  net = net_func()
  print(len(u_real), len(v_real), len(net))
  draw_graph(u_real, v_real, net)

if __name__ == "__main__":
  main()

