import math
from scipy.special import gamma, factorial
import matplotlib.pyplot as plt

t_d = 24 * 60 * 60
k_2 = 10**5
k_3 = 10**(-16)

def delta_t_generate(n : int) -> float:
  return 2 * t_d / n
def k_1(real_t : float) -> float:
  return 10**(-2) * max(0, math.sin(2 * math.pi * real_t / t_d))

c_1_real, c_2_real, c_3_real, c_4_real = [0], [0], [5 * 10 ** 11], [8 * 10 ** 11]

def c_1_tmp(delta_t : float, real_t : float) -> float:
  tmp = c_1_real[-1] + delta_t * (k_1(real_t) * c_3_real[-1] - k_2 * c_1_real[-1])
  return tmp
def c_2_tmp(delta_t : float, real_t : float) -> float:
  tmp = c_2_real[-1] + delta_t * (k_1(real_t) * c_3_real[-1] - k_3 * c_2_real[-1] * c_4_real[-1])
  return tmp
def c_3_tmp(delta_t : float, real_t : float) -> float:
  tmp = c_3_real[-1] + delta_t * (k_3 * c_2_real[-1] * c_4_real[-1] - k_1(real_t) * c_3_real[-1])
  return tmp
def c_4_tmp(delta_t : float, real_t : float) -> float:
  tmp = c_4_real[-1] + delta_t * (k_2 * c_1_real[-1] - k_3 * c_2_real[-1] * c_4_real[-1])
  return tmp

def first_step(k : int, delta_t : float, real_t : float, net : list[float]):
  for _ in range(k):
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

def bacward_differnaiton(delta_t : float, real_t : float, net : list[float], n : int):
  while real_t <= 2 * t_d:
    tmp_1 = c_1_tmp(delta_t, real_t)
    tmp_2 = c_2_tmp(delta_t, real_t)
    tmp_3 = c_3_tmp(delta_t, real_t)
    tmp_4 = c_4_tmp(delta_t, real_t)

    tmp_1_append = 2 / 3 * (2 * c_1_real[n] - 1 / 2 * c_1_real[n-1] + delta_t * (k_1(real_t + delta_t) * tmp_3 - k_2 * tmp_1))
    tmp_2_append = 2 / 3 * (2 * c_2_real[n] - 1 / 2 * c_2_real[n-1] + delta_t * (k_1(real_t + delta_t) * tmp_3 - k_3 * tmp_2 * tmp_4))
    tmp_3_append = 2 / 3 * (2 * c_3_real[n] - 1 / 2 * c_3_real[n-1] + delta_t * (k_3 * tmp_2 * tmp_4 - k_1(real_t + delta_t) * tmp_3))
    tmp_4_append = 2 / 3 * (2 * c_4_real[n] - 1 / 2 * c_4_real[n-1] + delta_t * (k_1(real_t + delta_t) * tmp_1 - k_3 * tmp_2 * tmp_4))

    n += 1
    real_t += delta_t
    net.append(real_t)

    c_1_real.append(tmp_1_append)
    c_2_real.append(tmp_2_append)
    c_3_real.append(tmp_3_append)
    c_4_real.append(tmp_4_append)
def net_generate(real_t : float) -> list[float]:
  return [real_t]
def draw_graph(c_1_real : list[float], c_2_real : list[float], c_3_real : list[float], c_4_real : list[float], net : list[float]):
  plt.plot(net, c_1_real, color = "blue", label = "c_1(t)")
  plt.plot(net, c_2_real, color = "red", label = "c_2(t)")
  plt.plot(net, c_3_real, color = "orange", label = "c_3(t)")
  plt.plot(net, c_4_real, color = "green", label = "c_4(t)")

  plt.legend()
  plt.show()

def main():
  real_t = 0
  net = net_generate(real_t)
  n = int(input()) # number of steps (size of net)
  delta_t = delta_t_generate(n) # generate delta_t
  real_t += delta_t
  first_step(2, delta_t, real_t, net) # eyler for first 3 nums
  print(c_1_real, c_2_real, c_3_real, c_4_real)
  bacward_differnaiton(delta_t, real_t, net, 2)
  print(c_1_real, c_2_real, c_3_real, c_4_real)
  print(net)
  draw_graph(c_1_real, c_2_real, c_3_real, c_4_real, net)


if __name__ == '__main__':
  main()