# FIRST MIPT LAB
## 1 __Exercise__ :
![exercise](lab1_2024.jpg)
## __Solve__:
1. first k iterations, where k = 3:

$$
u_{i+1} - u_{i} = \Delta tv_{i} \\
v_{i+1} - v_{i} = -\Delta t u_i^3
$$

2. after that adams

$$
u_{i+1} - u_{i} = \frac{\Delta t}{12}(23 v_i - 16v_{i-1} + 5v_{i-2}) \\
v_{i+1} - v_{i} = \frac{\Delta t}{12}(23 u_i^3 - 16u_{i-1}^3 + 5u_{i-2}^3) 
$$

3. solution:

![solution](graph1.png)

## __Standart Error__:
we need to draw graphs, which illustrates the 
$||f(T) - f(0)||$:

![e_delta_t](delta.png)

to compare the order of approximation, it is necessary to draw an error on a logarithmic scale, and draw a straight line with a logarithmic scale

![draw_error](draw_error.png)

to calculate the exact order of approximation, for each of the values u, v, we construct a dependence on the step of the discrepancy

![help](help.png)

from this we can conclude that the **order of approximation** of __our scheme__ is 3

## __Stability__:
For multistep methods such as Adams methods or prediction-correction methods, the stability domain can be represented in the complex plane as the domain within which all the roots of the characteristic equation corresponding to the method lie. If all the roots of the characteristic equation are inside the stability domain, then the method is considered stable.

let's consider our scheme and apply the canonical equation for this:

$$
\frac{dy}{dx}=ky
$$

substituting it into the original scheme, we obtain an equation depending on z, where $ z = k \Delta t $:

$$
\lambda ^3 + \frac{16}{12}\lambda z - \lambda^2(1+\frac{23}{12}z)=0
$$

for the rest of the reasoning, see the pictures below (wolframalpha, maple):

![1](maple1.png)
![2](maple2.png)
![3](maple3.png)

if we compare the obtained area of stability with the result from the well-known literature, we can find many similarities, but this inaccuracy is due to the number of points that I have selected, there are only 1000

![4](maple4.png)

![stability](stability.jpg)


## 2 __Exercise__:
![exercise](lab!_2024_2.png)

## __Solve__:
To solve this equation, we need to know that 
**backward differentiation**

$$
\frac{3}{2}y_{l+1} -2y_l + \frac{1}{2}y_{l-1}=f_{l+1}*\Delta t 
$$

**First attempt**:
![attempt](approach.jpg)

**Solution**:
![solution](solution2.png)

after some changes:

**Solution C_1**:

![c1](c_1_graph.png)

**Solution C_2**:

![c2](c_2_graph.png)

**Solution C_3**:

![c3](c_3_graph.png)

**Solution C_4**:

![c4](c_4_graph.png)

to set the maximum step for the explicit Euler method for this problem, I wrote the functions that you can find in the code, and the answer is that the maximum step is 17.28


