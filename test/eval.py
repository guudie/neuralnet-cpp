import matplotlib.pyplot as plt
import numpy as np

f = open("../dump/x_to_y.txt", "r")

x_eval = []
y_eval = []
n = int(next(f))
for i in range(n):
    tmp_x, tmp_y = [float(x) for x in next(f).split()]
    x_eval.append(tmp_x)
    y_eval.append(tmp_y)

# y = np.add(np.multiply(np.sin(x_eval), 2), 4)
y = np.add(np.multiply(np.add(x_eval, -5), np.add(x_eval, -5)), 5)

plt.plot(x_eval, y_eval)
plt.plot(x_eval, y)
plt.show()