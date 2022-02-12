from tkinter import Y
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

f = open("../dump/x_to_y.txt", "r")

x_eval = []
y_eval1 = []
y_eval2 = []
n = int(next(f))
for i in range(n):
    tmp_x, tmp_y1, tmp_y2 = [float(x) for x in next(f).split()]
    x_eval.append(tmp_x)
    y_eval1.append(tmp_y1)
    y_eval2.append(tmp_y2)


# y = np.add(np.multiply(np.sin(x_eval), 2), 4)
y1 = np.add(np.multiply(np.add(x_eval, -5), np.add(x_eval, -5)), 5)
# y2 = np.multiply(np.add(np.multiply(np.add(x_eval, -5), np.add(x_eval, -5)), -5), -1)
y2 = np.add(np.multiply(x_eval, 2), -3)

fig = plt.figure()
ax = plt.axes(projection='3d')

zline = x_eval
xline = y1
yline = y2

xline_eval = y_eval1
yline_eval = y_eval2

ax.plot3D(xline, yline, zline, 'orange')
ax.plot3D(xline_eval, yline_eval, zline, 'blue')
plt.show()

# plt.plot(x_eval, y_eval1)
# plt.plot(x_eval, y_eval2)
# plt.plot(x_eval, y1)
# plt.plot(x_eval, y2)
# plt.show()