import matplotlib.pyplot as plt

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
f.close()


f = open("../dump/training_data.txt", "r")

xx = []
y1 = []
y2 = []
n = int(next(f))
for i in range(n):
    tmp_x, tmp_y1, tmp_y2 = [float(x) for x in next(f).split()]
    xx.append(tmp_x)
    y1.append(tmp_y1)
    y2.append(tmp_y2)
f.close()


# plt.plot(x_eval, y_eval1, color='purple')
# plt.plot(x_eval, y_eval2, color='crimson')
# plt.scatter(xx, y1, color='lightsteelblue')
# plt.scatter(xx, y2, color='navajowhite')
# plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(y1, y2, xx, color='green')
ax.plot3D(y_eval1, y_eval2, x_eval, 'crimson', linewidth=2)
plt.show()