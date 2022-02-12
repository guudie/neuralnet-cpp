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


fig = plt.figure()
ax = plt.axes(projection='3d')

zline = x_eval

xline_eval = y_eval1
yline_eval = y_eval2

ax.scatter(y1, y2, xx, color='green')
ax.plot3D(xline_eval, yline_eval, zline, 'red', linewidth=2)
plt.show()

# plt.plot(x_eval, y_eval1)
# plt.plot(x_eval, y_eval2)
# plt.plot(x_eval, y1)
# plt.plot(x_eval, y2)
# plt.show()