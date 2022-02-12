import random
import numpy as np
import pandas as pd

n = 200
x = []
for i in range(0, n):
    x.append(random.uniform(0, 10))

x2 = np.multiply(x, x)
x3 = np.multiply(x2, x)
x4 = np.multiply(x3, x)

# y = []
# for i in range(len(x)):
#     y.append(15 * x3[i] + random.uniform(-5, 5))
# y = np.add(np.multiply(np.sin(x), 2), 4)
y1 = np.add(np.multiply(np.add(x, -5), np.add(x, -5)), 5)
# y2 = np.multiply(np.add(np.multiply(np.add(x, -5), np.add(x, -5)), -5), -1)
y2 = np.add(np.multiply(x, 2), -3)

# add some noise to data
y1 = np.add(y1, np.multiply(np.random.rand(n), 2))
y2 = np.add(y2, np.multiply(np.random.rand(n), 2))

fout = open("../dump/dataset.txt", "w")
pd.set_option('display.float_format', lambda x: '%.10f' % x)
pd.set_option("display.max_rows", None, "display.max_columns", None)
data = pd.DataFrame()
data["x"] = x
data["x2"] = x2
# data["x3"] = x3
# data["x4"] = x4
data["y1"] = y1
data["y2"] = y2
print(n, file=fout)
print(2, 2, file=fout)
print(data.to_string(index=False, header=False), file=fout)


tout = open("../dump/training_data.txt", "w")
print(n, file=tout)
print(data.drop('x2', axis=1).to_string(index=False, header=False), file=tout)