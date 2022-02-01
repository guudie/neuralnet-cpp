import random
import numpy as np
import pandas as pd

x = []
for i in range(0, 200):
    x.append(random.uniform(-4, 4))

x2 = np.multiply(x, x)
x3 = np.multiply(x2, x)
x4 = np.multiply(x3, x)
x5 = np.multiply(x4, x)

# y = []
# for i in range(len(x)):
#     y.append(15 * x3[i] + random.uniform(-5, 5))
y = np.add(np.multiply(np.sin(x), 2), 4)

fout = open("../dump/dataset.txt", "w")
pd.set_option('display.float_format', lambda x: '%.10f' % x)
pd.set_option("display.max_rows", None, "display.max_columns", None, "display.width", 1000)
data = pd.DataFrame()
data["x"] = x
data["x2"] = x2
data["x3"] = x3
# data["x4"] = x4
# data["x5"] = x5
data["y"] = y
print(len(data), file=fout)
print(3, 1, file=fout)
print(data, file=fout)