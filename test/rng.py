import random
import numpy as np
import pandas as pd

x = []
for i in range(0, 100):
    x.append(random.uniform(0, 10))

x2 = np.multiply(x, x)
x3 = np.multiply(x2, x)
x4 = np.multiply(x3, x)

# y = []
# for i in range(len(x)):
#     y.append(15 * x3[i] + random.uniform(-5, 5))
y = np.add(np.multiply(x, 2), 5)

fout = open("../dump/dataset.txt", "w")
pd.set_option('display.float_format', lambda x: '%.10f' % x)
pd.set_option("display.max_rows", None, "display.max_columns", None)
data = pd.DataFrame()
data["x"] = x
# data["x2"] = x2
# data["x3"] = x3
# data["x4"] = x4
data["y"] = y
print(len(data), file=fout)
print(1, 1, file=fout)
print(data, file=fout)