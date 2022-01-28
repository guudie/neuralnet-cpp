import random
import numpy as np
import pandas as pd

x = []
for i in range(0, 50):
    x.append(random.uniform(0, 10))

x2 = np.multiply(x, x)
x3 = np.multiply(x2, x)
x4 = np.multiply(x3, x)

y = np.sin(x)

fout = open("dataset.txt", "w")
data = pd.DataFrame()
data["x"] = x
data["x2"] = x2
data["x3"] = x3
data["x4"] = x4
data["y"] = y
print(data, file=fout)