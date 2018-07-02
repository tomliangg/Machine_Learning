import numpy as np

# study np reshape
a = np.arange(9).reshape(1, -1, 3)
d = a.reshape(-1, 3)
b = np.array([[1, 2], [3, 4]])
c = np.reshape(b, (-1))
print(d)
print('////')
print(a)