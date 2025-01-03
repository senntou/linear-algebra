# pydata/y0.csv pydata/y1.csv  pydata/y2.csvを読み込んで3次元にプロット
import numpy as np
import matplotlib.pyplot as plt

# データの読み込み
x = np.loadtxt('pydata/y0.csv', delimiter=',')
y = np.loadtxt('pydata/y1.csv', delimiter=',')
z = np.loadtxt('pydata/y2.csv', delimiter=',')

# 3次元プロット
# 3次元上に平面をプロットする
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='bwr')
plt.show()
