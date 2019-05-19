import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import multivariate_normal
# from mpl_toolkits.mplot3d import Axes3D

# #Parameters to set
# mu_x = 0
# variance_x = 3

# mu_y = 0
# variance_y = 15

# #Create grid and multivariate normal
# x = np.linspace(-10,10,500)
# y = np.linspace(-10,10,500)
# X, Y = np.meshgrid(x,y)
# pos = np.empty(X.shape + (2,))
# pos[:, :, 0] = X; pos[:, :, 1] = Y
# rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])

# #Make a 3D plot
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')
# plt.show()



# import matplotlib.pyplot as plt
# x, y = np.random.multivariate_normal(mean, cov, 5000).T
# plt.plot(x, y, 'x')
# plt.axis('equal')
# plt.show()




import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

h = [186, 176, 158, 180, 186, 168, 168, 164, 178, 170, 189, 195, 172,
     187, 180, 186, 185, 168, 179, 178, 183, 179, 170, 175, 186, 159,
     161, 178, 175, 185, 175, 162, 173, 172, 177, 175, 172, 177, 180]
h.sort()
hmean = np.mean(h)
hstd = np.std(h)
pdf = stats.norm.pdf(h, hmean, hstd)
plt.plot(h, pdf)	
plt.show()






