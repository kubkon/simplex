from simplex.algorithm import NelderMeadSimplex

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# Define objective function
def objective(xs):
    x1, x2 = xs[0], xs[1]
    return x1**4 + x2**4 - 1.25 * x1**2 + 0.25

# Define callback function
simplices = []
def callback(args):
    simplices.append(args[0])

# Initial simplex
# simplex = np.array([[-2,0], [-1,2], [-1.75,2]], dtype=np.float)
simplex = np.array([[2,0], [1,2], [1.75,2]], dtype=np.float)

# Initialise NelderMead simplex algorithm
nm = NelderMeadSimplex(objective, simplex, epsilon=1e-6, callback=callback)

# Minimise the objective function
solution = nm.solve()

print("Minimum at {}".format(solution))

# Tabulate objective function
x = np.linspace(-3, 3, 1000)
y = np.linspace(-3, 3, 1000)
X, Y = np.meshgrid(x, y)
Z = X**4 + Y**4 - 1.25 * X**2 + 0.25

# Plot function contours together with the evolution of
# the simplices as they approach the minimum
fig = plt.figure()
cs = plt.contour(X, Y, Z, 20)
plt.clabel(cs, inline=1, fontsize=10)

for simplex in simplices:

    lines = []
    for i in range(3):
        for j in range(i, 3):
            if j == i:
                continue

            plt.plot(*zip(simplex[i], simplex[j]), c='black')

plt.xlabel(r"$$x_1$$")
plt.ylabel(r"$$x_2$$")
plt.grid()
plt.savefig('two-minima.png', bbox_inches=0, dpi=80)
