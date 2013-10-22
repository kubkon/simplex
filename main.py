from simplex import NelderMead

import numpy as np
import matplotlib.pyplot as plt

# Define objective function
def objective(xs):
    x1, x2 = xs[0], xs[1]
    return x1**2 + x2**2 - 3*x1 - x1*x2 + 3

simplices = []
def callback(args):
    simplices.append(args[0])

# Initial simplex
simplex = np.array([[0,0], [0,1], [1,0]], dtype=np.float)

# Initialise NelderMead simplex algorithm
nm = NelderMead(objective, simplex, epsilon=1e-1, callback=callback)

# Minimise the objective function
solution = nm.solve()

print("Minimum at {}".format(solution))

# Plot
# Tabulate objective function
x = np.linspace(-1, 3, 1000)
y = np.linspace(-2, 2, 1000)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2 - 3*X - X*Y + 3

plt.figure()
cs = plt.contour(X, Y, Z, 25)
plt.clabel(cs, inline=1, fontsize=10)

for simplex in simplices:

    lines = []
    for i in range(3):
        for j in range(i, 3):
            if j == i:
                continue

            plt.plot(*zip(simplex[i], simplex[j]), c='black')

plt.grid()
plt.savefig('simplex.pdf')
