from simplex import NelderMead

import numpy as np

# Define objective function
def objective(xs):
    x1, x2 = xs[0], xs[1]
    return x1**2 + x2**2 - 3*x1 - x1*x2 + 3

# Initial simplex
simplex = np.array([[0,0], [0,1], [1,0]], dtype=np.float)

# Initialise NelderMead simplex algorithm
nm = NelderMead(objective, simplex)

# Minimise the objective function
solution = nm.solve()

print(solution)
