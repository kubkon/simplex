simplex
=======

My implementation of the Nelder-Mead Simplex algorithm for unconstrained nonlinear programming.

Pull requests, comments, and suggestions are welcomed!

Installation
============
I'm assuming you are wise and you're using virtualenv and virtualenvwrapper. If not, go and [install them now](http://virtualenvwrapper.readthedocs.org/en/latest/).

In order to install the package, run in the terminal:

``` console
$ python setup.py sdist
```

Then:

``` console
$ pip install dist/Simplex-0.1.0.tar.gz
```

And you're done!

Basic usage
===========

A typical usage would look as follows:

``` python
from simplex.algorithm import NelderMeadSimplex

import numpy as np

# Define objective function
def objective(xs):
    x1, x2 = xs[0], xs[1]
    return x1**2 + x2**2 - 3*x1 - x1*x2 + 3

# Initial simplex
simplex = np.array([[0,0], [0,1], [1,0]], dtype=np.float)

# Initialise NelderMead simplex algorithm
nm = NelderMeadSimplex(objective, simplex)

# Minimise the objective function
solution = nm.solve()

print("Minimum at {}".format(solution))
```

More examples
=============
You can find more examples in the ```examples/``` folder.

License
=======

License information can be found in License.txt.
