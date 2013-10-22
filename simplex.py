import logging

import numpy as np


class NelderMead:
    def __init__(self, objective, simplex, epsilon=1e-6, callback=None):
        """
        Arguments:
        objective -- objective function to be minimised. It is
        assumed the function takes a numpy array (vector) as input.
        simplex -- initial simplex. It is assumed the simplex is
        a 2D numpy array.
        epsilon -- (optional) 
        """
        self.objective = objective
        self.simplex = simplex
        self.alpha = 1
        self.beta = 0.5
        self.gamma = 2
        self.epsilon = epsilon
        self.__callback = callback

    def solve(self):
        """
        Returns the point (numpy array) that minimises
        the objective function.
        """
        # Initialise numpy arrays
        n = self.simplex.shape[0] - 1
        func_simplex = np.empty(n+1, dtype=np.float)
        func_simplex_without_1 = np.empty(n, dtype=np.float)

        while True:
            # Call back
            self.call_back()

            # Compute function values for the current simplex
            for i in np.arange(n+1):
                func_simplex[i] = self.objective(self.simplex[i])

            # Compute x for which the objective assumes the highest value
            high_index = np.argmax(func_simplex)
            high = self.simplex[high_index]
            f_high = self.objective(high)

            # Compute function values for the simplex without the highest
            # value
            j = 0
            for i in np.arange(n+1):
                value = self.simplex[i]

                if (value == high).all():
                    continue

                func_simplex_without_1[j] = self.objective(value)
                j += 1

            # Compute the centroid value
            centroid = self.__centroid(high)

            # Check for convergence to minimum
            centroids = np.ones(n+1, dtype=np.float) * self.objective(centroid)
            stopping_condition = np.sqrt(np.sum((func_simplex - centroids)**2) / (n+1))
            if stopping_condition < self.epsilon:
                break

            # Compute x for which the objective assumes the lowest value
            low = self.simplex[np.argmin(func_simplex)]
            f_low = self.objective(low)

            # Reflect
            reflection = self.__reflect(centroid, high)
            f_reflection = self.objective(reflection)

            if f_low > f_reflection:
                # Reflection is the new lowest value. Expand
                expansion = self.__expand(centroid, reflection)
                f_expansion = self.objective(expansion)

                if f_reflection > f_expansion:
                    # Expansion is the new highest value. Create new simplex
                    self.simplex[high_index] = expansion

                else:
                    # Expansion failed. Reflection is the new highest value.
                    # Create new simplex
                    self.simplex[high_index] = reflection

            elif np.amax(func_simplex_without_1, axis=0) >= f_reflection and f_reflection >= f_low:
                # Reflection is the new highest value. Create new simplex
                self.simplex[high_index] = reflection

            else:
                # Define augmented highest value
                if f_high > f_reflection:
                    high = reflection
                    f_high = self.objective(high)
                
                # Contract
                contraction = self.__contract(centroid, high)
                f_contraction = self.objective(contraction)

                if f_contraction <= f_high:
                    # Contraction is the new highest value.
                    # Create new simplex
                    self.simplex[high_index] = contraction

                else:
                    # Create new simplex
                    for i in np.arange(n+1):
                        self.simplex[i] = self.simplex[i] + 0.5 * (low - self.simplex[i])

        return self.simplex[np.argmin(func_simplex)]

    def __centroid(self, high):
        """
        Returns the centroid of all points in the simplex
        except the one that maximises the objective function.

        Arguments:
        high -- point that maximises the objective function
        """
        n = self.simplex.shape[0] - 1
        m = self.simplex.shape[1]
        values = np.empty((n,m), dtype=np.float)

        j = 0
        for i in np.arange(n+1):
            value = self.simplex[i]

            if (value == high).all():
                continue

            values[j] = value
            
            j += 1

        return 1 / n * np.sum(values, axis=0)

    def __reflect(self, centroid, value):
        """
        Returns point that correspond to the reflection
        step of the Nelder-Mead algorithm.

        Arguments:
        centroid -- the centroid
        value -- point to be reflected
        """
        return centroid + self.alpha * (centroid - value)

    def __expand(self, centroid, value):
        """
        Returns point that correspond to the expansion
        step of the Nelder-Mead algorithm.

        Arguments:
        centroid -- the centroid
        value -- point to be expanded
        """
        return centroid + self.gamma * (value - centroid)

    def __contract(self, centroid, value):
        """
        Returns point that correspond to the contraction
        step of the Nelder-Mead algorithm.

        Arguments:
        centroid -- the centroid
        value -- point to be contracted
        """
        return centroid + self.beta * (value - centroid)

    def call_back(self):
        if self.__callback:
            args = [np.copy(self.simplex).tolist()]
            self.__callback(args)
