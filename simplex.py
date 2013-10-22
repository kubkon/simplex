import logging

import numpy as np

logging.basicConfig(level=logging.DEBUG)

class NelderMead:
    def __init__(self, objective, simplex):
        """
        Arguments:
        objective -- objective function to be minimised. It is
        assumed the function takes a numpy array (vector) as input.
        simplex -- initial simplex. It is assumed the simplex is
        a 2D numpy array.
        """
        self.objective = objective
        self.simplex = simplex
        self.alpha = 1
        self.beta = 0.5
        self.gamma = 2
        self.epsilon = 1e-1

    def solve(self):
        """
        Returns numpy array that minimises the objective function.
        """
        # Initialise numpy arrays
        n = self.simplex.shape[0] - 1
        func_simplex = np.empty(n+1, dtype=np.float)
        func_simplex_without_1 = np.empty(n, dtype=np.float)

        for i in range(14):
            logging.debug("Current simplex:\n%s", self.simplex)

            # Compute function values for the current simplex
            for i in np.arange(n+1):
                func_simplex[i] = self.objective(self.simplex[i])
            logging.debug("Function values for the simplex:\n%s", func_simplex)

            # Compute x for which the objective assumes the highest value
            high_index = np.argmax(func_simplex)
            high = self.simplex[high_index]
            logging.debug("Current maximum: %s", high)

            # Compute function values for the simplex without the highest
            # value
            j = 0
            for i in np.arange(n):
                value = self.simplex[i]

                if (value == high).all():
                    continue

                func_simplex_without_1[j] = self.objective(value)
                j += 1

            # Compute the centroid value
            centroid = self.__centroid(high)

            # Check for convergence to minimum
            centroids = np.ones(n+1, dtype=np.float) * self.objective(centroid)
            print(np.sqrt(1 / n+1 * np.sum((func_simplex - centroids)**2)))
            if np.sqrt(1 / n+1 * np.sum((func_simplex - centroids)**2)) < self.epsilon:
                break

            # Compute x for which the objective assumes the lowest value
            low = self.simplex[np.argmin(func_simplex)]

            # Reflect
            reflection = self.__reflect(centroid, high)

            if self.objective(low) > self.objective(reflection):
                # Reflection is the new lowest value. Expand
                expansion = self.__expand(centroid, reflection)

                if self.objective(reflection) > self.objective(expansion):
                    # Expansion is the new highest value. Create new simplex
                    self.simplex[high_index] = expansion

                else:
                    # Expansion failed. Reflection is the new highest value.
                    # Create new simplex
                    self.simplex[high_index] = reflection

            elif np.amax(func_simplex_without_1, axis=0) >= self.objective(reflection):
                # Reflection is the new highest value. Create new simplex
                self.simplex[high_index] = reflection

            else:
                # Define augmented highest value
                if self.objective(high) > self.objective(reflection):
                    high = reflection
                
                # Contract
                contraction = self.__contract(centroid, high)

                if self.objective(contraction) <= self.objective(high):
                    # Contraction is the new highest value.
                    # Create new simplex
                    self.simplex[high_index] = contraction

                else:
                    # Create new simplex
                    for i in np.arange(n+1):
                        self.simplex[i] = self.simplex[i] + 0.5 * (low - self.simplex[i])

        return self.simplex[np.argmin(func_simplex)]

    def __centroid(self, high):
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
        return centroid + self.alpha * (centroid - value)

    def __expand(self, centroid, value):
        return centroid + self.gamma * (value - centroid)

    def __contract(self, centroid, value):
        return centroid + self.beta * (value - centroid)
