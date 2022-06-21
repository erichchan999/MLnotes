import numpy as np
import matplotlib.pyplot as plt
import fractions
np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})

a = np.array([[3,3],[4,4]])

print(np.linalg.inv(a))
