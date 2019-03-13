import math
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def test():
    print(sys.argv)
    c = math.e
    print("Exp",c)
    z = np.zeros(10)
    print(z)
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2*np.pi*t)
    plt.plot(t, s)
    plt.grid(True)
    plt.savefig("test.png")
    c = math.factorial(10) + math.e
    return c