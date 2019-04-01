import numpy as np
import matplotlib.pyplot as plt
import sys


#Принимает два массива, строим график, возвращаем сумму массивов
def test(vector1, vector2):
    print(sys.argv)
    print(vector1 , vector2)
    scalar = np.dot(vector1, vector2)
    print(scalar)
    t = np.arange(0.0, 0.1, 0.0001)
    s = np.cos(scalar*t)
    plt.plot(t, s)
    plt.grid(True)
    plt.savefig("result.png")
    c = vector1 + vector2
    print(c)
    return c