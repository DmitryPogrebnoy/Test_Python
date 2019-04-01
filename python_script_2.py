def cast_argv(str):
    str = str.split(',')
    result = [int(elem) for elem in str]
    return result


print(sys.argv)
vector1 = cast_argv(sys.argv[1])
vector2 = cast_argv(sys.argv[2])
print(vector1, vector2)
vector1_1 = np.array([7,8,13])
vector2_1 = np.array([2,9,5])
print(vector1_1, vector2_1)
scalar = np.dot(vector1, vector2)
print(scalar)
t = np.arange(0.0, 0.1, 0.0001)
print("133")
s = np.cos(scalar*t)
plt.plot(t, s)
plt.grid(True)
plt.savefig("result.png")
print("PYS 2 OK")
