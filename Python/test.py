import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(range(0,10), range(0,10))
fig.title("first test")
plt.show()
fig.savefig("test.png")

fig2 = plt.figure()
plt.plot(range(10,100), range(10,100))
plt.show()
fig2.savefig("test2.png")