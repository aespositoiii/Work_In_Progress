import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0, 10, 100)

fig1 = plt.figure('figure1')
plt.clf()
plt.title('sin(x)')
plt.plot(x, np.sin(x))
plt.draw()
plt.show(block=False)


fig2 = plt.figure('figure2')
plt.clf()
plt.title('cos(x)')
plt.plot(x, np.cos(x))
plt.draw()
plt.show(block=False)

plt.waitforbuttonpress(timeout=3)

fig1 = plt.figure('figure1')
plt.clf()
plt.title('sin(x)')
plt.plot(x, np.sin(2*x))
plt.draw()
plt.show(block=False)


fig2 = plt.figure('figure2')
plt.clf()
plt.title('cos(x)')
plt.plot(x, np.cos(2*x))
plt.show(block=False)
plt.draw()
plt.waitforbuttonpress(timeout=3)