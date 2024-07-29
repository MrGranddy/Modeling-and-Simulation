import numpy as np
import matplotlib.pyplot as plt

p_0 = 0
p_inf = 1000
omega = 0.4
mu = 0.1

def p(t):
    return p_inf + (p_0 - p_inf) * np.exp((-mu * t) / 2) * np.cos( np.sqrt(omega**2 - (mu**2 / 4)) * t)

t = np.linspace(0, 100, 1000)
plt.plot(t, p(t))

plt.xlabel('t')
plt.ylabel('p(t)')
plt.title('Oscillating Population')
plt.show()