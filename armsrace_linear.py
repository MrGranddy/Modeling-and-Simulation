import numpy as np
import matplotlib.pyplot as plt

# Competition case
# a1, a4 > 0, a2, a3, a5, a6 < 0, a2 * a6 > a3 * a5
# Equilibrium exists if p_ and q_ are positive where f(p_, q_) = 0 and g(p_, q_) = 0
# a2 / a5 > a1 / a4 > a3 / a6

# Case 1, Regular Case
# a = np.array([
#     5/2 + np.sqrt(3) / 24, -5/8, -np.sqrt(3) / 24,
#     7/8 + 3 * np.sqrt(3) / 2, -3 * np.sqrt(3) / 8, -7/8
# ])

# p0 = 1/4
# q0 = 3

# Case 2, Extinction of P
# a = np.array([
#     71/8, -23/12, -25/12,
#     73/8, -25/12, -23/12
# ])

# p0 = 1/4
# q0 = 1/2

# Case 3, Extinction of P and Q remains (becomes single species model)

# a = np.array([
#     7 - 3 * np.sqrt(3) / 32, -7/4, -3 * np.sqrt(3) / 16,
#     -13/8 + 12 * np.sqrt(3), -3 * np.sqrt(3), -13/4
# ])

# p0 = 1/4
# q0 = 1/2

# Case 4, Volterra-Lotka Model

a = np.array([
    -1/2, 0, 1/200,
    1/5, -1/50, 0,
])
p0 = 5
q0 = 50


p_range = [0, 30]
q_range = [0, 200]

infinitesimal = 1e-4
# time_interval = 10
time_interval = 80

def f(p, q):
    return a[0] + a[1] * p + a[2] * q

def g(p, q):
    return a[3] + a[4] * p + a[5] * q


t = 0
p = p0
q = q0

ps = [p]
qs = [q]
vps = [f(p, q) * p]
vqs = [g(p, q) * q]
ts = [t]

while t < time_interval:
    vp = f(p, q) * p
    vq = g(p, q) * q
    p += infinitesimal * vp
    q += infinitesimal * vq
    t += infinitesimal
    ps.append(p)
    qs.append(q)
    ts.append(t)
    vps.append(vp)
    vqs.append(vq)

plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.plot(ts, ps, label='p(t)')
plt.plot(ts, qs, label='q(t)')
plt.xlabel('t')
plt.ylabel('p(t), q(t)')
plt.title('Arms Race Population')
plt.legend()

qx = np.linspace(q_range[0], q_range[1], 30)
px = np.linspace(p_range[0], p_range[1], 30)
X, Y = np.meshgrid(px, qx)
U = f(X, Y)
V = g(X, Y)

norm = np.sqrt(U**2 + V**2)

U = U / norm
V = V / norm

plt.subplot(1, 2, 2)
plt.quiver(X, Y, U, V)
plt.plot(ps, qs)
plt.xlabel('p')
plt.ylabel('q')
plt.title('Phase Plane')
plt.show()

