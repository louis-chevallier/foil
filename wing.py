

from utillc import *
from pyxfoil import Xfoil, set_workdir, set_xfoilexe
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.interpolate import BSpline, make_interp_spline
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

print_everything()

N, B = 200, 2 * np.pi
R = 1
t = np.linspace(0, B, N)
EKOX(t.shape)


astep = pi/5
ee = pi/14
mm = R/100
a = np.arange(pi, pi/2 + ee, -astep)
X1, Y1 = R * np.cos(a), R * np.sin(a)
L = 10
X2, Y2 = [ X1[-1], L/2, L], [ Y1[-1], R*2, 0]
X3, Y3 = X2[::-1], np.asarray(Y2[::-1])-mm
a = np.arange(pi/2 + ee, -pi, -astep)
X4, Y4 = R * np.cos(a), R * np.sin(a)-mm

XX = (np.hstack((X1, X2, X3, X4)) + R) / (R+L)
YY = np.hstack((Y1, Y2, Y3, Y4)) / (R+L)

EKOX(XX.shape)
l = XX.shape[0]
y = np.c_[XX, YY]

EKOX(l)
theta = 2 * np.pi * np.linspace(0, 1, 5)

yyy = np.c_[np.cos(theta), np.sin(theta)]

EKOX(theta.shape)
cs = CubicSpline(theta, yyy, bc_type='periodic')

EKOX(np.arange(l).shape)
EKOX(y.shape)
cs = CubicSpline(np.arange(l), y)

EKOX(XX)
ss = np.arange(0, l, 1/100)
XXX, YYY = ss, cs(ss)




EKON(XX.shape, YY.shape)

if True :
	fig, ax = plt.subplots(figsize=(6.5, 4))
	plt.gca().set_aspect("equal")
	ax.plot(XX, YY)
	ax.set_xlim(-R -0.5, 1.5)

	plt.show()

h = "WINGFOIL"
l = '\n'.join([ ' '.join(map(str, (x, y))) for x,y in zip(XX, YY)])
#EKOX(l)
with open("WINGFOIL.dat", "w") as fd:
	fd.write(h + '\n' + l)

