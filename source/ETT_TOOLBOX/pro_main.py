"""
Learning the spline line paper of Honer, he has implemented the GGIW model in python.
Try to understand the GGIW model from the Karl's paper ETT-GGIW on marine data [Karl_TGRS15, ICIF16].

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

delta = 45.0  # degrees

angles = np.arange(0, 360 + delta, delta)
ells = [Ellipse((1, 1), 4, 2, a) for a in angles]

a = plt.subplot(111, aspect='equal')


elips = [Ellipse(xy=(6.9794254, 2.4994829), width=4.0386477, height=0.6633875, angle=0.9204723644874725),
         Ellipse(xy=(11.67689, 2.5883245), width=3.9415612, height=0.633762, angle=1.4044919057514278),
         Ellipse(xy=(16.585918, 2.9355721), width=3.9058225, height=0.63722557, angle=2.0703594706948603),
         Ellipse(xy=(21.305988, 3.3647437), width=3.9571068, height=0.63083816, angle=3.1618023557217416),
         Ellipse(xy=(26.124172, 4.021637), width=3.922539, height=0.6387129, angle=4.1194377794832535)
         ]
for e in elips:
    e.set_clip_box(a.bbox)
    e.set_alpha(0.5)
    a.add_artist(e)

plt.xlim(0, 30)
plt.ylim(0, 10)

plt.show()