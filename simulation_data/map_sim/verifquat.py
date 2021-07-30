# -*- coding: utf-8 -*-
"""
Created on Tue May 11 00:01:37 2021

@author: NLEGENDN
"""

from scipy.spatial.transform import Rotation as R
import numpy as np
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
SO3 = SpecialOrthogonal(n=3, point_type="vector")


q = np.transpose([0, 1, 0, 0])
A = SO3.matrix_from_quaternion(q)
r = R.from_matrix(A)
angles = r.as_euler('zyx')

q1 = np.transpose([1, 0, 0, 0])
A1 = SO3.matrix_from_quaternion(q1)
r1 = R.from_matrix(A1)
angles1 = r1.as_euler('zyx')

# ce test permet de voir que w est la premiere valeur (rotation autour de z de 90 degres)
q2 = np.transpose([np.sin(np.pi/4), 0, 0, np.cos(np.pi/4)])
A2 = SO3.matrix_from_quaternion(q2)
r2 = R.from_matrix(A2)
angles2 = r2.as_euler('zyx', degrees=True)

r3 = R.from_euler('z', 90, degrees=True)
q3 = R.as_quat(r3)
