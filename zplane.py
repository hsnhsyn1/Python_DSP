
# coding: utf-8

# In[ ]:

# Copyright (c) 2020 Hasan Huseyin Sonmez
#
# The following code implements a pole-zero diagram for discrete-time signal/systems
# It has been written to mimic the "zplane" function of Mathworks' MATLAB software.
# it is aimed as a supplementary tool to be used in
# ELM368 Fundamentals of Digital Signal Processing-laboratory course.
# Gebze Technical University, Kocaeli, Turkey
#
# The function parameters are:
#     Input:
#            b : the numerator coefficients of the discrete-time signal/system
#            a : the denominator coefficients of the discrete-time signal/system
#


# In[ ]:

# import the necessary libraries
import numpy as np              # for using basic array functions
import matplotlib.pyplot as plt # for this example, it may not be necessary

# the main package for signal processing is called "scipy" and we will use "signal" sub-package
import scipy.signal as sgnl 
# alternative syntax: from scipy import signal as sgnl
get_ipython().magic('matplotlib notebook')

# Input: numerator and denominator coefficients:
b = np.array([1, -3, 1, 1, 0, 4])
a = np.array([1, -6.5, -3.1875, -2.0625, -0.8594, -0.1094])

# zplane(b,a)

# def zplane(b, a)

tol = 1e-4

# Calculate the poles and zeros
zeross = np.roots(b)
poless = np.roots(a)

real_p = np.real(poless)
real_z = np.real(zeross)
imag_p = np.imag(poless)
imag_z = np.imag(zeross)

real_z[abs(real_z) < tol] = 0
imag_z[abs(imag_z) < tol] = 0
real_p[abs(real_p) < tol] = 0
imag_p[abs(imag_p) < tol] = 0

z = np.round(real_z,2) + 1j*np.round(imag_z,2)
p = np.round(real_p,2) + 1j*np.round(imag_p,2)

# plot the unit circle
N = 128
m = np.arange(0,N,1)
unitCircle = np.exp(1j*m*2*np.pi/N)

plt.figure()
plt.plot(np.real(unitCircle), np.imag(unitCircle), 'b--', linewidth=0.3)
plt.xlabel('Real Part'), plt.ylabel('Imaginary Part')

# calculate the plot limits
Cz, z_counts = np.unique(z, return_counts=True)
Cp, p_counts = np.unique(p, return_counts=True)

zz = [idx1 for idx1, valz in enumerate(z_counts) if valz > 1]
pp = [idx2 for idx2, valp in enumerate(p_counts) if valp > 1]
zval = z_counts[zz]
pval = p_counts[pp]
zs = Cz[zz]
ps = Cp[pp]

real_roots = np.concatenate((np.real(p),np.real(z)))
imag_roots = np.concatenate((np.imag(p),np.imag(z)))

# adjust plot limits
xlower = min(-1, min(real_roots)) - 0.3
xupper = max(1,  max(real_roots)) + 0.3
ylower = min(-1, min(imag_roots)) - 0.3
yupper = max(1,  max(imag_roots)) + 0.3
plt.xlim(xlower, xupper), plt.ylim(ylower, yupper)

# plot axes
Xaxis = np.arange(xlower, xupper, 0.1)
Yaxis = np.arange(ylower, yupper, 0.1)
plt.plot(np.real(Xaxis), np.imag(Xaxis), 'b--', linewidth=0.3)
plt.plot(np.imag(Yaxis), np.real(Yaxis), 'b--', linewidth=0.3)

# plot poles and zeros
plt.plot(np.real(z), np.imag(z), 'ro',  markerfacecolor = 'none')
plt.plot(np.real(p), np.imag(p), 'rx')

if zz:
    txtz = str(zval)[1:-1]
    plt.annotate(txtz, xy=(np.real(zs)+0.1,np.imag(zs)+0.1))
    
if pp:
    txtp = str(pval)[1:-1]
    plt.annotate(txtp, xy=(np.real(ps)+0.1,np.imag(ps)+0.1))

plt.grid()
plt.show()


# In[ ]:



