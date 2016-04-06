import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# Returns R, R_s, and R_p for reflections coming from medium 1 
# to medium 2 with an incident angle of theta
def fresnelR(n1, n2, theta):
    temp1 = np.sqrt(1.0-((n1/n2)*np.sin(theta))**2)
    temp2 = np.cos(theta)
    R_s = np.fabs( (n1*temp2-n2*temp1)/(n1*temp2+n2*temp1) )**2
    R_p = np.fabs( (n1*temp1-n2*temp2)/(n1*temp1+n2*temp2) )**2
    R = (R_s + R_p)/2
    return R

def R_eff(n1, n2):
    (R_Phi, error) = integrate.quad(lambda x: 2.0*np.sin(x)*np.cos(x)*\
            fresnelR(n1,n2,x), 0, np.pi/2)
    (R_J, error) = integrate.quad(lambda x: 3.0*np.sin(x)*(np.cos(x))**2*\
            fresnelR(n1,n2,x), 0, np.pi/2)
    return (R_Phi+R_J)/(2.0-R_Phi+R_J)
    
def surfacePhi(rho, t, mu_t, mu_a, L_s):
    c = 0.299792458 # mm/ps
    l_t = 1.0/mu_t # mm
    D = 1.0/(3.0*mu_t) # mm
    return c/(4.0*np.pi*c*D*t)**1.5 * np.exp(-mu_a*c*t) * np.exp(-rho**2.0/(4.0*c*D*t)) * \
            (np.exp(-l_t**2/(4.0*c*D*t)) - np.exp(-(2.0*L_s+l_t)**2/(4.0*c*D*t)))
            
def surfacePvsTime(t, radius, mu_t, mu_a, L_s):
    P, error = integrate.quad(lambda x: 2.0*np.pi*x*surfacePhi(x, t, mu_t, mu_a, L_s), \
            0, radius)
    return P

mu_s = 2083.0 # mm^{-1} 
g = 0.6
mu_a = 0.0 # mm^{-1}
c = 0.299792458 # mm/ps
n1 = 1.0 # background index
n2 = 1.6 # index of the medium
radius = 0.5 # collection radius into the detector
t_end = 1000 # ps
num_t = 1000 # number of time steps

mu_t = (1.0-g)*mu_s + mu_a # mm^{-1}
l_t = 1.0/mu_t # mm
D = 1.0/(3.0*mu_t) # mm
L_s = 2.0*D*(1.0+R_eff(n1,n2))/(1.0-R_eff(n1,n2)) # mm
dt = t_end/(num_t)
t_spread = radius**2/(8*D*c)

time = np.empty(num_t)
power = np.empty(num_t)
for i in range(0, num_t):
    time[i] = (i+1)*dt
    power[i] = surfacePvsTime(time[i], radius, mu_t, mu_a, L_s)

np.savetxt('p_vs_t_diffusion.dat', np.vstack((time,power)).T, \
        header = 'time (ps)\tpower(W)')

plt.figure(1)
plt.semilogy(time, power, 'r-')
plt.xlabel(r"$t$ (ps)")
plt.ylabel(r"$P ( t )$ W")

plt.figure(2)
plt.loglog(time, power, 'r-')
plt.xlabel(r"$t$ (ps)")
plt.ylabel(r"$P ( t )$ W")

plt.show()
