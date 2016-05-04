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

def longTimeP(t, radius, mu_t, mu_a, L_s):
    return (np.pi*radius**2*c)/((4.0*np.pi*D*c*t)**1.5) * (L_s*(L_s+l_t))/(D*c*t)*np.exp(-mu_a*c*t)

def spatialP(rho, D, l_t, mu_a, L_s):
    return 1.0/(4.0*np.pi*D) * ( np.exp(-np.sqrt(mu_a/D*(rho**2 + l_t**2)))/np.sqrt(rho**2 + l_t**2) - np.exp(-np.sqrt(mu_a/D*(rho**2 + (2.0*L_s+l_t)**2)))/np.sqrt(rho**2 + (2.0*L_s+l_t)**2) )

def spatialP_approx(rho, D, l_t, mu_a, L_s):
    return np.sqrt(mu_a/D) * L_s*(L_s+l_t)*np.exp(-np.sqrt(mu_a/D)*rho)/(2.0*np.pi*D*rho**2)

def spatialP_approx2(rho, D, l_t, L_s):
    return L_s*(L_s+l_t)/(2.0*np.pi*D*rho**3)

mu_s = 2083.0 # mm^{-1} 
g = 0.6
mu_a = 0.001 # mm^{-1}
c = 0.299792458 # mm/ps
n1 = 1.0 # background index
n2 = 1.6 # index of the medium
radius = 0.5 # collection radius into the detector
t_end = 10000 # ps
num_t = 1000 # number of time steps
rho_end = 1.0 # mm
num_rho = 1001 # number of radial steps

mu_t = (1.0-g)*mu_s + mu_a # mm^{-1}
l_t = 1.0/mu_t # mm
D = 1.0/(3.0*mu_t) # mm
L_s = 2.0*D*(1.0+R_eff(n1,n2))/(1.0-R_eff(n1,n2)) # mm
dt = t_end/(num_t)
drho = rho_end/(num_rho+1)
t_spread = radius**2/(8*D*c)

time = np.empty(num_t)
power = np.empty(num_t)
powerLong = np.empty(num_t)
for i in range(0, num_t):
    time[i] = (i+1)*dt
    power[i] = surfacePvsTime(time[i], radius, mu_t, mu_a, L_s)
    powerLong[i] = longTimeP(time[i], radius, mu_t, mu_a, L_s)

np.savetxt('p_vs_t_diffusion.dat', np.vstack((time,power)).T, \
        header = 'time (ps)\tpower(W)')

np.savetxt('approx.dat', np.vstack((time,power,powerLong)).T, \
        header = 'time (ps)\tpower(W)\tpower approximation(W)')

plt.figure(1)
plt.loglog(time, power, 'r-', time, powerLong, 'b-')
plt.xlabel(r"$t$ (ps)")
plt.ylabel(r"$P ( t )$ s$^{-1}$")

rho = np.empty(num_rho)
#power_s = np.empty(num_rho)
for i in range(0, num_rho):
    rho[i] = i*drho

power_s = spatialP(rho, D, l_t, mu_a, L_s)
power_s_approx = spatialP_approx(rho, D, l_t, mu_a, L_s)
power_s_approx2 = spatialP_approx2(rho, D, l_t, L_s)


plt.figure(2)
plt.semilogy(rho, power_s, 'r-', rho, power_s_approx, 'g-', rho, power_s_approx2, 'b-')
plt.xlabel(r"$\rho$ (mm)")
plt.ylabel(r"$P ( \rho )$ mm$^{-2}$")

plt.show()
