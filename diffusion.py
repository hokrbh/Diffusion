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
    return c/(4.0*np.pi*c*D*t)**0.5 * np.exp(-mu_a*c*t) * (1.0 - np.exp(-radius**2.0/(4.0*c*D*t))) * \
            ( np.exp(-(l_t)**2/(4.0*c*D*t)) - np.exp(-(2.0*L_s+l_t)**2/(4.0*c*D*t)) )

def surfacePvsTimeHole(t, radius, mu_t, mu_a, L_s, h):
    return c/(4.0*np.pi*c*D*t)**0.5 * np.exp(-mu_a*c*t) * (1.0 - np.exp(-radius**2.0/(4.0*c*D*t))) * \
            ( np.exp(-(l_t+h)**2/(4.0*c*D*t)) - np.exp(-(2.0*L_s+l_t+h)**2/(4.0*c*D*t)) )

def longTimeP(t, radius, mu_t, mu_a, L_s):
    return (np.pi*radius**2*c)/((4.0*np.pi*D*c*t)**1.5) * (L_s*(L_s+l_t))/(D*c*t)*np.exp(-mu_a*c*t)

def spatialP(rho, D, l_t, mu_a, L_s):
    return 1.0/(4.0*np.pi*D) * ( np.exp(-np.sqrt(mu_a/D*(rho**2 + l_t**2)))/np.sqrt(rho**2 + l_t**2) - np.exp(-np.sqrt(mu_a/D*(rho**2 + (2.0*L_s+l_t)**2)))/np.sqrt(rho**2 + (2.0*L_s+l_t)**2) )

def spatialP_approx(rho, D, l_t, mu_a, L_s):
    return np.sqrt(mu_a/D) * L_s*(L_s+l_t)*np.exp(-np.sqrt(mu_a/D)*rho)/(2.0*np.pi*D*rho**2)

def spatialP_approx2(rho, D, l_t, L_s):
    return L_s*(L_s+l_t)/(2.0*np.pi*D*rho**3)

def bh_int(x):
    return 0.5*np.sqrt(np.pi)*(1.0+x)*np.exp(-x)

def avg_t_theory(h, radius, D, l_t, mu_a, L_s):
    return 1.0/(2.0*np.sqrt(np.pi)*c*mu_a**(3.0/2.0)*np.sqrt(D)) * ( bh_int((l_t+h)*np.sqrt(mu_a/D)) - bh_int((2.0*L_s+l_t+h)*np.sqrt(mu_a/D)) - bh_int(np.sqrt(mu_a/D*((l_t+h)**2+radius**2))) + bh_int(np.sqrt(mu_a/D*((2.0*L_s+l_t+h)**2+radius**2))) )

def integrand(time, radius, mu_t, mu_a, L_s, h):
    return time*surfacePvsTimeHole(time, radius, mu_t, mu_a, L_s, h)

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

plt.figure(2)
plt.loglog(time, surfacePvsTimeHole(time, radius, mu_t, mu_a, L_s, 0.001), 'r-',\
        time, surfacePvsTimeHole(time, radius, mu_t, mu_a, L_s, 0.01), 'g-',\
        time, surfacePvsTimeHole(time, radius, mu_t, mu_a, L_s, 0.1), 'b-',\
        time, surfacePvsTimeHole(time, radius, mu_t, mu_a, L_s, 1.0), 'r-')
plt.xlabel(r"$t$ (ps)")
plt.ylabel(r"$P ( t )$ W")
plt.ylim([1E-10,1])

h_max = 0.2
num_h = 11
dh = h_max/(num_h-1)

depths = np.arange(0, h_max+dh, dh)
avg_t = np.empty(num_h)
for i in range(0, num_h):
    powerTemp = surfacePvsTimeHole(time, radius, mu_t, mu_a, L_s, depths[i])
    result = integrate.quad(integrand, 0, np.inf, args=(radius, mu_t, mu_a, L_s, depths[i]))
    avg_t[i] = result[0]

plt.figure(3)
plt.plot(depths, avg_t, 'rx', depths, avg_t_theory(depths, radius, D, l_t, mu_a, L_s), 'bo')
plt.xlabel(r"$h$ (mm)")
plt.ylabel(r"$\langle t \rangle$ ps")

rho = np.empty(num_rho)
#power_s = np.empty(num_rho)
for i in range(0, num_rho):
    rho[i] = i*drho

power_s = spatialP(rho, D, l_t, mu_a, L_s)
power_s_approx = spatialP_approx(rho, D, l_t, mu_a, L_s)
power_s_approx2 = spatialP_approx2(rho, D, l_t, L_s)

plt.figure(4)
plt.semilogy(rho, power_s, 'r-', rho, power_s_approx, 'g-', rho, power_s_approx2, 'b-')
plt.xlabel(r"$\rho$ (mm)")
plt.ylabel(r"$P ( \rho )$ mm$^{-2}$")

plt.show()
