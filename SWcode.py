import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.optimize import brentq

# h = m0 * Ms * H / (2* Ku)
# eta = E/(2* Ku * V) = 1.0/4.0 - 1.0/4.0 * np.cos(2*(phi-th)) -  h * np.cos(phi)
# deta/dphi = 0.5*np.sin(2*(phi-theta)) + h*np.sin(phi) = 0
# d2eta/d2phi = np.cos(2*(phi-theta)) + h*np.cos(phi) > 0 
# m = np.cos(phi)


#h = m0 * Ms * H / (2* Ku)


### Important variables

def SW(H, theta):

    phis1=[]
    phis2=[]
    
    ### Top part
    
    for h in H:
        
        F = lambda phi : 0.5*np.sin(2*(phi-theta)) + h*np.sin(phi) 
        phi = np.linspace(0, np.pi, points)
        
        phi_initial_guess = 0.8
        
        if max(F(phi)) > 0:
                p = brentq(F, 0, phi[np.argmax(F(phi))])
        else:
            p = np.pi
    
        #plt.plot(phi, F(phi))
        #plt.show()
            
        phis1.append(p)
    phis1 = np.array(phis1,dtype='float')
    
    
    
    ### Bottom part   
            
    for h in H:
        F = lambda phi : 0.5*np.sin(2*(phi-theta)) + h*np.sin(phi) 
        phi = np.linspace(2*np.pi, np.pi, points)
        
        phi_initial_guess = 4.0
        if max(F(phi)) > 0:
                p = brentq(F, np.pi, phi[np.argmax(F(phi))])
        else:
            p = 0
    
        #plt.plot(phi, F(phi))
        #plt.show()
            
        phis2.append(p)
    phis2 = np.array(phis2, dtype='float')
    
    ### Cheat it into a histogram
    
    for i in range(len(phis1)):
        if H[i] < 0 and np.cos(phis2[i]) > np.cos(phis1[i]):
            phis1[i] = phis2[i]
            
    
    for i in range(len(phis2)):
        if H[i] > 0 and np.cos(phis2[i]) > np.cos(phis1[i]):
            phis2[i] = phis1[i]

    return phis1, phis2
    
### Figure

#
#plt.plot(H, np.cos(phis1), 'b', H, np.cos(phis2), 'r', linewidth=4)
#plt.grid()
#plt.show()

points = 200
H = np.linspace(-1.0,1.0,points)

theta0 = 45.0
theta = theta0/360.0*2.0*np.pi
h0 = 0.0
global first
first = True

phis1, phis2 = SW(H, theta)

ax = subplot(111)
subplots_adjust(bottom=0.3)

l, = plot(H, np.cos(phis1), 'b', linewidth=4)
k, = plot(H, np.cos(phis2), 'b', lw = 4)
st, = plot(h0, np.cos(phis1[ argmin(abs(H-h0)) ] ), '*', color='r', ms=20)

ax.set_ylim([-1.1,1.1])
ax.set_xlim([-1.1,1.1])
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_xlabel('h', fontsize=20)
ax.set_ylabel('m'r'$_h$',fontsize=20)
plt.grid(b=True, which='both')

axcolor = 'lightgoldenrodyellow'
axtheta = axes([0.2, 0.15, 0.65, 0.03], axisbg=axcolor)
axh = axes([0.2, 0.1, 0.65, 0.03], axisbg=axcolor)

stheta = Slider(axtheta, 'Theta', 0.1, 89.9, valinit=theta0)
sh = Slider(axh, 'Applied Field', min(H), max(H), valinit=h0)

def update(val):
    ### Plot
    theta = stheta.val
    theta = theta/360.0*2.0*np.pi
    p1, p2 = SW(H, theta)
    l.set_ydata(np.cos(p1))
    k.set_ydata(np.cos(p2))
    
    ### Star
    h = sh.val
    hh = argmin(abs(H-h)) 
    
    global p, x
    try:
        p
    except: 
        p = p1
        x = 0
        
    if p1[hh] == p2[hh] and H[hh]>0:
        p = p1
        x = 1
    elif p1[hh] == p2[hh] and H[hh]<0:
        p = p2
        x = -1
    
    if x == 0 or x==1:
        p = p1
    elif x == -1:
        p = p2
    
    st.set_xdata(H[hh])
    st.set_ydata(np.cos(p[hh]))
    draw()
    
stheta.on_changed(update)
sh.on_changed(update)

resetax = axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
def reset(event):
    stheta.reset()
    sh.reset()
button.on_clicked(reset)

show()
