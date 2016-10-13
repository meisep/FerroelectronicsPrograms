import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D

a = 1.0
b = 1.0
c = 1.0

alpha = 90.0
beta = 90.0
gamma = 90.0

alpha = alpha/360.*2.*np.pi
beta = beta/360.*2.*np.pi
gamma = gamma/360.*2.*np.pi

def RS(a,b,c,alpha,beta,gamma):
    
    v = np.sqrt(1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 + 2*cos(alpha)*np.cos(beta)*np.cos(gamma))
    ar = [a, b*np.cos(gamma), c*np.cos(beta)]
    br = [0.0, b*np.sin(gamma), c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma)) / np.sin(gamma)]
    cr = [0.0, 0.0, c*v/np.sin(gamma)]
    
    V = np.dot(ar,np.cross(br,cr))
    af = np.cross(br,cr)/V
    bf = np.cross(cr,ar)/V
    cf = np.cross(ar,br)/V
    
    width = 10
    M = []
    for i in range(-width/2, width):
        for j in range(-width/2, width):
            for k in range(-width/2, width):
                temp = i*af + j*bf + k*cf
                M.append(temp)
    
    M = np.array(M)
    
    for i in np.arange(len(M)):
        if np.any(M[i,:] < -0.1):
            M[i,:] = [NaN, NaN, NaN]
        elif np.any(M[i,:] > 3.1):
            M[i,:] = [NaN, NaN, NaN]
        else:
            pass
     
    ar = np.array(ar)
    br = np.array(br)
    cr = np.array(cr)      
    R = []
    for i in range(-width/2, width):
        for j in range(-width/2, width):
            for k in range(-width/2, width):
                temp = i*ar + j*br + k*cr
                R.append(temp)
    
    R = np.array(R)
    
    for i in np.arange(len(R)):
        if np.any(R[i,:] < -0.1):
            R[i,:] = [NaN, NaN, NaN]
        elif np.any(R[i,:] > 3.1):
            R[i,:] = [NaN, NaN, NaN]
        else:
            pass
        
    return M, R

global l
M, R = RS(a,b,c,alpha,beta,gamma)

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax2 = fig.add_subplot(1, 2, 1, projection='3d')
subplots_adjust(bottom=0.25)
ax.set_xlim3d(0, 3)
ax.set_ylim3d(0, 3)
ax.set_zlim3d(0, 3)
l = ax.scatter(M[:,0],M[:,1],M[:,2], s=80, c='blue')
ax.view_init(elev=10., azim=0)
ax2.set_xlim3d(0, 3)
ax2.set_ylim3d(0, 3)
ax2.set_zlim3d(0, 3)
m = ax2.scatter(R[:,0],R[:,1],R[:,2], s=80, c='red')
ax2.view_init(elev=10., azim=0)


axcolor = 'lightgoldenrodyellow'
axa = axes([0.13, 0.2, 0.3, 0.03], axisbg=axcolor)
axb = axes([0.13, 0.15, 0.3, 0.03], axisbg=axcolor)
axc = axes([0.13, 0.1, 0.3, 0.03], axisbg=axcolor)

sa = Slider(axa, 'a', 0.5, 2.0, valinit=a)
sb = Slider(axb, 'b', 0.5, 2.0, valinit=b)
sc = Slider(axc, 'c', 0.5, 2.0, valinit=c)

axalpha = axes([0.55, 0.2, 0.3, 0.03], axisbg=axcolor)
axbeta = axes([0.55, 0.15, 0.3, 0.03], axisbg=axcolor)
axgamma = axes([0.55, 0.1, 0.3, 0.03], axisbg=axcolor)

salpha = Slider(axalpha, ''r'$\alpha$', 45.0, 120.0, valinit=alpha/(2.*np.pi)*360)
sbeta = Slider(axbeta, ''r'$\beta$', 45.0, 120.0, valinit=beta/(2.*np.pi)*360)
sgamma = Slider(axgamma, ''r'$\gamma$', 45.0, 120.0, valinit=gamma/(2.*np.pi)*360)

def update(val):
    a = sa.val
    b = sb.val
    c = sc.val
    
    alpha = salpha.val
    beta = sbeta.val
    gamma = sgamma.val
    
    alpha = alpha/360.*2.*np.pi
    beta = beta/360.*2.*np.pi
    gamma = gamma/360.*2.*np.pi
    
    M, R = RS(a,b,c,alpha,beta,gamma)
    ax.cla()
    l = ax.scatter(M[:,0],M[:,1],M[:,2], s=80,c='b')
    ax2.cla()
    m = ax2.scatter(R[:,0],R[:,1],R[:,2], s=80, c='r')
    
    ax.set_xlim3d(0, 3)
    ax.set_ylim3d(0, 3)
    ax.set_zlim3d(0, 3)
    
    plt.setp( ax.get_xticklabels(), visible=False)
    plt.setp( ax.get_yticklabels(), visible=False)
    plt.setp( ax.get_zticklabels(), visible=False)
    ax.set_xlabel('a*', fontsize=20)
    ax.set_ylabel('b*', fontsize=20)
    ax.set_zlabel('c*', fontsize=20)
    ax.set_title('Reciprocal Space',fontsize=20)
    
    plt.setp( ax2.get_xticklabels(), visible=False)
    plt.setp( ax2.get_yticklabels(), visible=False)
    plt.setp( ax2.get_zticklabels(), visible=False)
    ax2.set_xlabel('a', fontsize=20)
    ax2.set_ylabel('b', fontsize=20)
    ax2.set_zlabel('c', fontsize=20)
    ax2.set_title('Real Space',fontsize=20)
    
    draw()
sa.on_changed(update)
sb.on_changed(update)
sc.on_changed(update)
salpha.on_changed(update)
sgamma.on_changed(update)
sbeta.on_changed(update)

plt.setp( ax.get_xticklabels(), visible=False)
plt.setp( ax.get_yticklabels(), visible=False)
plt.setp( ax.get_zticklabels(), visible=False)
ax.set_xlabel('a*', fontsize=20)
ax.set_ylabel('b*', fontsize=20)
ax.set_zlabel('c*', fontsize=20)
ax.set_title('Reciprocal Space',fontsize=20)

plt.setp( ax2.get_xticklabels(), visible=False)
plt.setp( ax2.get_yticklabels(), visible=False)
plt.setp( ax2.get_zticklabels(), visible=False)
ax2.set_xlabel('a', fontsize=20)
ax2.set_ylabel('b', fontsize=20)
ax2.set_zlabel('c', fontsize=20)
ax2.set_title('Real Space',fontsize=20)

resetax = axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Cubic', color=axcolor, hovercolor='0.975')
def reset(event):
    sa.reset()
    sb.reset()
    sc.reset()
    salpha.reset()
    sbeta.reset()
    sgamma.reset()
button.on_clicked(reset)


show()    


