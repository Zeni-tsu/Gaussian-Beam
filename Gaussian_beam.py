import matplotlib.pyplot as plt
from matplotlib import  cm
import numpy as np

fig = plt.figure(figsize=(8,8))

E_0 = 500
lamda = 652*10**(-9)
n=1
w_0 = 1*10**(-3)
z_R = 3.14*w_0**2*n/lamda


def gauss(z,c):
    w_z = w_0*np.sqrt(1+(z/z_R)**2) 
    
    ax = fig.add_subplot(2,2,c,projection="3d")

    X = np.arange(-0.02,0.02,0.001)
    Y = np.arange(-0.02,0.02,0.001)

    X,Y = np.meshgrid(X,Y)

    Z = E_0*(w_0/w_z)*np.exp(-(X**2+Y**2)/w_z**2)
    
    print(len(Z),len(Z[0]))

    ax.plot_surface(X,Y,Z,cmap=cm.viridis)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Amplitude")
    ax.set_title("Amplitude at z = {}m".format(z))
    plt.tight_layout()
    ax.view_init(25,-75)

gauss(10,1)
gauss(25,2)
gauss(50,3)
gauss(75,4)
plt.show()