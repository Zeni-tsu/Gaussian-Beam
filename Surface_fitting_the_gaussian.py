from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,2,1,projection="3d")

from PIL import Image

img = Image.open(r"C:\Users\baruv\Downloads\\Z4.jpg")
imgWidth, imgHeight = img.size
img = img.convert("RGBA")
imgdata = img.getdata()

x_pos = 0
y_pos = 1

pixel_value = []
x_data = []
y_data = []

for item in imgdata:
    if (x_pos) == imgWidth:
        x_pos = 1
        y_pos += 1
    else:
        x_pos += 1

    pixel_value.append(item[0])
    x_data.append(x_pos)
    y_data.append(y_pos)



ax.plot3D(x_data,y_data,pixel_value,"b,",alpha=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Amplitude")
ax.set_title("Amplitude of a Gaussian Beam (4th plane)")
ax.view_init(15,-50)

def function(data, a, b, c,d):
    x = data[0]
    y = data[1]
    return a * np.exp(-(x-b)**2/(2*c**2)-(y-d)**2/(2*c**2))

model_x_data = np.linspace(min(x_data)-250, max(x_data)+250, 300)
model_y_data = np.linspace(min(y_data)-250, max(y_data)+250, 300)

X, Y = np.meshgrid(model_x_data, model_y_data)

parameters, covariance = curve_fit(function, [x_data, y_data], pixel_value,p0=[2,1,1,1],maxfev = 5000)

Z = function(np.array([X, Y]), *parameters)

ax = fig.add_subplot(1,2,2,projection="3d")

ax.plot_surface(X, Y, Z,cmap=cm.viridis)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Amplitude")
ax.set_title("Surface fit of amplitude(4th plane)")
ax.view_init(15,-50)
print(parameters)
print(img.size)


plt.show()
