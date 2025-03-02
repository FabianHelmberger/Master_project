
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the spherical coordinate grid
phi = np.linspace(0, 2 * np.pi, 100)  # Azimuthal angle
theta = np.linspace(0, np.pi, 50)     # Polar angle
phi, theta = np.meshgrid(phi, theta)

# Define the height function (e.g., h = cos(theta))
h = np.cos(phi)**2

# Convert to Cartesian coordinates
R = h+1  # Sphere radius with height perturbation
x = R * np.sin(theta) * np.cos(phi)
y = R * np.sin(theta) * np.sin(phi)
z = R * np.cos(theta)

# Plot the sphere with height variation
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis', edgecolor='k')

# Labels and show
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("Height Function on a 3D Sphere")
plt.show()
