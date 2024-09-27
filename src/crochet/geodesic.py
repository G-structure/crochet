import igl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def calculate_geodesic_isolines(v, f, start_vertex, strip_size):
    vs = np.array([start_vertex])
    vt = np.arange(v.shape[0])

    d = igl.exact_geodesic(v, f, vs, vt)

    c = np.abs(np.sin((d / strip_size * np.pi)))
    return c, d

def plot_geodesic_isolines(v, f, c):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, color='lightgray', alpha=0.5)

    num_levels = 10
    isolines = ax.tricontour(v[:, 0], v[:, 1], v[:, 2], c, triangles=f, colors='k', linewidths=0.5, levels=num_levels)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Geodesic Isolines on Mesh')

    plt.show()

def main():
    root_folder = os.getcwd()

    v, f = igl.read_triangle_mesh(os.path.join(root_folder, "data", "bent_pipe_closed_lr.off"))

    start_vertex = 0
    strip_size = 0.2

    c, _ = calculate_geodesic_isolines(v, f, start_vertex, strip_size)
    plot_geodesic_isolines(v, f, c)
