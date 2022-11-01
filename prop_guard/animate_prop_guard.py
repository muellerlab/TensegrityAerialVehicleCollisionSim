# IMPORTS
import numpy as np
from py3dmath.py3dmath import *
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from prop_guard.design_prop_guard import prop_guard_design


class prop_guard_animator():
    def __init__(self,prop_guard:prop_guard_design) -> None:
        self.prop_guard = prop_guard
        # Attaching 3D axis to the figure
        self.fig = plt.figure()
        self.ax = p3.Axes3D(self.fig)
        pass

    def animate_scatters(self,iteration, nodePosHist, nodes, rod):
        """
        Update the data held by the scatter plot and therefore animates it.
        Args:
            iteration (int): Current iteration of the animation
            data (list): List of the data positions at each iteration.
            scatters (list): List of all the scatters (One per element)
        Returns:
            list: List of scatters (One per element) with new coordinates
        """
        for i in range(self.prop_guard.nodeNum):
            nodes[i]._offsets3d = (nodePosHist[iteration,i,0:1], nodePosHist[iteration,i,1:2], nodePosHist[iteration,i,2:])
        
        for j in range(len(self.prop_guard.rods)):
            b,e = self.prop_guard.rods[j]
            rod[j].set_data([nodePosHist[iteration,b,0],nodePosHist[iteration,e,0]], [nodePosHist[iteration,b,1],nodePosHist[iteration,e,1]])
            rod[j].set_3d_properties([nodePosHist[iteration,b,2],nodePosHist[iteration,e,2]])
        return nodes, rod


    def plot_wall(self, p:Vec3, dir, l = 0.2):
        
        if dir == 'x':
            x = [p.x,p.x,p.x,p.x]
            y = [p.y+l,p.y+l,p.y-l,p.y-l]
            z = [p.z+l,p.z-l,p.z-l,p.z+l]
        elif dir == 'y':
            x = [p.x+l,p.x+l,p.x-l,p.x-l]
            y = [p.y,p.y,p.y,p.y]
            z = [p.z+l,p.z-l,p.z-l,p.z+l]
        elif dir =='z':
            x = [p.x+l,p.x+l,p.x-l,p.x-l]
            y = [p.y+l,p.y-l,p.y-l,p.y+l]
            z = [p.z,p.z,p.z,p.z]

        verts = [list(zip(x,y,z))]
        collection = Poly3DCollection(verts,alpha=0.2)
        self.ax.add_collection3d(collection)
        return


    def animate_prop_guard(self, nodePosHist, show = False, save=False, fps = 24, name = "prop_guard_sim"):
        """
        Creates the 3D figure and animates it with the input data.
        Args:
            data (list): List of the data positions at each iteration.
            save (bool): Whether to save the recording of the animation. (Default to False).
        """


        # Initialize scatters
        nodes = [self.ax.scatter(nodePosHist[0,i,0:1], nodePosHist[0,i,1:2], nodePosHist[0,i,2:]) for i in range(self.prop_guard.nodeNum)]

        rod = []
        for b,e in self.prop_guard.rods:
            rod.append(self.ax.plot([nodePosHist[0,b,0],nodePosHist[0,e,0]], [nodePosHist[0,b,1],nodePosHist[0,e,1]], [nodePosHist[0,b,2],nodePosHist[0,e,2]], 'b-')[0])

        # Number of iterations
        iterations = nodePosHist.shape[0]

        # Setting the axes properties
        self.ax.set_xlim3d([-0.25, 0.25])
        self.ax.set_xlabel('X')

        self.ax.set_ylim3d([-0.25, 0.25])
        self.ax.set_ylabel('Y')

        self.ax.set_zlim3d([-0.25, 0.25])
        self.ax.set_zlabel('Z')

        self.ax.set_title('Prop Guard')
        # Provide starting angle for the view.
        self.ax.view_init(15, 60)
        self.ax.grid(False)
        self.ax.tick_params(axis='both', labelsize=10)
        self.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        self.ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        self.ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

        ani = animation.FuncAnimation(self.fig, self.animate_scatters, iterations, fargs=(nodePosHist, nodes,rod),
                                        interval=10, blit=False, repeat=True)

        if save:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='user'), bitrate=6000, extra_args=['-vcodec', 'libx264'])
            ani.save(name + '.mp4', writer=writer)

        if show:
            plt.show()