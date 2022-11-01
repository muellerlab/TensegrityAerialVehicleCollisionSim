# IMPORTS
import numpy as np
from py3dmath.py3dmath import *
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from tensegrity.design_tensegrity import tensegrity_design

class tensegrity_animator():
    def __init__(self,tensegrity:tensegrity_design) -> None:
        self.tensegrity = tensegrity
        self.nodeNum = tensegrity.nodeNum
        self.propNum = 4
        self.numRod = tensegrity.numRod
        self.numString = tensegrity.numString
        self.rods = tensegrity.rods
        self.strings = tensegrity.strings

        # Attaching 3D axis to the figure
        self.fig = plt.figure()
        self.ax = p3.Axes3D(self.fig)
        pass

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

    def animate_scatters(self, iteration, nodePosHist, nodes, props, rod, string):
        """
        Update the data held by the scatter plot and therefore animates it.
        Args:
            iteration (int): Current iteration of the animation
            data (list): List of the data positions at each iteration.
            scatters (list): List of all the scatters (One per element)
        Returns:
            list: List of scatters (One per element) with new coordinates
        """
        for i in range(self.nodeNum-self.propNum):
            nodes[i]._offsets3d = (nodePosHist[iteration,i,0:1], nodePosHist[iteration,i,1:2], nodePosHist[iteration,i,2:])

        for i in range(self.propNum):
            props[i]._offsets3d = (nodePosHist[iteration,(self.nodeNum-self.propNum)+i,0:1], nodePosHist[iteration,(self.nodeNum-self.propNum)+i,1:2], nodePosHist[iteration,(self.nodeNum-self.propNum)+i,2:])

        for j in range(self.numRod):
            b,e = self.rods[j]
            rod[j].set_data([nodePosHist[iteration,b,0],nodePosHist[iteration,e,0]], [nodePosHist[iteration,b,1],nodePosHist[iteration,e,1]])
            rod[j].set_3d_properties([nodePosHist[iteration,b,2],nodePosHist[iteration,e,2]])

        for k in range(self.numString):
            b,e = self.strings[k]
            string[k].set_data([nodePosHist[iteration,b,0],nodePosHist[iteration,e,0]], [nodePosHist[iteration,b,1],nodePosHist[iteration,e,1]])
            string[k].set_3d_properties([nodePosHist[iteration,b,2],nodePosHist[iteration,e,2]])

        return nodes, rod, string

    def animate_tensegrity(self, nodePosHist, show=False, save=False, fps = 24, name="TensegritySim"):
        """
        Creates the 3D figure and animates it with the input data.
        Args:
            data (list): List of the data positions at each iteration.
            save (bool): Whether to save the recording of the animation. (Default to False).
        """

        # Initialize scatters
        nodes = [self.ax.scatter(nodePosHist[0,i,0:1], nodePosHist[0,i,1:2], nodePosHist[0,i,2:], color = (43/255,255/255,255/255,0.8)) for i in range(self.nodeNum-self.propNum)]
        props = [self.ax.scatter(nodePosHist[0,(self.nodeNum-self.propNum)+i,0:1], nodePosHist[0,(self.nodeNum-self.propNum)+i,1:2], nodePosHist[0,(self.nodeNum-self.propNum)+i,2:], color=(247/255,147/255,29/255,0.9)) for i in range(self.propNum)]


        rod = []
        for b,e in self.rods:
            rod.append(self.ax.plot([nodePosHist[0,b,0],nodePosHist[0,e,0]], [nodePosHist[0,b,1],nodePosHist[0,e,1]], [nodePosHist[0,b,2],nodePosHist[0,e,2]], '-', linewidth=3, color=(3/255,107/255,251/255,0.8))[0])

        string = [] 
        for b,e in self.strings:
            string.append(self.ax.plot([nodePosHist[0,b,0],nodePosHist[0,e,0]], [nodePosHist[0,b,1],nodePosHist[0,e,1]], [nodePosHist[0,b,2],nodePosHist[0,e,2]], '--', linewidth=2, color=(251/255,17/255,19/255,0.8))[0])

        # Number of iterations
        iterations = nodePosHist.shape[0]

        # Setting the axes properties
        self.ax.set_xlim3d([-0.25, 0.25])
        self.ax.set_xlabel('X')

        self.ax.set_ylim3d([-0.25, 0.25])
        self.ax.set_ylabel('Y')

        self.ax.set_zlim3d([-0.25, 0.25])
        self.ax.set_zlabel('Z')

        self.ax.set_title('Tensegrity')
        # Provide starting angle for the view.
        self.ax.view_init(10, 75)
        self.ax.grid(False)
        self.ax.tick_params(axis='both', labelsize=10)
        self.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        self.ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        self.ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

        ani = animation.FuncAnimation(self.fig, self.animate_scatters, iterations, fargs=(nodePosHist, nodes, props, rod, string),
                                        interval=10, blit=False, repeat=True)

        if save:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='user'), bitrate=12000, extra_args=['-vcodec', 'libx264'])
            ani.save(name+'.mp4', writer=writer)

        if show:
            plt.show()