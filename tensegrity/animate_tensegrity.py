# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from tensegrity.design_tensegrity import tensegrity_design



class tensegrity_animator():
    def __init__(self,tensegrity:tensegrity_design) -> None:
        self.tensegrity = tensegrity
        self.nodeNum = tensegrity.nodeNum
        self.numRod = tensegrity.numRod
        self.numString = tensegrity.numString
        self.rods = tensegrity.rods
        self.strings = tensegrity.strings
        pass

    def animate_scatters(self,iteration, nodePosHist, nodes, rod, cable):
        """
        Update the data held by the scatter plot and therefore animates it.
        Args:
            iteration (int): Current iteration of the animation
            data (list): List of the data positions at each iteration.
            scatters (list): List of all the scatters (One per element)
        Returns:
            list: List of scatters (One per element) with new coordinates
        """
        for i in range(self.nodeNum):
            nodes[i]._offsets3d = (nodePosHist[iteration,i,0:1], nodePosHist[iteration,i,1:2], nodePosHist[iteration,i,2:])
        
        for j in range(self.numRod):
            b,e = self.rods[j]
            rod[j].set_data([nodePosHist[iteration,b,0],nodePosHist[iteration,e,0]], [nodePosHist[iteration,b,1],nodePosHist[iteration,e,1]])
            rod[j].set_3d_properties([nodePosHist[iteration,b,2],nodePosHist[iteration,e,2]])

        for k in range(self.numString):
            b,e = self.strings[k]
            cable[k].set_data([nodePosHist[iteration,b,0],nodePosHist[iteration,e,0]], [nodePosHist[iteration,b,1],nodePosHist[iteration,e,1]])
            cable[k].set_3d_properties([nodePosHist[iteration,b,2],nodePosHist[iteration,e,2]])

        return nodes, rod, cable

    def animate_tensegrity(self,nodePosHist, show=False, save=False, name="TensegritySim"):
        """
        Creates the 3D figure and animates it with the input data.
        Args:
            data (list): List of the data positions at each iteration.
            save (bool): Whether to save the recording of the animation. (Default to False).
        """
        # Attaching 3D axis to the figure
        fig = plt.figure()
        ax = p3.Axes3D(fig)

        # Initialize scatters
        nodes = [ax.scatter(nodePosHist[0,i,0:1], nodePosHist[0,i,1:2], nodePosHist[0,i,2:]) for i in range(self.nodeNum)]

        rod = []
        for b,e in self.rods:
            rod.append(ax.plot([nodePosHist[0,b,0],nodePosHist[0,e,0]], [nodePosHist[0,b,1],nodePosHist[0,e,1]], [nodePosHist[0,b,2],nodePosHist[0,e,2]], 'b-')[0])

        cable = [] 
        for b,e in self.strings:
            cable.append(ax.plot([nodePosHist[0,b,0],nodePosHist[0,e,0]], [nodePosHist[0,b,1],nodePosHist[0,e,1]], [nodePosHist[0,b,2],nodePosHist[0,e,2]], 'r-')[0])

        # Number of iterations
        iterations = nodePosHist.shape[0]

        # Setting the axes properties
        ax.set_xlim3d([-0.25, 0.25])
        ax.set_xlabel('X')

        ax.set_ylim3d([-0.25, 0.25])
        ax.set_ylabel('Y')

        ax.set_zlim3d([-0.25, 0.25])
        ax.set_zlabel('Z')

        ax.set_title('Tensegrity')
        # Provide starting angle for the view.
        ax.view_init(20, 60)

        ani = animation.FuncAnimation(fig, self.animate_scatters, iterations, fargs=(nodePosHist, nodes, rod, cable),
                                        interval=10, blit=False, repeat=True)

        if save:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=30, metadata=dict(artist='Clark'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
            ani.save(name+'.mp4', writer=writer)

        if show:
            plt.show()