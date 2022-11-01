import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable, get_cmap
import matplotlib.colors as colors
from visualization_helper.helper import LinearNDInterpolatorExt, plot_stacked_bar
from scipy.integrate import odeint, solve_ivp
import pickle
import seaborn as sns
import pandas as pd
sns.set_theme()
from scipy import stats

# Turn of warning to avoid the later matplotlib versions complain about setting up ticks with a traditional way.
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def analyze_and_plot_data(tensegrityData, propGuardData, scaleText):
    """
    Function to analyze and plot the graph.
    """
    fig = plt.figure(figsize=(20, 11), dpi=100)
    n = tensegrityData.shape[0]
    ratioData = np.zeros((n,4))
    pointsXYZ = tensegrityData[:,0:3] #scattered sample points in x-y-z cartesian coordinate
    
    pointsPolar = np.zeros((tensegrityData.shape[0],2)) #scattered sample points in polar coordinate
    pointsPolar[:,0] = np.arctan2(pointsXYZ[:,1],pointsXYZ[:,0])
    pointsPolar[:,1] = np.arccos(pointsXYZ[:,2])
    values = (propGuardData[:,6]/tensegrityData[:,6]).squeeze()

    ratioData[:,0:3] = pointsXYZ
    ratioData[:,3] = values
    linInterExt= LinearNDInterpolatorExt(pointsPolar, values) #A linear interpolator/extrapolator

    # Interpoloate/extrapolate scattered datapoints to a polar grid
 
    n_theta = 100 # number of values for theta
    n_phi = 100  # number of values for phi
    theta, phi = np.mgrid[np.pi/2:np.pi:n_theta*1j, 0.0:np.pi/2:n_phi*1j]

    # Prepare grid data
    x_g = np.sin(phi) * np.cos(theta)
    y_g = np.sin(phi) * np.sin(theta)
    z_g = np.cos(phi)
    v_g = np.zeros((n_theta, n_phi)) # value grid
    for i in range(n_theta):
        for j in range(n_phi):
            polarCoord = np.array([theta[i,j],phi[i,j]])
            v_g[i,j] = linInterExt(polarCoord)
    
    print("scale=",scaleText)
    print("meanPropGuardStress=", np.mean(propGuardData[:,6]))
    print("meanTensegrity=", np.mean(tensegrityData[:,6]))
    print("maxPropGuardStress=", np.max(propGuardData[:,6]))
    print("maxTensegrity=", np.max(tensegrityData[:,6]))

    """
    Fig top: max stress scatter plot for both tensegrity and prop guard
    """
    minStress = min([np.min(propGuardData[:,6]), np.min(propGuardData[:,6])])
    maxStress = min([np.max(propGuardData[:,6]), np.max(propGuardData[:,6])])
    normStress = plt.Normalize(minStress, maxStress)

    ax0 = fig.add_subplot(2, 2, 1, projection='3d')
    cs0 = ax0.scatter(tensegrityData[:,0], tensegrityData[:,1], tensegrityData[:,2], c = tensegrityData[:,6], cmap = "cool", lw=0, s=25, norm = normStress)
    ax0.set_xlim3d([-1.0, 0.2])
    ax0.set_xlabel('Body X',fontsize=axisLabelSize)

    ax0.set_ylim3d([-0.2, 1.0])
    ax0.set_ylabel('Body Y',fontsize=axisLabelSize)

    ax0.set_zlim3d([-0.2, 1.0])
    ax0.set_zlabel('Body Z',fontsize=axisLabelSize)

    ax0.set_title('Tensegrity Maximum Stress [Pa]')
    ax0.view_init(25, 135)
    ax0.set_facecolor('white')
    ax0.locator_params(nbins=4)
    ax0.grid(False)

    ax1 = fig.add_subplot(2, 2, 2, projection='3d')
    cs1 = ax1.scatter(propGuardData[:,0], propGuardData[:,1], propGuardData[:,2], c=propGuardData[:,6], cmap = "cool", lw=0, s=25, norm = normStress)
    ax1.set_xlim3d([-1.0, 0.2])
    ax1.set_xlabel('Body X',fontsize=axisLabelSize)

    ax1.set_ylim3d([-0.2, 1.0])
    ax1.set_ylabel('Body Y',fontsize=axisLabelSize)

    ax1.set_zlim3d([-0.2, 1.0])
    ax1.set_zlabel('Body Z',fontsize=axisLabelSize)

    ax1.set_title('Prop-guard Maximum Stress [Pa]')
    ax1.view_init(25, 135)
    ax1.set_facecolor('white')
    ax1.locator_params(nbins=4)
    ax1.grid(False)
    cbar = fig.colorbar(cs0, ax=[ax0, ax1],orientation='horizontal', pad=0.03, fraction=0.08, aspect=30)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.xaxis.get_offset_text().set(size=12)

    """
    Fig bottom: stress ratio plot
    """

    colorMap = plt.cm.spring  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [colorMap(i) for i in range(colorMap.N)]
    # create the new map
    customCMap = colors.LinearSegmentedColormap.from_list('Custom', cmaplist, colorMap.N)
    # define the bins and normalize

    bounds = np.linspace(np.min(ratioData[:,3]), np.max(ratioData[:,3]), 6)
    boundNorm = colors.BoundaryNorm(bounds, customCMap.N, extend='both')
    contNorm = colors.Normalize(vmin= np.min(ratioData[:,3]), vmax = np.max(ratioData[:,3]))

    ax2 = fig.add_subplot(2, 1, 2, projection='3d')
    scamap = ScalarMappable(cmap=customCMap,norm=boundNorm)
    fcolors = scamap.to_rgba(v_g)
    heatMap = ax2.plot_surface(x_g,y_g,z_g, rstride=1, cstride=1, facecolors=fcolors, linewidth=1,shade=False) 

    cs1 = ax2.scatter(ratioData[:,0], ratioData[:,1], ratioData[:,2], c = ratioData[:,3], cmap = customCMap, edgecolors='black', linewidths=0.5, s=5, alpha=0.3, norm = boundNorm)
    cbar1 = fig.colorbar(scamap, shrink=0.67, aspect=16.7)

    ax2.set_xlim3d([-1.0, 0.2])
    ax2.set_xlabel('Body X',fontsize=axisLabelSize)

    ax2.set_ylim3d([-0.2, 1.0])
    ax2.set_ylabel('Body Y',fontsize=axisLabelSize)

    ax2.set_zlim3d([-0.2, 1.0])
    ax2.set_zlabel('Body Z',fontsize=axisLabelSize)

    ax2.view_init(25, 135)
    ax2.set_facecolor('white')
    ax2.set_title('Scale:'+scaleText+'Ratio of Maximum Stress: Prop-guard/Tensegrity [Pa/Pa]')
    ax2.locator_params(nbins=4)
    return fig, ratioData

# Setup plot
labelSize = 50
axisLabelSize = 12
plotEachScale = True
figList = []
scalePressStress = True # use the experiments that scale the pre-tension based on mass
folderName = "AnalysisResultPaper/"
filePrefix = "result_x10_"
scaleTextList = ["n1","n075","n05","n025","0","025","05","075","1"]
ratioPlotType = "line"

ratioLevel = [2**(i) for i in range(-1,3)]
sampleSize = len(scaleTextList)
ratioCount = np.zeros((sampleSize,len(ratioLevel)+1)) # 3 different ration categories, 0~1,1~2,2+

for i in range(sampleSize):
    scaleText = scaleTextList[i]
    file_O = open(folderName+filePrefix+scaleText+".pickle", 'rb')
    data_O = pickle.load(file_O)
    file_O.close()

    if plotEachScale:
        fig, ratioData = analyze_and_plot_data(data_O["tensegrity"], data_O["prop_guard"],scaleText)
    else:
        tensegrityData = data_O["tensegrity"]
        propGuardData = data_O["prop_guard"]
        n = tensegrityData.shape[0]
        ratioData = np.zeros((n,4))
        ratioData[:,0:3] = tensegrityData[:,0:3]
        ratioData[:,3] = (propGuardData[:,6]/tensegrityData[:,6]).squeeze()
    
    totalNum = int(ratioData.shape[0])
    remainedData = ratioData
    for j in range(len(ratioLevel)):
        dataInRange = remainedData[np.where(remainedData[:,3]<=ratioLevel[j])]
        ratioCount[i,j] = dataInRange.shape[0]/totalNum
        remainedData = remainedData[np.where(remainedData[:,3]>ratioLevel[j])]
        if remainedData.shape[0] <=0:
            ratioCount[i,j+1:] = 0
            break
    ratioCount[i,-1] = remainedData.shape[0]/totalNum

# Stacked Box Ratio Plot for scale analysis 
series_labels = []
leftBorder = 0
for i in range(len(ratioLevel)):
    rightBorder = ratioLevel[i]
    series_labels.append('Ratio of Max Stress $\in$ (' + str(leftBorder)+','+str(rightBorder)+')')
    leftBorder = rightBorder
series_labels.append('Ratio of Max Stress >' + str(leftBorder))
r = [i for i in range(sampleSize)]
names = ('$10^{-1}$','$10^{-3/4}$','$10^{-1/2}$','$10^{-1/4}$','$10^0$','$10^{1/4}$','$10^{1/2}$','$10^{3/4}$','$10^1$')

if ratioPlotType == "box":
    fig = plt.figure(figsize=(30, 25), dpi=300)
    raw_data = dict()
    for j in range(len(ratioLevel)+1):
        raw_data[j] = ratioCount[:,j].squeeze()
    df = pd.DataFrame(raw_data)
    bars = []
    for key in raw_data:
        bars.append(100*df[key])
    barWidth = 0.85
    plot_stacked_bar(ratioCount.T, series_labels, category_labels=None, 
                        show_values=False, value_format="{}", y_label=None, 
                        colors=None, grid=True, reverse=False)

    plt.tick_params(labelsize=labelSize)
    plt.legend(fontsize=labelSize)
    plt.xticks(r, names)
    plt.xlabel("Scale Factor",fontsize=labelSize)
    plt.ylabel("Sample Percentage",fontsize=labelSize)

if ratioPlotType == "line":
    fig = plt.figure(figsize=(30, 22), dpi=300)
    ax = fig.add_subplot(111)
    markList = ["o","v","^","s","p","d"]
    for i in range(ratioCount.shape[1]):
        ax.plot(r, ratioCount[:,i], label = series_labels[i], marker=markList[i], linestyle='-', linewidth=3, markersize=labelSize)
    vals = ax.get_yticks()
    ax.set_xticks(r, names)
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

    ax.set_xlabel("Scale Factor",fontsize=labelSize)
    ax.set_ylabel("Sample Percentage",fontsize=labelSize)
    ax.tick_params(labelsize=labelSize)
    ax.legend(fontsize=labelSize)
    
    # plt.xticks(r, names)
    plt.xlabel("Scale Factor",fontsize=labelSize)
    plt.ylabel("Sample Percentage",fontsize=labelSize)
    plt.savefig("Scale Analysis.svg",  format = 'svg')
    plt.savefig("Scale Analysis.png",  format = 'png')

# Show graphic
plt.show()