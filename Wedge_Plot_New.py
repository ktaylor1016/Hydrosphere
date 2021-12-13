
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection


# read in data file
df = np.loadtxt('/Users/karleetaylor/desktop/T1M10Rb1.txt',delimiter=" ", skiprows=1)
depth=df[:,0]
phase=df[:,5]
T=df[:,6]
P=df[:,7]


# Find inflection points of phase graph --- INCLUDE IN OUTPUT FILE
from numpy import diff
# Find derivatives of graph
x = depth
y = phase
dydx = diff(y)/diff(x)
# Find index of non-zero points
inflection_pts = np.where(dydx != 0) 
# Find x value (depth) that corresponds to index
phase_change = []
for i in inflection_pts:
    inflection_depth = depth[i]
    phase_change.append(inflection_depth)  
phase_change = np.append(phase_change, depth[0]) # add top and bottom of hydrosphere
phase_change = np.append(phase_change, depth[99])
phase_change = np.sort(phase_change) # list in ascending order


# Create dictionary with radius (value) and phase name (key)
phase_names_dict={'Water':0,'Ice Ih':1,'Ice II':2, 'Ice III':3,'Ice IV':4,'Ice V':5,'Ice VI':6,'Ice VII':7}
key_list = list(phase_names_dict.keys())
val_list = list(phase_names_dict.values())
data = {}
keys = [] # keys are phase number, then phase name
values = phase_change.tolist() # value is radius/depth
phase_numbers=[]
for i in range(len(phase_change)):
    index = np.where(depth == phase_change[i]) # get index of this depth
    phase_number = phase[index] # get phase of this depth
    phase_numbers.append(phase_number) # create list of keys with phase #  
for i in range(len(phase_change)):
    position = val_list.index(phase_numbers[i])
    phase_name=key_list[position] # call key from phase_names_dict based on value (phase number)
    keys.append(phase_name)  
for i in range(len(keys)):
    data[keys[i]] = values[i]
phase_names=list(data.keys())

# GRAPH PLANETARY WEDGE (Python)

# Starting point for each wedge
x = [0,0,0,0,0]
y = [0,0,0,0,0]
  
# THICKNESS OF LAYERS
r_b=0.6*6.36e+06 #radius of core
# Radius of each wedge (thickness of layers)
radius=[]
for i in range(0,5):
    radius_i = (phase_change[0]-phase_change[i]+r_b)/1000000
    radius.append(radius_i)
  
# start angle of the wedge
start_angle = [75,75,75,75,75]
  
# end angle of the wedge
end_angle = [105,105,105,105,105]

# color value of the wedges
import matplotlib.colors as mcolors
colors = ["aqua", "navy", "dodgerblue","mediumpurple","saddlebrown"]

labels = ['Water', 'Ice Ih', 'Ice VI', 'Ice VII', 'Silicate Core']

patches = []
for x1, y1, r, t1, t2 in zip(x, y, radius, start_angle, end_angle):
    wedge = Wedge((x1, y1), r, t1, t2)
    patches.append(wedge)


fig, ax = plt.subplots()
p = PatchCollection(patches, alpha=0.8, color=colors, label=labels)
ax.add_collection(p)
ax.set_xlim(-1,1)
ax.set_ylim(2.5,3.85)
plt.ylabel('Depth (1000 km)')
fig.suptitle('Planetary Profile')

#add legend
ax.legend(patches, labels, loc=4)

plt.show()

fig.savefig('WedgePlotTest.png')

# colors 
# legend