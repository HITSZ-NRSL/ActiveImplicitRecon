import os
import numpy as np
import matplotlib.pyplot as plt

method_list = [
        "Occlusion Aware", 
        "Unobserved Voxel", 
        "Rear Side Voxel", 
        "Rear Side Entropy", 
        "Proximity Count", 
        "Area Factor", 
        "Average Entropy", 
    ]

linestyle = '-'
linewidth = 2
marker = 'o'
markersize = 3

# x = np.array([1, 2, 4, 6, 10])
# x_entropy = np.arange(21)

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_entropy = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
type = "armadillo"

# draw entropy

if type == "bunny":
    y0 = np.loadtxt("./exp/ral/bunny/entropy.txt")[x_entropy[1:]-1]
    
elif type == "armadillo":
    y0 = np.loadtxt("./exp/ral/armadillo/entropy.txt")[x_entropy[1:]-1]
    
elif type == "dragon":
    y0 = np.loadtxt("./exp/ral/dragon/entropy.txt")[x_entropy[1:]-1]

y0 = np.concatenate([np.ones(1), y0])

yx_list = []
for method in range(7):
    yx = [1]
    path = os.path.join("./exp/nbv_res", str(method + 1), "pcl_" + type)
    for file in sorted(os.listdir(path), key=lambda s: int(s.split('.')[0].split('_')[2])):
        if file.startswith("octomap"):
            data = float(file.split('_')[4][:8])
            yx.append(data)
    yx = np.array(yx)
    yx_list.append(yx)

plt.figure()
plt.plot(x_entropy, y0, label="Ours", linewidth = linewidth, marker = marker, markersize = markersize)
for num, method in enumerate(method_list):
    yx = yx_list[num]    
    plt.plot(x_entropy, yx, label=method, 
        linestyle = linestyle, linewidth = linewidth, marker = marker, markersize = markersize)

plt.xlim(0, 21)
plt.xticks(np.linspace(0,21,22,endpoint=True))
plt.ylim(0.0,1.0)
plt.yticks(np.linspace(0,1,5,endpoint=True))

legend_font = {"family" : "Times New Roman", "size" : 11}
plt.legend(prop = legend_font, bbox_to_anchor=(1.0, 0.35), loc = "center right")

plt.ylabel("Entropy in map", fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.xlabel("Reconstruction Step", fontdict={'family' : 'Times New Roman', 'size'   : 16})
# plt.title('Etropy Evaluation')


# draw surface coverage

if type == "bunny":
    y0 = np.loadtxt("./exp/ral/bunny/sc.txt")[x-1]
    y1 = np.array([0.440, 0.607, 0.810, 0.915, 0.924]) 
    y2 = np.array([0.440, 0.568, 0.800, 0.915, 0.924]) 
    y3 = np.array([0.440, 0.605, 0.721, 0.868, 0.916]) 
    y4 = np.array([0.440, 0.622, 0.794, 0.869, 0.920]) 
    y5 = np.array([0.440, 0.707, 0.861, 0.898, 0.917]) 
    y6 = np.array([0.440, 0.716, 0.880, 0.908, 0.916])
    y7 = np.array([0.440, 0.575, 0.870, 0.899, 0.910]) 
    y8 = np.array([0.440, 0.583, 0.781, 0.865, 0.903])
    

elif type == "armadillo":
    y0 = np.loadtxt("./exp/ral/armadillo/sc.txt")[x-1]
    y1 = np.array([0.494, 0.645, 0.882, 0.980, 0.993])
    y2 = np.array([0.494, 0.594, 0.857, 0.980, 0.994])
    y3 = np.array([0.494, 0.647, 0.833, 0.908, 0.986])
    y4 = np.array([0.494, 0.636, 0.811, 0.933, 0.988])
    y5 = np.array([0.494, 0.766, 0.950, 0.979, 0.990])
    y6 = np.array([0.494, 0.776, 0.944, 0.973, 0.981])
    y7 = np.array([0.494, 0.652, 0.903, 0.974, 0.989])
    y8 = np.array([0.494, 0.643, 0.876, 0.957, 0.986])

elif type == "dragon":
    y0 = np.loadtxt("./exp/ral/dragon/sc.txt")[x-1]
    y1 = np.array([0.429, 0.494, 0.717, 0.883, 0.936])
    y2 = np.array([0.429, 0.521, 0.741, 0.907, 0.934]) 
    y3 = np.array([0.429, 0.429, 0.615, 0.859, 0.927]) 
    y4 = np.array([0.429, 0.461, 0.599, 0.883, 0.930]) 
    y5 = np.array([0.429, 0.604, 0.789, 0.873, 0.932]) 
    y6 = np.array([0.429, 0.622, 0.798, 0.858, 0.910]) 
    y7 = np.array([0.429, 0.467, 0.795, 0.869, 0.911]) 
    y8 = np.array([0.429, 0.524, 0.737, 0.820, 0.899])


plt.figure()

markersize = 5
plt.plot(x, y0, label = "Ours", 
        linestyle = linestyle, linewidth = linewidth, marker = marker, markersize = markersize)
plt.plot(x, y1, label = "Occlusion Aware", 
        linestyle = linestyle, linewidth = linewidth, marker = marker, markersize = markersize)
plt.plot(x, y2, label = "Unobserved Voxel", 
        linestyle = linestyle, linewidth = linewidth, marker = marker, markersize = markersize)
plt.plot(x, y3, label = "Rear Side Voxel", 
        linestyle = linestyle, linewidth = linewidth, marker = marker, markersize = markersize)
plt.plot(x, y4, label = "Rear Side Entropy", 
        linestyle = linestyle, linewidth = linewidth, marker = marker, markersize = markersize)
plt.plot(x, y5, label = "Proximity Count", 
        linestyle = linestyle, linewidth = linewidth, marker = marker, markersize = markersize)
plt.plot(x, y6, label = "Area Factor", 
        linestyle = linestyle, linewidth = linewidth, marker = marker, markersize = markersize)
plt.plot(x, y7, label = "Average Entropy", 
        linestyle = linestyle, linewidth = linewidth, marker = marker, markersize = markersize)
plt.plot(x, y8, label = "Random View", 
        linestyle = linestyle, linewidth = linewidth, marker = marker, markersize = markersize)

plt.xlim(0, 11)
plt.xticks(np.linspace(0,11,12,endpoint=True))
plt.ylim(0.0,1.0)
plt.yticks(np.linspace(0,1,5,endpoint=True))

legend_font = {"family" : "Times New Roman", "size" : 11}
plt.legend(prop = legend_font)

plt.ylabel("Surface Coverage", fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.xlabel("Reconstruction Step", fontdict={'family' : 'Times New Roman', 'size'   : 16})
# plt.title('Surface Coverage Evaluation')

plt.show()
