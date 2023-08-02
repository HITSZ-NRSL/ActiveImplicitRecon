import numpy as np
import matplotlib.pyplot as plt


linestyle = "-"
linewidth = 2
marker = "o"
markersize = 3

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_entropy = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
type = "bunny"

# # draw entropy

# if type == "bunny":
#     y0 = np.loadtxt("./exp/ablation/cv/entropy.txt")[x - 1]
#     y1 = np.loadtxt("./exp/ablation/e2eag/entropy.txt")[x - 1]
#     y2 = np.loadtxt("./exp/ablation/sample/entropy.txt")[x - 1]
#     y3 = np.loadtxt("./exp/ablation/sampleag_e2eag/entropy.txt")[x - 1]

# plt.figure()
# plt.plot(
#     x_entropy,
#     y0,
#     label="Candidate Views",
#     linewidth=linewidth,
#     marker=marker,
#     markersize=markersize,
# )
# plt.plot(
#     x_entropy,
#     y1,
#     label="Opti.",
#     linestyle=linestyle,
#     linewidth=linewidth,
#     marker=marker,
#     markersize=markersize,
# )
# plt.plot(
#     x_entropy,
#     y2,
#     label="Sampling",
#     linestyle=linestyle,
#     linewidth=linewidth,
#     marker=marker,
#     markersize=markersize,
# )
# plt.plot(
#     x_entropy,
#     y3,
#     label="Sampling + Opti.",
#     linestyle=linestyle,
#     linewidth=linewidth,
#     marker=marker,
#     markersize=markersize,
# )

# plt.xlim(0, 11)
# plt.xticks(np.linspace(0, 11, 12, endpoint=True))
# plt.ylim(0.0, 0.3)
# plt.yticks(np.linspace(0, 0.3, 5, endpoint=True))

# legend_font = {"family": "Times New Roman", "size": 11}
# plt.legend(prop=legend_font, bbox_to_anchor=(1.0, 0.35), loc="center right")

# plt.ylabel("Entropy in map", fontdict={"family": "Times New Roman", "size": 16})
# plt.xlabel("Reconstruction Step", fontdict={"family": "Times New Roman", "size": 16})
# # plt.title('Etropy Evaluation')


# draw surface coverage

if type == "bunny":
    y0 = np.loadtxt("./exp/ablation/cv/sc.txt")[x - 1]
    y1 = np.loadtxt("./exp/ablation/e2eag/sc.txt")[x - 1]
    y2 = np.loadtxt("./exp/ablation/sample/sc.txt")[x - 1]
    y3 = np.loadtxt("./exp/ablation/sampleag_e2eag/sc.txt")[x - 1]


markersize = 5
plt.plot(
    x,
    y0,
    label="Candidate Views",
    linestyle=linestyle,
    linewidth=linewidth,
    marker=marker,
    markersize=markersize,
)
plt.plot(
    x,
    y1,
    label="Optimization",
    linestyle=linestyle,
    linewidth=linewidth,
    marker=marker,
    markersize=markersize,
)
plt.plot(
    x,
    y2,
    label="Sampling",
    linestyle=linestyle,
    linewidth=linewidth,
    marker=marker,
    markersize=markersize,
)
plt.plot(
    x,
    y3,
    label="Ours",
    linestyle=linestyle,
    linewidth=linewidth,
    marker=marker,
    markersize=markersize,
)

plt.xlim(0, 11)
plt.xticks(np.linspace(0, 11, 12, endpoint=True))
plt.ylim(0.4, 1.0)
plt.yticks(np.linspace(0.4, 1, 13, endpoint=True))

legend_font = {"family": "Times New Roman", "size": 11}
plt.legend(prop=legend_font)

plt.ylabel("Surface Coverage", fontdict={"family": "Times New Roman", "size": 16})
plt.xlabel("Reconstruction Step", fontdict={"family": "Times New Roman", "size": 16})
# plt.title('Surface Coverage Evaluation')

plt.show()
