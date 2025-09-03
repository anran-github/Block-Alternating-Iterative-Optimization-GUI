import matplotlib.pyplot as plt
import numpy as np

cost_ours = [6.065]*7
cost_PSO = [6.065]*7
cost_PSO[0] = 6.225
cost_GA = [6.156,6.104,6.098,6.078,6.071,6.066,6.065]
costlist = [cost_ours,cost_PSO,cost_GA]
name_order = ['Block-Alternating','PSO','GA']

time_ours = [0.32,0.36,0.89,1.54,3,5.95,10.36]
time_PSO = [0.41,0.81,1.2,2.05,4.11,8.78,17.06]
time_GA = [0.48,0.51,1.3,2.13,3.96,8.27,17.72]
timelist = [time_ours,time_PSO,time_GA]

x_label = [8,16,32,64,128,256,512]

plt.subplot(211)
plt.plot(x_label,costlist[1],label=name_order[1],color='blue',marker='^', linewidth=2)
plt.plot(x_label,costlist[2],label=name_order[2],color='green',marker='o', linewidth=2)
plt.plot(x_label,costlist[0],label=name_order[0],color='orange',marker='*', linewidth=2)
plt.legend()
plt.ylabel('Cost Function Value')
plt.grid(linestyle='--', alpha=0.7)
plt.xscale("log", base=2)  #  make x-axis logarithmic
plt.xticks(x_label, x_label)  # keep original labels
plt.tight_layout()

plt.subplot(212)
plt.plot(x_label,timelist[1],label=name_order[1],color='blue',marker='^', linewidth=2)
plt.plot(x_label,timelist[2],label=name_order[2],color='green',marker='o', linewidth=2)
plt.plot(x_label,timelist[0],label=name_order[0],color='orange',marker='*', linewidth=2)
plt.legend()
plt.xlabel('Number of Initial Points')
plt.ylabel('Computing time [s]')
plt.grid(linestyle='--', alpha=0.7)
plt.xscale("log", base=2)  #  same for time plot
plt.xticks(x_label, x_label)
plt.tight_layout()
plt.savefig('initial_pts_affect_plot.png',dpi=300)

plt.show()
