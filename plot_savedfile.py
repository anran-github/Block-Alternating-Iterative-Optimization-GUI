import matplotlib.pylab as plt
import json
import numpy as np


def plot_savedfile(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    cost_data = data['results']
    cost = np.array(cost_data)

    # plot
    plt.figure(figsize=(5, 3))
    plt.plot(cost, label='Cost', color='orange',marker='*', linewidth=2)
    # plt.title('Cost Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function Value')
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('cost_plot.png')
    plt.show()


plot_savedfile('Multi_SCD_results_paper.json')