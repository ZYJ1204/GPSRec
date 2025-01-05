import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def draw_curve(path,best_epoch):
    f1 = open(os.path.join(path, 'history.pkl'), 'rb')
    history = pickle.load(f1)

    x = range(1, len(history['ja']) + 1)
    y1 = history['ja']
    y2 = history['prauc']
    y3 = history['avg_p']
    y4 = history['avg_r']
    y5 = history['avg_f1']
    y6 = history['ddi_rate']

    plt.title('Evaluation indicators', fontsize=14)
    # Draw the graph
    plt.plot(x, y1, 'b--', alpha=0.5, linewidth=1, label='ja')
    plt.plot(x, y2, 'r--', alpha=0.5, linewidth=1, label='prauc')
    plt.plot(x, y3, 'g--', alpha=0.5, linewidth=1, label='avg_p')
    plt.plot(x, y4, 'y--', alpha=0.5, linewidth=1, label='avg_r')
    plt.plot(x, y5, 'k--', alpha=0.5, linewidth=1, label='avg_f1')

    plt.legend(loc="upper left")  # position of the legend
    plt.xlabel('Epoch')
    plt.xticks(x[::2])
    # The best_epoch point is marked with a small red dot, and the ordinate value is output to the figure with 4 decimal places
    plt.scatter(x[best_epoch], y1[best_epoch], color='r', s=10)
    plt.text(x[best_epoch], y1[best_epoch], '%.4f' % y1[best_epoch], ha='center', va='bottom', fontsize=10)
    plt.scatter(x[best_epoch], y2[best_epoch], color='r', s=10)
    plt.text(x[best_epoch], y2[best_epoch], '%.4f' % y2[best_epoch], ha='center', va='bottom', fontsize=10)
    plt.scatter(x[best_epoch], y3[best_epoch], color='r', s=10)
    plt.text(x[best_epoch], y3[best_epoch], '%.4f' % y3[best_epoch], ha='center', va='bottom', fontsize=10)
    plt.scatter(x[best_epoch], y4[best_epoch], color='r', s=10)
    plt.text(x[best_epoch], y4[best_epoch], '%.4f' % y4[best_epoch], ha='center', va='bottom', fontsize=10)
    plt.scatter(x[best_epoch], y5[best_epoch], color='r', s=10)
    plt.text(x[best_epoch], y5[best_epoch], '%.4f' % y5[best_epoch], ha='center', va='bottom', fontsize=10)
    plt.savefig(os.path.join(path, 'eval.png'))
    plt.show()

    # Draw ddi_rate separately, with the ordinate retaining 4 decimal places
    plt.title('ddi_rate', fontsize=14)
    plt.plot(x, y6, 'b--', alpha=0.5, linewidth=1, label='ddi_rate')  # '
    plt.legend(loc="upper left")  # position of the legend
    plt.xlabel('Epoch')
    plt.xticks(x[::2])
    plt.scatter(x[best_epoch], y6[best_epoch], color='r', s=10)
    plt.text(x[best_epoch], y6[best_epoch], '%.4f' % y6[best_epoch], ha='center', va='bottom', fontsize=10)
    plt.savefig(os.path.join(path, 'ddi_rate.png'))
    plt.show()
