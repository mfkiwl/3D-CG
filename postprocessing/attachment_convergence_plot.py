from typing import Set
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

iter = np.arange(12)
error_thresh = np.ones_like(iter)*0.05
setpoint = np.zeros_like(iter)

error_vec = np.array([-1, -1, -1, -0.98875350268169, -0.913520, -0.6522351, -0.065760097, 0.4177390, 0.18495867, -0.0657600, 0.0696838, 0.014674157158])

fig, ax = plt.subplots(figsize=(20,15))
ax.plot(iter, error_thresh, c='blue', label = 'Stopping threshold',  linewidth=5.0)
ax.plot(iter, setpoint, c='black', label = 'Setpoint',  linewidth=5.0)
ax.plot(iter, error_vec, c='red', label='Positive corona charge error',  linewidth=5.0)

ax.legend(fontsize=25)
ax.set_xlabel('Iteration', fontsize=25)
ax.set_ylabel('Error', fontsize=25)
ax.set_ylim([-1, 0.5])
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
ax.set_xticks(iter)
ax.set_xlim((0,11))
plt.title('Iteration error for a representative first leader attachment cycle', fontsize=25)
ax.add_patch(mpl.patches.Rectangle((0, 0), 12, 0.05, hatch='xx', fill=True, facecolor='blue', alpha=0.4, snap=False, linewidth=0))
plt.tight_layout()

# plt.show()
plt.savefig('attachment_error_convergence_plot.png')