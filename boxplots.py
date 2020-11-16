import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

ap_voxel, auc_voxel = np.load("apauc.npz")['arr_0']
ap_sample, auc_sample = np.load("apauc_sample.npz")['arr_0']

write_pvals = False

fig, axs = plt.subplots(2,2)
kwargs = dict(ha='center', va='bottom', fontsize=10)

axs[0,0].boxplot(ap_voxel.T)
axs[0,0].set_title('AP voxel')
axs[0,0].plot([1,1,4,4], [0.78,.79,.79,0.78], c='black')
if write_pvals:
    axs[0,0].text(2.5,.78, "7e-4", **kwargs)
else:
    axs[0,0].text(2.5,.78, "*", **kwargs)
axs[0,0].plot([1,1,2,2], [0.68+.05,.69+.05,.69+.05,0.68+.05], c='black')
if write_pvals:
    axs[0,0].text(1.5,.73, "4e-4", **kwargs)
else:
    axs[0,0].text(1.5,.73, "*", **kwargs)
axs[0,0].set_ylim(0,0.85)

axs[0,1].boxplot(auc_voxel.T)
axs[0,1].set_title('AUC voxel')
axs[0,1].plot([1,1,2,2], [0.98+.03,.99+.03,.99+.03,0.98+.03], c='black')
if write_pvals:
    axs[0,1].text(1.5,1.01, "2e-8", **kwargs)
else:
    axs[0,1].text(1.5,1.01, "***", **kwargs)
axs[0,1].plot([2,2,3,3], [0.99+.05,1+.05,1+.05,0.99+.05], c='black')
if write_pvals:
    axs[0,1].text(2.5,1.04, "3e-6", **kwargs)
else:
    axs[0,1].text(2.5,1.04, "***", **kwargs)
axs[0,1].plot([2,2,4,4], [0.98+.1,.99+.1,.99+.1,0.98+.1], c='black')
if write_pvals:
    axs[0,1].text(3,1.08, "4e-5", **kwargs)
else:
    axs[0,1].text(3,1.08, "**", **kwargs)
axs[0,1].set_ylim(0.4,1.15)

axs[1,0].boxplot(ap_sample.T[:88])
axs[1,0].set_title('AP sample')
axs[1,0].plot([1,1,4,4], [0.93,.94,.94,0.93], c='black')
if write_pvals:
    axs[1,0].text(2.5,.94, "9e-4", **kwargs)
else:
    axs[1,0].text(2.5,.94, "*", **kwargs)
axs[1,0].set_ylim(0.55,1)

axs[1,1].boxplot(auc_sample.T[:88])
axs[1,1].set_title('AUC sample')
axs[1,1].plot([1,1,4,4], [0.93,.94,.94,0.93], c='black')
if write_pvals:
    axs[1,1].text(2.5,.94, "9e-4", **kwargs)
else:
    axs[1,1].text(2.5,.94, "*", **kwargs)
axs[1,1].set_ylim(0.6,1)

for i in range(2):
    for j in range(2):
        axs[i,j].set_xticklabels(["BM", "CM", "PM", "AE"])


fig.subplots_adjust(wspace=0.3)
plt.show()

significance = np.zeros((4,4,4), dtype=np.bool)
pvals = np.ones((4,4,4))
bonferroni = 2*2*(3*2*1)
for i, metric in enumerate([ap_voxel, auc_voxel, ap_sample, ap_sample]):
    for j in range(4):
        for k in range(4):
            if j<=k:
                continue
            if i >= 2:
                cileft = np.percentile(metric[j,:]-metric[k,:], 2.5/bonferroni)
                ciright = np.percentile(metric[j,:]-metric[k,:], 100-2.5/bonferroni)
                if np.sign(cileft) == np.sign(ciright):
                    significance[i,j,k] = True
                    if np.sign(cileft) > 0:
                        pvals[i,j,k] = np.mean(metric[j,:]-metric[k,:] < 0)
                    else:
                        pvals[i,j,k] = np.mean(metric[j,:]-metric[k,:] > 0)
            else:
                pval = wilcoxon(metric[j,:],metric[k,:]).pvalue
                significance[i,j,k] = pval < .05/bonferroni
                pvals[i,j,k] = pval