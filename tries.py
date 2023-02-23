import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from scipy.optimize import curve_fit

df = pd.read_excel('./df.xlsx')
tries = df.iloc[:, 5:12]

def normal_distribution(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

# for i in range(len(tries)):
k=0
fig, ax = plt.subplots(2, 3)
plt.rcParams.update({"font.size": 12})
plt.figure(figsize=(15, 13))
plt.grid(linestyle='--')
ax[0][0].set_ylabel('tries percentage')
ax[0][1].set_yticks([])
ax[0][2].set_yticks([])
ax[1][0].set_ylabel('tries percentage')
ax[1][1].set_yticks([])
ax[1][2].set_yticks([])
fig.tight_layout()

RMSEs = []
for i in range(6):
    t = np.array(tries.iloc[i,:], dtype='float32')
    t_norm = t/100
    x = np.linspace(0,len(t_norm),100)
    popt, pcov = curve_fit(normal_distribution, np.arange(len(t_norm)), t_norm)
    fit = normal_distribution(x, *popt)
    t = t_norm * 100
    RMSEs.append(np.sqrt(metrics.mean_squared_error(t_norm*100,100*normal_distribution(range(len(t_norm)), *popt))))
    ax[int(k/3)][k%3].plot(np.arange(len(t)), t_norm, '-', color='lightskyblue', label='data')
    ax[int(k/3)][k%3].plot(x, fit, '--', color='orange', label='fit')
    ax[int(k/3)][k%3].axvline(x=popt[0], ls=":", c="darkgrey")
    ax[int(k/3)][k%3].text(x=popt[0]+0.2, y=0, s='mean:%.2f\nstd:%.2f'%(popt[0],popt[1]))
    ax[int(k/3)][k%3].text(x=0, y=0, s='{}'.format(str(df['Word'][i])))
    k+=1
RMSEs = np.array(RMSEs)
print(RMSEs)
print(RMSEs.mean())
lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc = 'upper right')
plt.show()
