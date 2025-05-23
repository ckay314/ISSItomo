import numpy as np
import os
import matplotlib.pyplot as plt

#date ='20170910'
#date ='20201129'
date ='20211028'

# Date should be a string of just the date (no /)
dir_path = date+'/fits/'

allFiles = np.sort(os.listdir(dir_path))

fig, axes = plt.subplots(4, 2, figsize=(8,10))
axes = [axes[0,0], axes[0,1], axes[1,0], axes[2,0], axes[2,1], axes[3,0], axes[3,1], axes[1,1]]
for aFile in allFiles:
    if aFile[0] == '.':
        idx = np.where(allFiles == aFile)[0]
        allFiles = np.delete(allFiles, idx)
        
for aFile in allFiles:
    data = np.genfromtxt(dir_path+aFile, dtype=None)
    myName = aFile.replace(date, '')
    myName = myName.replace('_CPfits.dat', '')
    # order is Rs, lats, lons, tilts, AWs, AWps, ax1s, ax2s, rho, M 
    res = [], [], [], [], [], [], [], [], [], []    
    for i in range(len(data)):
        myRow = data[i]
        if data[i][1] != 9.999:
            for j in range(10):
                res[j].append(data[i][1+j])
    # Clean up tilts based on range
    if np.mean(np.abs(res[3])) > 45:
        for i in range(len(res[3])):
            if res[3][i] < 0:
                res[3][i] += 180.
                
    for j in range(7):
        if j == 0:
            axes[j].plot(res[0], res[1+j], label=myName)
        else:
            axes[j].plot(res[0], res[1+j])
    axes[7].plot(res[0], res[9])

ylabs = ['Lat (deg)', 'Lon (deg)','Tilt (deg)','AW$_{FO}$ (deg)', 'AW$_{EO}$ (deg)', 'Semimajor l (au)', 'Semiminor l (au)', 'Mass (1e15 g)']
for i in range(8):
    axes[i].set_xlabel('R (au)')            
    axes[i].set_ylabel(ylabs[i])            
            
fig.legend(loc='upper center', fancybox=True, fontsize=13, labelspacing=0.4, handletextpad=0.4, framealpha=0.5, ncol=len(allFiles)/2)
plt.subplots_adjust(wspace=0.4, hspace=0.4,left=0.1,right=0.95,top=0.9,bottom=0.05)   
    
plt.savefig('fitEvol'+date+'.png')