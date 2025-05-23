import numpy as np
import matplotlib.pyplot as plt
import datetime

#thisdate = '20170910'
#thisdate = '20201129'
thisdate = '20211028'
MHDfolder = 'events/'+thisdate+'/insitu_profiles/'
MHDnames = ['', '_20degE_of', '_10degE_of', '_10degW_of', '_20degW_of', '_10degN_of', '_10degS_of']
colors = ['k', '#151B8D', '#1E90FF', '#6CC417', '#004225', '#810541', '#FF4500']

# Plot shows B, Br, Bt, Bn, v, Np, Tp, Beta
fig, ax = plt.subplots(8, 7, sharex=True, figsize=(12,8))

# MHD file has Time [0], Btot [1], Br [2], Bt [3], Bn [4], vtot [5], vr [6], vt [7], vn [8], np [9], Tp [10], beta [11]
# Time as 2021-10-28T15:30:00

allBounds =[np.empty(3) for i in range(len(MHDnames))]
for j in range(len(MHDnames)):
    f = MHDnames[j]
    # Read in txt file
    dataALL = np.genfromtxt(MHDfolder+thisdate+'_insitu'+f+'_L1.txt', dtype=str)
    
    # Format time into datetime obj
    timeSTR = dataALL[:,0]
    npts = len(timeSTR)
    times = []
    for i in range(npts):
        times.append( datetime.datetime.strptime(timeSTR[i], "%Y-%m-%dT%H:%M:%S"))
    times = np.array(times)
    data = np.zeros([npts,9])
    i2idx = [1,2,3,4,5,9,10,11]
    
    # calculate time derivative
    dt = []
    midts = []
    for i in range(npts-1):
        dt.append((times[i+1]-times[i]).total_seconds())
        midts.append(times[i]+datetime.timedelta(seconds=dt[i]))
    dt = np.array(dt)
    midts = np.array(midts)
        
    # Load the needed data into a 2d array
    for i in range(8):
        data[:,i] = dataALL[:,i2idx[i]]
    data[:,6] = data[:,6] / 1e3 # scale down temperature
    data[:,8] = np.arctan2(data[:,2], data[:,1])*180/3.14 # clock Ang of B
    
    
    # calculate derivatives
    ders = np.zeros([npts-1,9])
    for i in range(9):
        ders[:,i] = (data[1:,i]-data[:-1,i])/dt
    
    # Plot the data    
    for i in range(8):
        ax[i,j].plot(times, data[:,i], color=colors[j])
    #ax[-1,j].plot(midts, ders[:,0], color=colors[j])
    xlim = ax[0,0].get_xlim()
    
    # Find the front of sheath
    # shock front from max in deriv of T
    shockt = midts[np.where(ders[:,6]==np.max(ders[:,6]))]
    
    # Front of CME
    # option A for CME front - min in deriv of T
    CMEta = midts[np.where(ders[:,6]==np.min(ders[:,6]))]
    # or just set to min in beta
    #CMEta = times[np.where(data[:,7]==np.min(data[:,7]))]
    
    # Back of CME
    # set at first zero in dB/dt following min in dB/dt
    '''halfBtime = times[np.max(np.where(data[:,0] > 0.5*np.max(data[:,0]))[0])]
    minidx = np.where(ders[:,0]==np.min(ders[:,0]))[0]
    posidx = np.where((ders[:,0]>-0.00005) & (midts > halfBtime))[0]
    afteridx = posidx[np.min(np.where(posidx > minidx)[0])]
    CMEtb = midts[afteridx]'''
    # set CME end at where beta first reached 0.5 after the min in beta
    # can't do beta=1 bc some problem cases that have long tail
    tminbeta = times[np.where(data[:,7]==np.min(data[:,7]))]
    afteridx = np.where((data[:,7] > 0.5) * (times > tminbeta))[0]
    CMEtb = times[np.min(afteridx)]
    allBounds[j] = [shockt[0], CMEta[0], CMEtb]
    
    # get upstream properties
    names = ['B ', 'Br ', 'Bt ', 'Bn ', 'v ', 'n ', 'T ', 'beta ']
    units = [' nT', ' nT', ' nT', ' nT', ' km/s', ' cm-3', ' K', '' ]
    if j == 0:
       #b4idx = np.where((times < (allBounds[0][0]- datetime.timedelta(hours=6))) & (times >= (allBounds[0][0] - datetime.timedelta(hours=18))))[0]
       b4idx = np.where((times >= times[0]) & (times <= (times[0] + datetime.timedelta(hours=12))))[0]
       for i in range(8):
           if i == 6: 
               print (names[i]+'%5.2f'%(1e3*np.mean(data[b4idx,i])) + units[i])
           else:
               print (names[i]+'%5.2f'%(np.mean(data[b4idx,i])) + units[i])
        

for ii in range(7):
    ax[7,ii].set_yscale('log')
    n = 2
    [l.set_visible(False) for (i,l) in enumerate(ax[-1,ii].xaxis.get_ticklabels()) if i % n != 0]

for i in range(8):
    for j in range(len(MHDnames)):
        ylim = ax[i,j].get_ylim()
        ax[i,j].plot([allBounds[j][0], allBounds[j][0]], ylim, '--', color=colors[j])
        ax[i,j].plot([allBounds[j][1], allBounds[j][1]], ylim, ':', color=colors[j])
        ax[i,j].plot([allBounds[j][2], allBounds[j][2]], ylim, ':', color=colors[j])
        ax[i,j].set_ylim(ylim)

#ax[6,0].plot(ax[6,0].get_xlim(), [0,0],'k--')
    
        
ytits = ['B [nT]', 'B$_\\mathrm{R}$ [nT]', 'B$_\\mathrm{T}$ [nT]', 'B$_\\mathrm{N}$ [nT]', 'V [km s$\\mathrm{^{-1}}$]', 'N$_\\mathrm{p}$ [cm$^{\\mathrm{-3}}$]', 'T$_\\mathrm{p}$ [10$^\\mathrm{3}$ K]', '$\\beta$' ]
for i in range(8):
    ax[i,0].set_ylabel(ytits[i], fontdict=dict(weight='bold'))    
for i in range(7):
    ax[0,i].set_title(MHDnames[i][1:7])
fig.autofmt_xdate()
plt.subplots_adjust(hspace=0.1,left=0.06,right=0.98,top=0.95,bottom=0.1,wspace=0.25)


for i in range(7):
    outstr = MHDnames[i][1:7] + ' '
    outstr2 = ' '
    if i == 0:
        outstr = '0deg   '
    for item in allBounds[i]:
        outstr += item.strftime("%Y-%m-%dT%H:%M") + ' '
        dt = (item - datetime.datetime(item.year, 1,1)).total_seconds()/3600./24. + 1
        outstr2 += '{:.3f}'.format(dt) + ' '
    print (outstr+outstr2)
        
#plt.show()    
plt.savefig('MHDbounds_'+thisdate+'.png')