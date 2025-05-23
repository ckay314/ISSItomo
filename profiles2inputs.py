import numpy as np
import os
import datetime 


dates = ['20170910', '20201129', '20211028']
# lon shifts are the sat lon in HEEQ at t=0 in HEEQ
lonShifts = {'20201129':-87.74, '20170910':82.54, '20211028':-0.66}

print ('ID                     21.5 Time         Lat (deg)   Lon(T)   Lon(M)    Tilt      AW       AWp    a (AU)   b (AU)   eclAW   v (km/s)  dens (1e-18 kg/m3)   mass (1e15 g)')

for aDate in dates:
    # Open the containing folder
    dir_path = aDate+'/fits/'
    
    # Find all the files
    allFiles = np.sort(os.listdir(dir_path))
    for aFile in allFiles:
        if aFile[0] == '.':
            idx = np.where(allFiles == aFile)[0]
            allFiles = np.delete(allFiles, idx)
    satSetups = [a.replace(aDate, '').replace('_CPfits.dat', '') for a in allFiles]
    
    # Open the file with MHD 2 real time
    t2t = np.genfromtxt(aDate+'/sim_ut_times_cor.txt', dtype=str)
    MHDtime = []
    realtime = []
    for i in range(len(t2t[:,0])):
        MHDtime.append(int(t2t[i,0]))
        realtime.append(datetime.datetime.strptime(t2t[i,2], "%Y-%m-%dT%H:%M:%S" ))
    MHDtime = np.array(MHDtime)
    realtime = np.array(realtime) 
    
    
    for i in range(len(allFiles)):
        myFile = allFiles[i]
        myName = satSetups[i]
        data = np.genfromtxt(dir_path+myFile, dtype=None)
        
        # Collect the data from the file
        # order is time, Rs, lats, lons, tilts, AWs, AWps, ax1s, ax2s 
        res = [[], [], [], [], [], [], [], [], [], [], []]
        for ii in range(len(data)):
            myRow = data[ii]
            if data[ii][1] != 9.999:
                for j in range(11):
                    if j == 0:
                        this = str(data[ii][j])
                        res[j].append(this.replace(aDate, '').replace(myName,'').replace('_',''))
                    else:    
                        res[j].append(float(data[ii][j]))
        
        # Clean up tilts based on range
        if np.mean(np.abs(res[3])) > 45:
            for ii in range(len(res[3])):
                if res[3][ii] < 0:
                    res[3][ii] += 180.

        for j in range(11):
            res[j] = np.array(res[j]).astype(float)
        
       
        # Print this profile to screen to check things
        if False:            
            for j in range(len(res[0])):
                print (res[0][j], res[1][j], res[2][j], res[3][j], res[4][j], res[5][j], res[6][j], res[7][j])
        

        # Get the time at 0.1 AU
        critDist = 0.1
        cIdx = np.max(np.where(res[1].astype(float) < critDist)) # critical index
        fa = (res[1][cIdx+1] - critDist) / (res[1][cIdx+1] - res[1][cIdx])
        fb = (critDist- res[1][cIdx]) / (res[1][cIdx+1] - res[1][cIdx])
        
        lowTime = realtime[cIdx]
        timeDiff = realtime[cIdx+1] - lowTime
        cTime = lowTime + timeDiff * fa

        # Take the median of the GCS parameters above 0.08 (where it seems to level out)
        levelH = 0.08
        levelH2 = 0.1
        hiIdx = np.where(res[1] >= levelH)
        peakIdx = np.where((res[1] >= levelH) & (res[1] <= levelH2))
        mLat = np.median(res[2][hiIdx])
        mLon = np.median(res[3][hiIdx])
        mTilt = np.median(res[4][hiIdx])
        mAW = np.median(res[5][hiIdx])
        mAWp = np.median(res[6][hiIdx])
        mass = np.median(res[10][peakIdx])
        # replace these wit vals at 0.1 au
        #ma = np.median(res[7][hiIdx])
        #mb = np.median(res[8][hiIdx])
        ca = fa*res[7][cIdx] + fb*res[7][cIdx+1]
        cb = fa*res[8][cIdx] + fb*res[8][cIdx+1]
        dens = fa*res[9][cIdx] + fb*res[9][cIdx+1]
        
        # Get the velocity
        timeDiffs = []
        for k in range(len(res[0])-1):
            timeIdx = np.where(MHDtime == res[0][k])[0]
            timeDiffs.append((realtime[timeIdx[0]+1] - realtime[timeIdx[0]]).seconds)
        meanTD = np.mean(np.array(timeDiffs))
        
        pfit = np.polyfit(res[0], res[1], 1) # linear fit to h vs mhd time
        v = pfit[0] * 1.496e8 / meanTD
            
        
        # Estimate the angular width in the Eq plane (from Mateja 2019)
        eqAW = mAW - (mAW - mAWp) * np.abs(mTilt) / 90
        
        print ((aDate+'_'+myName).ljust(20), cTime.strftime("%Y-%m-%dT%H:%M:%S"), '{:8.2f}'.format(mLat), '{:8.2f}'.format(mLon), '{:8.2f}'.format(mLon+lonShifts[aDate]),   '{:8.2f}'.format(mTilt), '{:8.2f}'.format(mAW), '{:8.2f}'.format(mAWp), '{:8.3f}'.format(ca), '{:8.3f}'.format(cb), '{:8.2f}'.format(eqAW), '{:8.1f}'.format(v), '{:8.2f}'.format(dens), '{:8.2f}'.format(mass))    
    #print (sd)
    