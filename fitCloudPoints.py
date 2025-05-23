# CK's code to take in a cloud of points from the tomography
# and clean and fit an ellipse to it

# Right now it just prints out the best fit ellipse at the end
# and separate code is in progress to use ellipse params to get
# actual GCS equivalent numbers

# Uploaded 25 Sep 2024


import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import opening, disk, dilation
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline
import math
import os
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning) 

dtor = np.pi/180.
silent = True
saveFig = False
printPretty = False

#data = np.genfromtxt('20201129/3d_densities/007pb/x2_solved_035_0400_0120_0060.dat')
#data = np.genfromtxt('20170910/3d_densities/003/x2_solved_025_0400_0120_0060.dat')
#data = np.genfromtxt('20211028/3d_densities/003_ring/x2_solved_056_0400_0120_0060.dat')

def fitCloudFun(file_in, figName='temp'):
    data = np.genfromtxt(file_in)
    dens = data[:,0]
    r = data[:,1]
    lon = data[:,2]
    colat = data[:,3] # was colat
    lat   = np.pi/2 - colat
    shiftLon = np.copy(lon)
    shiftLon[np.where(shiftLon > np.pi)] -= 2*np.pi
    densr2 = dens * r**2
    logDR2 = np.log10(densr2)

    x = r * np.sin(colat) * np.cos(lon)
    y = r * np.sin(colat) * np.sin(lon)
    z = r * np.cos(colat)

    cloudxs = []
    cloudys = []
    cloudzs = []
    cloudLats = []
    cloudLons = []
    cloudRs = []
    cloudDR2s = []

    for plotR in np.arange(0.02, 0.15, 0.01):   
        goodIdx = np.where((np.abs(r-plotR) < 0.005) & (logDR2 > 5.5 ))[0]
        if len(goodIdx) > 0:
            newCut = np.percentile(densr2[goodIdx], 90)
            gooderIdx = goodIdx[np.where(densr2[goodIdx] > newCut)]
            weightLon = np.sum(shiftLon[goodIdx]*densr2[goodIdx]) / np.sum(densr2[goodIdx])
            weightLat = np.sum(lat[goodIdx]*densr2[goodIdx]) / np.sum(densr2[goodIdx])
            lonSpread = np.mean(np.abs(shiftLon[gooderIdx] - weightLon))
            if not silent:
                print ('Res at r =', '{:.2f}'.format(plotR), '{:.2f}'.format(lonSpread), lonSpread <= np.pi/4) 
            if lonSpread <= np.pi/4:
                for idx in gooderIdx:
                    cloudxs.append(x[idx])
                    cloudys.append(y[idx])
                    cloudzs.append(z[idx])
                    cloudLats.append(lat[idx])
                    cloudLons.append(shiftLon[idx])
                    cloudRs.append(r[idx])
                    cloudDR2s.append(densr2[idx])
        else:
            if not silent:
                print('No points at r = ', '{:.2f}'.format(plotR))

    cloudxs = np.array(cloudxs)
    cloudys = np.array(cloudys)
    cloudzs = np.array(cloudzs)
    cloudRs = np.array(cloudRs)
    cloudLats = np.array(cloudLats)
    cloudLons = np.array(cloudLons)
    cloudDR2s = np.array(cloudDR2s)
    finalCut = np.percentile(cloudDR2s, 10)
    finalIdx = np.where(cloudDR2s >= finalCut)[0]

    # Get the final set of cleaned points in nice names
    x = cloudxs[finalIdx]
    y = cloudys[finalIdx]
    z = cloudzs[finalIdx]
    r = cloudRs[finalIdx]
    lat = cloudLats[finalIdx]
    lon = cloudLons[finalIdx]
    densr2 = cloudDR2s[finalIdx]
    logDR2 = np.log10(densr2)
    

    if saveFig: 
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x,y,z, c=logDR2)
        ax.set_aspect('equal')
        axlims = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
        plt.savefig(figName+'A.png')

    # This hardcode is probably ok until get very far CMEs?
    # Also might not be necessary now with better cloud extraction
    #goodidx = np.where((r >0.01) & (r < 1.4))[0]

    # Get mean lat and lon from the full cloud
    meanlon = np.sum(lon*r)/np.sum(r)
    # the lons can extend over wider range at high/low lat due to projection
    # try and weight by abs distance in lon from mean lon
    lonDist = np.abs(lon-meanlon)
    # set some dist bounds to prevent overweighting
    lonDist[np.where(lonDist < 1)] = 1
    lonDist[np.where(lonDist > 30)] = 30.
    lonw8 = 1. / lonDist
    #meanlat = np.sum(lat*r*lonw8)/np.sum(r*lonw8)
    meanlat = (np.min(cloudLats)+np.max(cloudLats))/2
    if False:
        plt.scatter(lon, lat, c=r)
        plt.plot(meanlon, meanlat, c='r', zorder=10, ms=30)
        plt.show()
    if not silent:
        print ('Initial estimate of mean Lat/Lon:', meanlat/dtor, meanlon/dtor)
    
    # Get initial estimate of the tilt - rotate nose to [0,0] (lat,lon) then fit slope of yz points
    # rotate by lon about z
    x2 = np.cos(-meanlon)*x - np.sin(-meanlon) * y
    y2 = np.sin(-meanlon)*x + np.cos(-meanlon) * y
    z2 = z
    # rotate by lat about y
    z3 = -np.sin((meanlat)) * x2  + np.cos((meanlat)) *z2
    x3 =  np.cos((meanlat)) * x2  + np.sin((meanlat)) *z2


    ywid = np.percentile(y2, 90) - np.percentile(y2, 10)
    zwid = np.percentile(z3, 90) - np.percentile(z3, 10)

    # mostly vertical
    if ywid < zwid:
        uplim = np.percentile(z3, 90)
        downlim = np.percentile(z3, 10)
        downy, upy = np.median(y2[np.where(z3 <=downlim)]), np.median(y2[np.where(z3 >=uplim)])
        tilt = np.arctan2(uplim-downlim, upy-downy)
    # mostly horizontal
    else:
        rlim = np.percentile(y2, 90)
        llim = np.percentile(y2, 10)
        lz, rz = np.median(z3[np.where(y2 <=llim)]), np.median(z3[np.where(y2 >=rlim)])
        tilt = np.arctan2(rz-lz, rlim-llim)
    
    if not silent:    
        print ('Initial estimate of Tilt:', tilt/dtor)

    # Rotate points by est tilt so axis along y
    y4 = y2 * np.cos(-tilt) - np.sin(-tilt) * z3
    z4 = y2 * np.sin(-tilt) + np.cos(-tilt) * z3




    # calc radial distance from nose
    rnose = np.sqrt(y4**2 + z4**2)

    # Convert our set of points to an image so we can use fancy ML techniques
    ngrid = 400
    im = np.zeros([ngrid,ngrid])
    hlims = [np.min(y4),np.max(y4)]
    vlims = [np.min(z4),np.max(z4)]
    dh, dv = (hlims[1]-hlims[0])/(ngrid-1),  (vlims[1]-vlims[0])/(ngrid-1)
    for i in range(len(y4)):
        thish, thisv = y4[i], z4[i]
        hidx = int((thish-hlims[0])/dh)
        vidx = int((thisv-vlims[0])/dv)
        im[vidx, hidx] = rnose[i]
        # need to unhardcode appropriate distance from nose, not looking for perfect cut
        # but trying to elimate outermost unattached garbage
        if rnose[i] < 0.2:
            im[vidx, hidx] = 1
            
    # Need to unhardcode or at least better understand what appropriate dilation
    # and opening values are    
    footprint = disk(6)
    dil = dilation(im, footprint)
    footprint2 = disk(8)
    oped = opening(dil, footprint2)
    
    # Convert the cleaned image back into a structure
    newvs = []
    newhs = []
    for i in range(ngrid):
        for j in range(ngrid):
            if oped[i,j]:
                newvs.append(vlims[0]+i*dv)
                newhs.append(hlims[0]+j*dh)
    newvs = np.array(newvs)
    newhs = np.array(newhs)
        
    # Convert the cleaned points into polar
    angs = np.arctan2(newvs,newhs)
    rads = np.sqrt(newvs**2 + newhs**2)

    # Split the points into angular bins and attempt to find the FR edge within each one
    # Is 100 angle bins best option?
    npts = 100
    summed_angs = np.linspace(-np.pi, np.pi, npts)
    act_angs = []
    dAng = summed_angs[1] - summed_angs[0]
    summed_rads = []
    
    for i in range(npts -1):
        these_idx = np.where((angs >= summed_angs[i]) & (angs < summed_angs[i+1]))[0]
        if len(these_idx) > 0:
            sortRad = np.sort(rads[these_idx])
            maxRad = sortRad[-1]
            diffs = sortRad[1:]-sortRad[:-1]
            medD = np.median(diffs)
            stdD = np.std(diffs)
            bigJumps = np.where(diffs > (medD + 10*stdD))[0]
            if len(bigJumps) > 0:
                subJumps = np.where(sortRad[bigJumps+1] > np.median(sortRad))[0]
                if len(subJumps) > 0:
                    sortRad = sortRad[:bigJumps[subJumps[0]]]
                    maxRad = sortRad[-1]
            act_angs.append(summed_angs[i])
            summed_rads.append(maxRad)
    midangs = np.array(act_angs)+0.5*dAng
    summed_rads = np.array(summed_rads)
    
    # Filter spikes out of mid angs/summed rads
    medRad = np.median(summed_rads)
    # compare to neighbors and look for jumps
    biggerRads = np.empty(len(summed_rads)+2)
    biggerRads[1:-1] = summed_rads
    biggerRads[0] = summed_rads[-1]
    biggerRads[-1] = summed_rads[0]
    diff = np.abs(summed_rads - biggerRads[:-2]) + np.abs(summed_rads - biggerRads[2:])
    medDiff = np.median(diff)
    stdDiff = np.std(diff)
    
    # Plot comparison of edge with all points
    if False:
        fig = plt.figure()
        plt.plot(rads*np.cos(angs), rads*np.sin(angs), 'o')
        plt.plot(summed_rads*np.cos(midangs), summed_rads*np.sin(midangs), '+')
        plt.show()

    def calcErr(yEs, zEs, phi, y0, z0, a, b, plotit=False):
        y1 = yEs - y0
        z1 = zEs - z0

        y2 = y1 * np.cos(-phi) - z1 * np.sin(-phi)
        z2 = y1 * np.sin(-phi) + z1 * np.cos(-phi)
    
        myts = np.arccos(y2/a)
        myts[np.where(z2 < 0)] *= -1
        goodidx = np.where(~np.isnan(myts))[0]

        fakezs = b * np.sin(myts)
        allerr = np.abs(z2[goodidx] - fakezs[goodidx])
        maxGoodErr = np.max(allerr)
        nBad = len(yEs) - len(allerr)
        err = np.sum(allerr) + 2 * nBad * maxGoodErr
        return err 

    # Get best guess at properties based on edges
    yEs = summed_rads*np.cos(midangs)
    zEs = summed_rads*np.sin(midangs)
    g_y0 = 0.5*(np.percentile(yEs, 90) + np.percentile(yEs, 10))
    g_z0 =  0.5*(np.percentile(zEs, 90) + np.percentile(zEs, 10))
    g_a  = 0.5*(np.percentile(yEs, 90) - np.percentile(yEs, 10))
    g_b  = 0.5*(np.percentile(zEs, 90) - np.percentile(zEs,10))

    bestErr = 999999999
    for i in np.linspace(-90*dtor, 90*dtor, 181):
        myErr = calcErr(yEs, zEs, i, g_y0, g_z0, g_a, g_b)
        if myErr < bestErr:
            params = [i, g_y0, g_z0, g_a, g_b]
            bestErr = myErr
            #print (i/dtor, myErr, bestErr)
    g_p = float(params[0])

    if not silent:
        print ('Center guess at:', g_y0, g_z0)
        print ('Shape guess of: ', g_a, g_b)
        print ('Phi guess of:   ', g_p/dtor)

 
    nIter = 10 # 10 for testing, 20 seems better for more accurate "final" values?
    counter = 0
    bestErr = 999999999
    for phi in np.linspace(g_p-25*dtor, g_p+25*dtor, nIter):
        counter += 1
        #print ('Step ', counter, '/', nIter)
        for y0 in np.linspace(g_y0-g_a/3, g_y0+g_a/3, nIter):
            for z0 in np.linspace(g_z0-g_b/3, g_z0+g_b/3, nIter):
                for a in np.linspace(0.75*g_a, 1.25*g_a, nIter):
                    for b in np.linspace(0.75*g_b, 1.25*g_b, nIter):
                        myErr = calcErr(yEs, zEs, phi, y0, z0, a, b)
                        if myErr < bestErr:
                            bestErr = myErr
                            bestVals = [phi, y0, z0, a, b]
    phi, y0, z0, a, b = bestVals[0], bestVals[1], bestVals[2], bestVals[3], bestVals[4]  

    # Check if b > a, might be after actually fitting things
    # swap if so a flag to change tilt by 90 at end
    flipTilt = False
    if b > a:
        temp = b
        b = a
        a = temp
        flipTilt = True
    if not silent:
        print("Best fit to projected edge:")
        print(phi/dtor, y0, z0, a, b)

    # Check the fit to the projected edge vals
    if False:     
                # Ellipse to proj transofrm
                fakets = np.linspace(-np.pi, np.pi, 100)
                fakeys = a * np.cos(fakets)
                fakezs = b * np.sin(fakets)

                fakeys2 = fakeys * np.cos(phi) - fakezs * np.sin(phi) + y0
                fakezs2 = fakeys * np.sin(phi) + fakezs * np.cos(phi) + z0

                fig = plt.figure()
                plt.plot(fakeys2, fakezs2, 'bo')
                plt.plot(yEs, zEs, 'ro')
            
                ax = plt.gca()
                ax.set_aspect('equal', adjustable='box')
                plt.show()

    # Correct y4, z4 based on fit
    y5 = y4 - y0
    z5 = z4 - z0
    y6 = np.cos(-phi) * y5 - np.sin(-phi) * z5
    z6 = np.sin(-phi) * y5 + np.cos(-phi) * z5 

    thetas = np.arctan2(z6, y6)
    rs = np.sqrt(y6**2 + z6**2)

    fakets = np.linspace(-np.pi, np.pi, 100)
    fakeys = a * np.cos(fakets)
    fakezs = b * np.sin(fakets)
    thetaEs = np.arctan2(fakezs, fakeys)
    rEs = np.sqrt(fakeys**2 + fakezs**2)


    thefit = CubicSpline(thetaEs,rEs,bc_type='periodic')
    keepIdxs = []
    for i in range(len(thetas)):
        rlimit = thefit(thetas[i])
        if rs[i] < rlimit:
            keepIdxs.append(i)
    keepIdxs = np.array(keepIdxs)

    # throw out garbage far behind CME
    cleanx4 = x3[keepIdxs]
    xcutVal = np.median(cleanx4) - 2 * np.std(cleanx4)
    xcut = np.where(cleanx4 >= xcutVal)[0]
    keepIdxs = keepIdxs[xcut]
    cleanx4  = cleanx4[xcut]

    # Get the clean versions of the coords at diff projections
    cx, cy, cz = x[keepIdxs], y[keepIdxs], z[keepIdxs]
    cleany4, cleanz4 = y4[keepIdxs], z4[keepIdxs]
    cleany6, cleanz6 = y6[keepIdxs], z6[keepIdxs]
    ryz = rs[keepIdxs] # distance from axis through nose


    rxyz = np.sqrt(cx**2 + cy**2 + cz**2)
    maxrxyz = np.max(rxyz)
    maxryz = np.max(ryz)
    centidx = np.where((ryz < 0.25 * maxryz) & (rxyz > 0.5 * maxrxyz))

    # Don't know if 95 percentile is the best? want to elim some garbage but not shrink too much`
    rNose = np.percentile(cleanx4[centidx], 95)

    # get x dist of max width
    outidx = np.where(np.abs(cleany6) > 0.8 * np.max(np.abs(cleany6)))
    AWs = np.abs(np.arctan(cleany6/cleanx4)/dtor)
    AWps = np.abs(np.arctan(cleanz6/cleanx4)/dtor)
    AW = np.percentile(AWs, 95)
    AWp = np.percentile(AWps, 95)

    # get other final vals
    latf = meanlat/dtor + np.arctan(y0/rNose)/dtor
    lonf = meanlon/dtor + np.arctan(z0/rNose)/dtor
    tiltf = tilt/dtor + phi/dtor

    # switch things if fit swapped semi major/minor
    if flipTilt:
        temp = AWp
        AWp  = AW
        AW   = temp
        tiltf = tiltf + 90

    # force tilt to live in -90 to 90    
    if tiltf > 90:
        tiltf  -= 180
    elif tilt < -90:
        tiltf += 180
        
    # Get the mean density and total CME mass
    dens = densr2/r/r
    mean_dens = np.mean(dens[keepIdxs])
    # approx CME as ellipsoid 
    vol = (4/3)*np.pi*(a * 1.496e11)*(b* 1.496e11)**2
    # tomo dens is in e- per m^3 but use proton mass!
    massp = 1.67e-24
    meanDens = np.mean(dens[keepIdxs])*massp
    #totM2 = meanDens * vol  / 1e15
    dr = 1.496e11 / 200
    dp = 2*np.pi / 120.
    dq = np.pi / 60.
    pixVol = (rxyz*1.496e11)**2 * dr * dp * dq
    totM = np.sum(dens[keepIdxs] * massp * pixVol)/1e15
    
    # Print final results
    if not silent:
        print ('  ')
        print('Final results: ')
        print('Latitude: ', latf, ' deg')
        print('Longitude: ', lonf, ' deg')
        print('Tilt: ', tiltf, ' deg')
        print('Front Dist: ', rNose, ' au')
        print ('AW:', AW, ' deg')
        print ('AWp:', AWp , ' deg')
        print ('Semimajor: ', a, ' au')
        print ('Semiminor: ', b, ' au')
        print ('Dens:', meanDens/1e8, 'x10^8 g / cm3')
        print ('Mass:', totM, 'x10^15g')
        
    
    if saveFig:
       fig = plt.figure()
       ax = fig.add_subplot(projection='3d')
       ax.scatter(cx, cy, cz, c=np.log10(densr2[keepIdxs]))
       ax.axes.set_xlim3d(axlims[0][0], axlims[0][1]) 
       ax.axes.set_ylim3d(axlims[1][0], axlims[1][1]) 
       ax.axes.set_zlim3d(axlims[2][0], axlims[2][1])
       ax.set_aspect('equal')
       plt.savefig(figName+'B.png') 
    
    
    # This prints in nice format
    if printPretty:
        print('{:.1f}'.format(latf), '{:.1f}'.format(lonf),  '{:.1f}'.format(tiltf), '{:.3f}'.format(rNose), '{:.1f}'.format(AW), '{:.1f}'.format(AWp), '{:.4f}'.format(a), '{:.4f}'.format(b))
    
    # dens is in 1e-18 kg/m^3 and Mass is in 1e15 g    
    return [latf, lonf, tiltf, rNose, AW, AWp, a, b, meanDens*1e-3/1e-18, totM]


def loopIt(date):
    # Date should be a string of just the date (no /)
    dir_path = date+'/3d_densities/'
    # Check for fig folder
    if not os.path.exists(date+'/figs/'):
        os.mkdir(date+'/figs/')
    if not os.path.exists(date+'/fits/'):
        os.mkdir(date+'/fits/')
    files_dir = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
    for aDir in files_dir:
        f1 = open(date+'/fits/'+date+aDir+'_CPfits.dat', 'w')    
        allFiles = np.sort(os.listdir(dir_path+aDir))
        for aFile in allFiles:
            # Assuming everything has this format, can mod if changes
            if 'x2_solved_' in aFile:
                myTime = aFile[10:13]
                myName = date+'_'+aDir+'_'+myTime
                try:
                    res = fitCloudFun(dir_path+aDir+'/'+aFile, figName=date+'/figs/'+myName)
                except:
                    res = [99.9, 99.9, 99.9, 9.999, 99.9, 99.9, 9.999, 9.999, 99.99, 99.99]
                outstuff = [myName, '{:.3f}'.format(res[3]), '{:.1f}'.format(res[0]), '{:.1f}'.format(res[1]),  '{:.1f}'.format(res[2]),  '{:.1f}'.format(res[4]), '{:.1f}'.format(res[5]), '{:.4f}'.format(res[6]), '{:.4f}'.format(res[7]), '{:.2f}'.format(res[8]), '{:.2f}'.format(res[9])]
                outstr = ''
                for item in outstuff:
                    outstr += item + ' '
                print (outstr)
                f1.write(outstr+'\n')
        f1.close()
        
#print(fitCloudFun('20211028/3d_densities/003_ring/x2_solved_056_0400_0120_0060.dat'))
#loopIt('20170910')
#loopIt('20201129')
loopIt('20211028')