from numpy import *
import numpy as np
#from scipy import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from pylab import *
from shutil import copy
from os import mkdir
import shutil,os,glob
import scipy.signal as scisig
from maketopo import getTopo2D
import logging
from replace_data import replace_data
import sys

logging.basicConfig(level=logging.DEBUG)

_log = logging.getLogger(__name__)


def make_run(amp=200, u0=0, N0=1e-3, f0=1e-4, tideamp=0.025,
             tidefreq=0.000140752359, alongw=50e3, crossw=50e3,
             average_space=1000e3,
             comments='None', runtype='low'):


    runname='AHMany%04d_%03damp%03df%02daw%03dcw%03dN%04d'%(average_space/1e3, u0*100, amp, f0*10**5, alongw/1e3, crossw/1e3, N0*1e4)

    # to change U we need to edit external_forcing recompile

    outdir0='../results/'+runname+'/'

    indir =outdir0+'/indata/'

    ## Params for below as per Nikurashin and Ferrari 2010b
    H = 4000.
    U0 = u0

    # add tidal intitial condition:
    U0 += tideamp

    # need some info.  This comes from `leewave3d/EmbededRuns.ipynb` on `valdez.seos.uvic.ca`
    # the maxx and maxy are for finer scale runs.
    dx0=4000.
    dy0=4000.

    # reset f0 in data
    shutil.copy('data0', 'dataF0')
    replace_data('dataF0', 'f0', '%1.3e'%f0)

    shutil.copy('data1', 'dataF1')
    replace_data('dataF1', 'f0', '%1.3e'%f0)

    # set params in `data.btforcing`
    # NO BT FORCING!!!
    replace_data('data.btforcing', 'btforcingTideFreq', '%1.5e'%tidefreq)
    replace_data('data.btforcing', 'btforcingTideAmp', '%1.5e'%tideamp)
    replace_data('data.btforcing', 'btforcingU0', '%1.5e'%u0)


    # topography parameters:
    useFiltTop=False
    useLowTopo=False
    gentopo=False # generate the topography.
    if runtype=='full':
        gentopo=True
    if runtype=='filt':
        useFiltTop=True
    elif runtype=='low':
        useLowTopo=True


    # model size
    nx = 8*82
    ny = 8*52
    nz = 200

    _log.info('nx %d ny %d', nx, ny)

    #### Set up the output directory
    backupmodel=1
    if backupmodel:
      try:
        mkdir(outdir0)
      except:
        import datetime
        import time
        ts = time.time()
        st=datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
        shutil.move(outdir0[:-1],outdir0[:-1]+'.bak'+st)
        mkdir(outdir0)

        _log.info(outdir0+' Exists')

      outdir=outdir0
      try:
        mkdir(outdir)
      except:
        _log.info(outdir+' Exists')
      outdir=outdir+'input/'
      try:
        mkdir(outdir)
      except:
        _log.info(outdir+' Exists')
      try:
          mkdir(outdir+'/figs/')
      except:
        pass

      copy('gendata.py',outdir)
    else:
      outdir=outdir+'input/'

    ## Copy some other files
    _log.info( "Copying files")

    try:
      shutil.rmtree(outdir+'/../code/')
    except:
      _log.info("code is not there anyhow")
    shutil.copytree('../code', outdir+'/../code/')
    shutil.copytree('../python', outdir+'/../python/')

    try:
      shutil.rmtree(outdir+'/../build/')
    except:
      _log.info("build is not there anyhow")
    _log.info(outdir+'/../build/')
    mkdir(outdir+'/../build/')

    # copy any data that is in the local indata
    shutil.copytree('../indata/', outdir+'/../indata/')

    shutil.copy('../build/mitgcmuv', outdir+'/../build/mitgcmuv')
    #shutil.copy('../build/mitgcmuvU%02d'%u0, outdir+'/../build/mitgcmuv%02d'%u0)
    shutil.copy('../build/Makefile', outdir+'/../build/Makefile')
    shutil.copy('dataF0', outdir+'/data0')
    shutil.copy('dataF1', outdir+'/data1')
    shutil.copy('eedata', outdir)
    shutil.copy('data.kl10', outdir)
    shutil.copy('data.btforcing', outdir)
    try:
      shutil.copy('data.kpp', outdir)
    except:
      pass
    #shutil.copy('data.rbcs', outdir)
    try:
        shutil.copy('data.obcs', outdir)
    except:
        pass
    try:
      shutil.copy('data.diagnostics', outdir)
    except:
      pass
    try:
      shutil.copy('data.pkg', outdir+'/data.pkg')
    except:
      pass
    try:
      shutil.copy('data.rbcs', outdir+'/data.rbcs')
    except:
      pass

    _log.info("Done copying files")

    ####### Make the grids #########

    # Make grids:

    ##### Dx ######

    expand_fac = 1.04
    r = 80

    dx = np.zeros(nx) + dx0
    for ind in range(nx-2*r, nx-r):
        dx[ind] = dx[ind-1] * expand_fac
    for ind in range(nx-r, nx):
        dx[ind] = dx[ind-1] / expand_fac
    dx = np.roll(dx, r)

    # dx = zeros(nx)+100.
    x=np.cumsum(dx)
    x=x-x[0]
    maxx=np.max(x)
    _log.info('XCoffset=%1.4f'%x[0])

    ##### Dy ######


    dy = np.zeros(ny) + dy0
    for ind in range(ny-2*r, ny-r):
        dy[ind] = dy[ind-1] * expand_fac
    for ind in range(ny-r, ny):
        dy[ind] = dy[ind-1] / expand_fac
    dy = np.roll(dy, r)

    # dx = zeros(nx)+100.
    y=np.cumsum(dy)
    y=y-y[0]
    maxy=np.max(y)
    _log.info('YCoffset=%1.4f'%y[0])

    _log.info('dx %f dy %f', dx[0], dy[0])


    # save dx and dy
    with open(indir+"/delX.bin", "wb") as f:
      dx.tofile(f)
    f.close()
    with open(indir+"/delY.bin", "wb") as f:
      dy.tofile(f)
    f.close()
    # some plots
    fig, ax = plt.subplots(2,1)
    ax[0].plot(x/1000.,dx)
    ax[1].plot(y/1000.,dy)
    #xlim([-50,50])
    fig.savefig(outdir+'/figs/dx.pdf')

    ######## Bathy ############
    # get the topo:
    X, Y = np.meshgrid(x, y)
    d=zeros((ny,nx))

    # figure out how many:
    nhx = np.round((x[-1] - x[0])/average_space)
    nhy = np.round((y[-1] - y[0])/average_space)
    Nhills = nhx * nhy
    _log.info(f'Making {Nhills} hills')
    np.random.seed(20200615)
    for nh in np.arange(Nhills):
        x0 = np.random.rand(1) * (x[-1] - x[0]) + x[0]
        y0 = np.random.rand(1) * (y[-1] - y[0]) + y[0]
        if nh == 0:
            x0 = (x[-1] - x[0]) / 2
            y0 = (y[-1] - y[0]) / 2
        print(x0, y0)
        h = np.exp(-((X - x0)/alongw)**2) * amp
        indy = np.where((Y[:, 0] - y0) > crossw)[0];
        if len(indy) > 0:

            yy = Y[indy, :] - Y[indy[0], :]
            h[indy,:] = h[indy, :] * np.exp(-(yy/alongw)**2)
        indy = np.where(Y[:, 0] - y0 < -crossw)[0];
        if len(indy) > 0:
            yy = Y[indy, :] - Y[indy[-1], :]
            h[indy,:] = h[indy, :] * np.exp(-(yy/alongw)**2)

        d += h

    # we will add a seed just in case we want to redo this exact phase later...


    d = d - H

    # put a wall at top to stop waves across overlap...

    d[0, :] = 0
    with open(indir+"/topog.bin", "wb") as f:
      d.tofile(f)
    f.close()

    _log.info(shape(d))

    fig, ax = plt.subplots(2,1)
    _log.info('%s %s', shape(x),shape(d))
    ax[0].plot(x/1.e3,d[0,:].T)
    pcm=ax[1].pcolormesh(x/1.e3,y/1.e3,d,rasterized=True, vmax=-3000)
    fig.colorbar(pcm,ax=ax[1])
    fig.savefig(outdir+'/figs/topo.png')


    ##################
    # dz:
    # dz is from the surface down (right?).  Its saved as positive.
    dz = ones((1,nz))*H/nz

    with open(indir+"/delZ.bin", "wb") as f:
    	dz.tofile(f)
    f.close()
    z=np.cumsum(dz)

    ####################
    # temperature profile...
    #
    # temperature goes on the zc grid:
    g=9.8
    alpha = 2e-4
    T0 = 28+cumsum(N0**2/g/alpha*(-dz))

    with open(indir+"/TRef.bin", "wb") as f:
    	T0.tofile(f)
    f.close()
    #plot
    plt.clf()
    plt.plot(T0,z)
    plt.savefig(outdir+'/figs/TO.pdf')

    ###########################
    # velcoity data
    aa = np.zeros((nz,ny,nx))
    for i in range(nx):
        aa[:,:,i]=U0
    with open(indir+"/Uforce.bin", "wb") as f:
        aa.tofile(f)
    aa = 0 * aa
    with open(indir+"/Vforce.bin", "wb") as f:
        aa.tofile(f)




    ########################
    # RBCS sponge and forcing
    # In data.rbcs, we have set tauRelaxT=17h = 61200 s
    # here we wil set the first and last 50 km in *y* to relax at this scale and
    # let the rest be free.


    aa = np.zeros((nz,ny,nx))

    ysponge = 0 * X
    spongew = 28

    ysponge[:spongew,:] = (np.arange(spongew, 0, -1) / spongew)[:, np.newaxis]

    ysponge[-(spongew+1):-1,:] = (np.arange(0, spongew) / spongew)[:, np.newaxis]

    xsponge = 0 * X

    xsponge[:, :spongew] += (np.arange(spongew, 0, -1) / spongew)[np.newaxis, :]

    xsponge[:, -(spongew+1):-1] += (np.arange(0, spongew) / spongew)[np.newaxis, :]
    # make the y sponge weaker than xsponge, since things tend to transit along the
    # sides for a long time....
    # reinstate the y-sponge
    sponge = np.maximum(ysponge / 3.0, xsponge)

    for i in range(nz):
        aa[i, :, :] = sponge[:, :]

    plt.clf()
    plt.pcolormesh(x-np.mean(x), y - np.mean(y), sponge, rasterized=True, vmin=0, vmax=1)
    plt.savefig(outdir+'/figs/sponge.pdf')


    with open(indir+"/spongeweight.bin", "wb") as f:
        aa.tofile(f)
    f.close()

    aa=np.zeros((nz,ny,nx))
    aa+=T0[:,newaxis,newaxis]
    _log.info(shape(aa))

    with open(indir+"/Tforce.bin", "wb") as f:
        aa.tofile(f)
    f.close()



    ###### Manually make the directories
    #for aa in range(128):
    #    try:
    #        mkdir(outdir0+'%04d'%aa)
    #    except:
    #        pass

    _log.info('Writing info to README')
    ############ Save to README
    with open('README','r') as f:
      data=f.read()
    with open('README','w') as f:
      import datetime
      import time
      ts = time.time()
      st=datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
      f.write( st+'\n')
      f.write( outdir+'\n')
      f.write(comments+'\n\n')
      f.write(data)

    _log.info('All Done!')

    _log.info('Archiving to home directory')

    try:
        shutil.rmtree('../archive/'+runname)
    except:
        pass

    shutil.copytree(outdir0+'/input/', '../archive/'+runname+'/input')
    shutil.copytree(outdir0+'/python/', '../archive/'+runname+'/python')
    shutil.copytree(outdir0+'/code', '../archive/'+runname+'/code')

    print('DONE')
    print(f'Run as: qsub -N {runname} runModel.sh')

# make_run(amp=305, u0=0.0, tideamp=0.00)

if 0:
    make_run(amp=600, u0=0.1, tideamp=0.00, alongw=50e3, crossw=50e3)
    make_run(amp=400, u0=0.1, tideamp=0.00, alongw=50e3, crossw=50e3)
    make_run(amp=400, u0=0.1, tideamp=0.00, alongw=50e3, crossw=50e3, N0=5e-4)
    make_run(amp=400, u0=0.1, tideamp=0.00, alongw=50e3, crossw=50e3, N0=2e-3)
    make_run(amp=200, u0=0.1, tideamp=0.00, alongw=50e3, crossw=50e3)
    make_run(amp=400, u0=-0.15, tideamp=0.00, alongw=50e3, crossw=50e3, N0=1e-3)
    make_run(amp=400, u0=0.15, tideamp=0.00, alongw=50e3, crossw=50e3, N0=1e-3)
make_run(amp=400, u0=0.2, tideamp=0.00, alongw=50e3, crossw=50e3, N0=1e-3, average_space=250e3)
make_run(amp=400, u0=0.1, tideamp=0.00, alongw=50e3, crossw=50e3, N0=1e-3, average_space=250e3)
make_run(amp=400, u0=0.05, tideamp=0.00, alongw=50e3, crossw=50e3, N0=1e-3, average_space=250e3)

#make_run(amp=320, u0=0.07, tideamp=0.00, alongw=50e3, crossw=50e3)
#make_run(amp=320, u0=0.1, tideamp=0.00, alongw=50e3, crossw=50e3, N0=2e-3)
#make_run(amp=320, u0=0.1, tideamp=0.00, alongw=50e3, crossw=50e3, N0=5e-4)
