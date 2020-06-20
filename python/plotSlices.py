import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import os


run = 'AHIu0010amp310f10aw050cw050N0010'

try:
    os.makedirs(f'../reduceddata/{run}/slices/zoom/')
    os.makedirs(f'../reduceddata/{run}/slices')
except OSError as e:
    pass

with xr.open_dataset(f'../results/{run}/input/spinup.nc' ) as ds3d:
    for ind in range(0, ds3d.sizes['record']):
        print('ind', ind)
        print(ds3d)
        fig, axs = plt.subplots(3, 2, figsize=(6, 6), sharex=True, sharey=True, constrained_layout=True)
        x = ds3d['XC'][0, :] / 1e3
        x = x - x.mean()
        z = ds3d['Z']

        for nn, yind in enumerate([320, 400, 480]):

            ds = ds3d.isel(j=yind, j_g=yind)
            levels = np.sort(ds['THETA'][0, :, 0].values)[::10]
            vr = 0.2

            ax = axs[nn, 0]
            print(ds['UVEL'][ind, :, :].where(ds['THETA'][ind, :, :].values>0))
            pc = ax.pcolormesh(x, z, ds['UVEL'][ind, :, :].where(ds['THETA'][ind, :, :].values>0)-0.1, cmap='RdBu_r', vmin=-vr, vmax=vr, )
            ax.contour(x, z,
                       ds['THETA'][ind, :, :].where(ds['THETA'][ind, :, :]>0), levels=levels,
                       colors='0.5', linewidths=0.6)
            ax.set_facecolor('0.3')

            ax = axs[nn, 1]
            pc = ax.pcolormesh(x, z, ds['VVEL'][ind, :, :].where(ds['THETA'][ind, :, :].values>0), cmap='RdBu_r', vmin=-vr, vmax=vr, )
            ax.contour(x, z,
                       ds['THETA'][ind, :, :].where(ds['THETA'][ind, :, :]>0), levels=levels,
                       colors='0.5',linewidths=0.6)

        ax.set_xlim(-200, 200)
        fig.savefig(f'../reduceddata/{run}/slices/slices{ind:04d}.png', dpi=300)
        ax.set_ylim(-4000, -2800)
        ax.set_facecolor('0.3')
        fig.savefig(f'../reduceddata/{run}/slices/zoom/slices{ind:04d}.png', dpi=300)
        plt.clf()
        plt.close()

        #dsakj
