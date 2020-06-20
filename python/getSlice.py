import xarray as xr

#for f in ['038', '073', '100', '126', '141']:
if True:
    td = 'AHIu0010amp305f10aw050cw050N0010'
    todo = '../results/{}/input/spinup.nc'.format(td)

    with xr.open_dataset(todo) as ds:
        print(ds)
        ds = ds.isel(j=200, j_g=200)
        ds.to_netcdf('../reduceddata/Slice200{}.nc'.format(td))
