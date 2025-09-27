import rioxarray as rxr


WGS84_EPSG = 'EPSG:4326' # The EPSG for WGS-84 lat/lon is 4326 


def parse_raster(fname: str,
                 x_lim: list=None,
                 y_lim: list=None) -> rxr.rioxarray.raster_dataset.xarray.DataArray:
    '''
    Read-in raster file, crop if x or y limits are provided and return
    as a rioxarray object

    Parameters
    ----------
    fname
        File name/path of the raster file
    x_lim
        Optional 1x2 array-like object that defines the minimum
        and maximum x coordinates the raster can have --> [min x, max x]
    y_lim
        Optional 1x2 array-like object that defines the minimum
        and maximum y coordinates the raster can have --> [min y, max y]

    Returns
    -------
    rxr.rioxarray.raster_dataset.xarray.DataArray
        DataArray of the (optionally) cropped raster
    '''

    ds = rxr.open_rasterio(fname, masked=True, decode_coords="all")

    if x_lim is not None:
        x_min  = x_lim[0]
        x_max  = x_lim[1]
        mask_x = (ds.x >= x_min) & (ds.x <= x_max)
    else:
        mask_x = ds.all()

    if y_lim is not None:
        y_min  = y_lim[0]
        y_max  = y_lim[1]
        mask_y = (ds.y >= y_min) & (ds.y <= y_max)
    else:
        mask_y = ds.all()

    return ds.where(mask_x & mask_y, drop=True)


if __name__ == '__main__':
    from os.path import dirname, join
    
    import matplotlib.pylab as plt
    
    
    SRC_DIR  = dirname(dirname(__file__))
    BASE_DIR = dirname(SRC_DIR)
    DATA_DIR = join(BASE_DIR, 'data')
    TEST_DIR = join(DATA_DIR, 'test')
    
    plt.ion()
    
    
    map_fname = join(TEST_DIR, r'R2508-Medium-Altitude_MAG-asc.gxf')
    
    # Establish x and y limits (in m) to crop the map
    x_lim = [  409_060.0,   498_650.0] # [min, max]
    y_lim = [3_878_380.0, 3_938_030.0] # [min, max]
    
    map = parse_raster(map_fname, False, x_lim, y_lim)
    map.plot()
    plt.show(block=True)