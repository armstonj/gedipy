
"""
Generation of gridded GEDI data products
"""
# This file is part of GEDIPy
# Copyright (C) 2020

import numpy
import json
from pyproj import Transformer

import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from affine import Affine
from scipy import ndimage
from scipy import stats

from . import userfunctions

GEDIPY_EASE2_PAR = {'second_reference_latitude': 30.0, 'map_equatorial_radius_m': 6378137.0,
                    'map_eccentricity': 0.081819190843, 'binsize': [1000.895023349556141,1000.895023349562052],
                    'xmin': -17367530.445161499083042, 'ymax': 7314540.830638599582016, 'ncol': 34704, 'nrow': 14616}

GEDIPY_RIO_DEFAULT_PROFILE = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': -9999.0, 'width': 34704,
                              'height': 14616, 'count': 5, 'crs': CRS.from_epsg(6933),
                              'transform': Affine(1000.895023349556141, 0.0, -17367530.445161499083042, 0.0, -1000.895023349562052, 7314540.8306386),
                              'blockxsize': 256, 'blockysize': 256, 'tiled': True, 'compress': 'deflate', 'interleave': 'pixel'}


class GEDIGrid:
    def __init__(self, profile=GEDIPY_RIO_DEFAULT_PROFILE):
        self.profile = profile

    def rowcol_to_wgs84(self, rows, cols, binsize):
        """
        Function to convert EASE 2.0 grid row/cols to WGS84 (EPSG:4326 coordinates)
        """
        e2 = GEDIPY_EASE2_PAR['map_eccentricity']**2.0
        e4 = GEDIPY_EASE2_PAR['map_eccentricity']**4.0
        e6 = GEDIPY_EASE2_PAR['map_eccentricity']**6.0
        sin_phi1 = numpy.sin( numpy.radians(GEDIPY_EASE2_PAR['second_reference_latitude']) )
        cos_phi1 = numpy.cos( numpy.radians(GEDIPY_EASE2_PAR['second_reference_latitude']) )
        kz = cos_phi1 / numpy.sqrt( 1.0 - e2 * sin_phi1**2 )

        s0 = ( cols.size - 1 ) / 2.0
        r0 = ( rows.size - 1 ) / 2.0
        x = ( cols - s0 ) * binsize[0]
        y = ( r0 - rows ) * binsize[1]

        qp = ( ( 1.0 - e2 ) * ( ( 1.0 / ( 1.0 - e2 ) ) - ( 1.0 / ( 2.0 * GEDIPY_EASE2_PAR['map_eccentricity'] ) ) *
             numpy.log( ( 1.0 - GEDIPY_EASE2_PAR['map_eccentricity'] ) / ( 1.0 + GEDIPY_EASE2_PAR['map_eccentricity'] ) ) ) )
        beta = numpy.arcsin( 2.0 * y * kz / ( GEDIPY_EASE2_PAR['map_equatorial_radius_m'] * qp ) )
        phi = ( beta + ( ( ( e2 / 3.0 ) + ( ( 31.0 / 180.0 ) * e4 ) + ( ( 517.0 / 5040.0 ) * e6 ) ) *
              numpy.sin( 2.0 * beta ) ) + ( ( ( ( 23.0 / 360.0 ) * e4) + ( ( 251.0 / 3780.0 ) * e6 ) ) *
              numpy.sin( 4.0 * beta ) ) + ( ( 761.0 / 45360.0 ) * e6 ) * numpy.sin( 6.0 * beta ) )
        lam = x / ( GEDIPY_EASE2_PAR['map_equatorial_radius_m'] * kz )

        longitude = numpy.degrees(lam)
        latitude = numpy.degrees(phi)
        while numpy.any(longitude < -180):
            longitude[longitude < -180] += 360
        while numpy.any(longitude > 180):
            longitude[longitude > 180] -= 360

        return longitude, latitude

    def init_grid_from_config(self, grid_config, **kwargs):
        """
        Pepare the output grid using a JSON config file
        """
        if not isinstance(grid_config, dict):
            with open(grid_config, 'r') as f:
                grid_config = json.load(f)

        if 'epsg' in grid_config.keys():
            self.profile['crs'] = CRS.from_epsg(grid_config['epsg'])
        if grid_config.keys() >= {'pixelxsize','pixelysize','ulx','uly'}:
            self.profile['transform'] = Affine(grid_config['pixelxsize'], 0.0, grid_config['ulx'],
                0.0, -grid_config['pixelysize'], grid_config['uly'])
        for key,value in grid_config.items():
            if key in self.profile:
                self.profile[key] = value

        for key,value in kwargs.items():
            if key in self.profile:
                self.profile[key] = value

        self.outgrid = numpy.zeros((self.profile['count'], self.profile['height'], self.profile['width']),
            dtype=self.profile['dtype'])

    def init_grid_from_reference(self, reference_image, **kwargs):
        """
        Prepare the output grid using a reference image
        """
        with rasterio.open(reference_image) as src:
            self.profile = src.profile

        for key,value in kwargs.items():
            if key in self.profile:
                self.profile[key] = value

        self.outgrid = numpy.zeros((self.profile['count'], self.profile['height'], self.profile['width']),
            dtype=self.profile['dtype'])

    def init_gedi_ease2_grid(self, bbox=None, tile=None, tilesize=72, resolution=1000, gedi_domain=52, **kwargs):
        """
        Prepare the output GEDI grid
        """
        if resolution not in (100,500,1000,3000,9000,18000,24000):
            print('Only 100 m, 500 m, 1 km, 3 km, 9 km, 18 km, and 24 km resolutions accepted')
            exit(1)

        if resolution < 1000:
            rkm = 1.0
        else:
            rkm = resolution / 1000

        ncol = int(GEDIPY_EASE2_PAR['ncol'] / rkm)
        nrow = int(GEDIPY_EASE2_PAR['nrow'] / rkm)
        binsize = [b * rkm for b in GEDIPY_EASE2_PAR['binsize']]

        rows = numpy.arange(nrow, dtype=numpy.float32)
        cols = numpy.arange(ncol, dtype=numpy.float32)
        longrid, latgrid = self.rowcol_to_wgs84(rows, cols, binsize)

        if bbox:
            xmask = (longrid < bbox[2]) & (longrid >= bbox[0])
            ncol = numpy.count_nonzero(xmask)
            ymask = (latgrid < bbox[1]) & (latgrid >= bbox[3])
            nrow = numpy.count_nonzero(ymask)

            idx = numpy.ix_(xmask,ymask)
            xmin = GEDIPY_EASE2_PAR['xmin'] + numpy.min(idx[0]) * binsize[0]
            ymax = GEDIPY_EASE2_PAR['ymax'] - numpy.min(idx[1]) * binsize[1]

            longrid = longrid[xmask]
            latgrid = latgrid[ymask]
        elif tile:
            ncol = nrow = int(tilesize / rkm)
            xmin = GEDIPY_EASE2_PAR['xmin'] + (tile[0] - 1) * binsize[0] * tilesize
            ymax = GEDIPY_EASE2_PAR['ymax'] - (tile[1] - 1) * binsize[1] * tilesize

            c0 = (tile[0] - 1) * ncol 
            r0 = (tile[1] - 1) * nrow
            longrid = longrid[r0:r0+nrow]
            latgrid = latgrid[c0:c0+ncol]
        else:
            xmin = GEDIPY_EASE2_PAR['xmin']
            ymax = GEDIPY_EASE2_PAR['ymax']

        self.profile.update(height=nrow, width=ncol,
            transform=Affine(binsize[0], 0.0, xmin, 0.0, -binsize[1], ymax))
        for key,value in kwargs.items():
            if key in self.profile:
                self.profile[key] = value

        self.outgrid = numpy.zeros((self.profile['count'], nrow, ncol), dtype=self.profile['dtype'])

        self.gedimask = numpy.ones((1, nrow, ncol), dtype=numpy.bool)
        idx = numpy.argwhere((latgrid > gedi_domain) | (latgrid < -gedi_domain))
        self.gedimask[0,idx,:] = False

        if resolution < 1000:
            self.zoom_gedi_grid(resolution=resolution)

    def zoom_gedi_grid(self, resolution=1000):
        if resolution in (100,500):
            res_factor = numpy.floor(1000 / resolution)
        else:
            print('Only 500 m and 100 m resolutions accepted.')
            exit(1)

        self.outgrid = ndimage.zoom(self.outgrid, [1, res_factor, res_factor], order=0, mode='nearest')
        if hasattr(self, 'gedimask'):
            self.gedimask = ndimage.zoom(self.gedimask, [1, res_factor, res_factor], order=0, mode='nearest')

        binsize = [b / res_factor for b in GEDIPY_EASE2_PAR['binsize']]

        self.profile.update(height=self.outgrid.shape[1], width=self.outgrid.shape[2],
            transform=Affine(binsize[0], 0.0, self.profile['transform'][2], 0.0,
                             -binsize[1], self.profile['transform'][5]))

    def reset_grid(self):
        self.outgrid = numpy.zeros((self.profile['count'], self.profile['height'], self.profile['width']),
            dtype=self.profile['dtype'])

    def write_grid(self, filename, descriptions=None):
        if hasattr(self, 'gedimask'):
            self.outgrid = numpy.where(self.gedimask, self.outgrid, self.profile['nodata'])

        with rasterio.Env():
            with rasterio.open(filename, 'w', **self.profile) as dst:
                dst.write(self.outgrid)
                dst.build_overviews([2,4,8,16], Resampling.average)
                if descriptions:
                    for i in range(self.outgrid.shape[0]):
                        dst.set_band_description(i+1, descriptions[i])


    def add_data(self, longitude, latitude, dataset, func='grid_moments', **kwargs):
        valid = ~numpy.isnan(longitude) & ~numpy.isnan(latitude) & ~numpy.isnan(dataset)
        if numpy.any(valid):
            transformer = Transformer.from_crs('epsg:4326', str(self.profile['crs']), always_xy=True)
            x,y = transformer.transform(longitude[valid], latitude[valid])
            if func == 'grid_moments':
                userfunctions.grid_moments(x, y, dataset[valid], self.outgrid,
                    self.profile['transform'][2], self.profile['transform'][5],
                    self.profile['transform'][0], -self.profile['transform'][4])
            elif func == 'grid_counts':
                userfunctions.grid_counts(x, y, dataset[valid], self.outgrid,
                    self.profile['transform'][2], self.profile['transform'][5],
                    self.profile['transform'][0], -self.profile['transform'][4])
            elif func == 'grid_quantiles':
                userfunctions.grid_quantiles(x, y, dataset[valid], self.outgrid,
                    self.profile['transform'][2], self.profile['transform'][5],
                    self.profile['transform'][0], -self.profile['transform'][4],
                    kwargs['quantiles'], kwargs['step'])
            else:
                x_edge = numpy.sort(self.profile['transform'][2] + numpy.arange(self.profile['width'] + 1) *
                    self.profile['transform'][0])
                y_edge = numpy.sort(self.profile['transform'][5] + numpy.arange(self.profile['height'] + 1) *
                    self.profile['transform'][4])
                tmp, x_edge, y_edge, binnumber = stats.binned_statistic_2d(x, y, dataset[valid],
                    statistic=func, bins=[x_edge, y_edge])
                tmp = numpy.expand_dims(tmp.T[::-1,...], axis=0)
                self.outgrid = numpy.where(numpy.isnan(tmp), self.profile['nodata'], tmp)


    def finalize_grid(self, gain=1, offset=0, func='grid_moments'):
        """
        Finalize statistics and scale and o
        """
        if func == 'grid_moments':

            # Initialize the output
            tmpshape = (4, self.outgrid.shape[1], self.outgrid.shape[2])
            tmpgrid = numpy.empty(tmpshape, dtype=self.outgrid.dtype)

            # Mean
            tmpgrid[0] = numpy.where(self.outgrid[4] > 0, self.outgrid[0], self.profile['nodata'])

            # Standard deviation
            tmp = numpy.full(self.outgrid[1].shape, self.profile['nodata'], dtype=self.outgrid.dtype)
            numpy.divide(self.outgrid[1], self.outgrid[4] - 1, out=tmp, where=self.outgrid[4] > 1)
            numpy.sqrt(tmp, out=tmp, where=self.outgrid[4] > 1)
            tmpgrid[1] = tmp

            # Skewness
            tmp = numpy.full(self.outgrid[2].shape, self.profile['nodata'], dtype=self.outgrid.dtype)
            numpy.sqrt(self.outgrid[4], out=tmp, where=self.outgrid[4] > 2)
            numpy.multiply(tmp, self.outgrid[2], out=tmp, where=self.outgrid[4] > 2)
            numpy.divide(tmp, self.outgrid[1]**1.5, out=tmp, where=self.outgrid[4] > 2)
            tmpgrid[2] = tmp

            # Kurtosis
            tmp = numpy.full(self.outgrid[3].shape, self.profile['nodata'], dtype=self.outgrid.dtype)
            numpy.multiply(self.outgrid[4], self.outgrid[3], out=tmp, where=self.outgrid[4] > 3)
            numpy.divide(tmp, self.outgrid[1]**2, out=tmp, where=self.outgrid[4] > 3)
            numpy.subtract(tmp, 3, out=tmp, where=self.outgrid[4] > 3)
            tmpgrid[3] = tmp

            # Scale and offset
            for i in range(tmpgrid.shape[0]):
                tmp = tmpgrid[i]
                numpy.multiply(tmp, gain, out=tmp, where=self.outgrid[4] > i)
                numpy.add(tmp, offset, out=tmp, where=self.outgrid[4] > i)
                self.outgrid[i] = tmp.astype(self.profile['dtype'])

        elif func == 'grid_quantiles':

            for i in range(self.outgrid.shape[0] - 1):
                tmp = self.outgrid[i]
                numpy.multiply(tmp, gain, out=tmp, where=self.outgrid[-1] > 0)
                numpy.add(tmp, offset, out=tmp, where=self.outgrid[-1] > 0)
                self.outgrid[i] = tmp.astype(self.profile['dtype'])

        else:

            numpy.multiply(self.outgrid, gain, out=self.outgrid,
                where=self.outgrid != self.profile['nodata'])
            numpy.add(self.outgrid, offset, out=self.outgrid,
                where=self.outgrid != self.profile['nodata'])
            self.outgrid = self.outgrid.astype(self.profile['dtype'])

        if hasattr(self, 'gedimask'):
            self.outgrid = numpy.where(self.gedimask, self.outgrid, self.profile['nodata'])


