
"""
Generation of gridded GEDI data products
"""
# This file is part of GEDIPy
# Copyright (C) 2020

import numpy
import pyproj

import rasterio
from rasterio.crs import CRS
from affine import Affine
from scipy import ndimage


GEDIPY_EASE2_PAR = {'second_reference_latitude': 30.0, 'map_equatorial_radius_m': 6378137.0,
                    'map_eccentricity': 0.081819190843, 'binsize': 1000.8950233495561,
                    'xmin': -17367530.45, 'ymax': 7314540.83, 'ncol': 34704, 'nrow': 14616}

GEDIPY_RIO_DEFAULT_PROFILE = {'driver': 'GTiff', 'dtype': 'uint16', 'nodata': 65535.0, 'width': 34704,
                              'height': 14616, 'count': 3, 'crs': CRS.from_epsg(6933),
                              'transform': Affine(1000.8950233495561, 0.0, -17367530.4451615, 0.0, -1000.8950233495561, 7314540.8306386),
                              'blockxsize': 256, 'blockysize': 256, 'tiled': True, 'compress': 'deflate', 'interleave': 'pixel'}


class GEDIGrid:
    def __init__(self, filename, profile=GEDIPY_RIO_DEFAULT_PROFILE, gedi_domain=52):
        self.filename = filename
        self.profile = profile
        self.inproj = pyproj.Proj(init='epsg:4326')
        self.outproj = pyproj.Proj(init=str(profile.crs))
        self.gedi_domain = gedi_domain

    def init_gedi_ease2_grid(self, bbox=None, resolution=1):
        """
        Prepare the output image
        """
        if resolution in (1,3,9,18,24):
            ncol = int(GEDIPY_EASE2_PAR['ncol'] / resolution)
            nrow = int(GEDIPY_EASE2_PAR['nrow'] / resolution)
            binsize = GEDIPY_EASE2_PAR['binsize'] * resolution
        else:
            print('Only 1, 3, 9, 18, and 24 km resolutions accepted')
            exit(1)

        e2 = GEDIPY_EASE2_PAR['map_eccentricity']**2.0
        e4 = GEDIPY_EASE2_PAR['map_eccentricity']**4.0
        e6 = GEDIPY_EASE2_PAR['map_eccentricity']**6.0
        sin_phi1 = numpy.sin( numpy.radians(GEDIPY_EASE2_PAR['second_reference_latitude']) )
        cos_phi1 = numpy.cos( numpy.radians(GEDIPY_EASE2_PAR['second_reference_latitude']) )
        kz = cos_phi1 / numy.sqrt( 1.0 - e2 * sin_phi1**2 )

        x, y = numpy.mgrid[0:ncol,0:nrow]
        qp = ( ( 1.0 - e2 ) * ( ( 1.0 / ( 1.0 - e2 ) ) - ( 1.0 / ( 2.0 * GEDIPY_EASE2_PAR['map_eccentricity'] ) ) * 
             numpy.log( ( 1.0 - GEDIPY_EASE2_PAR['map_eccentricity'] ) / ( 1.0 + GEDIPY_EASE2_PAR['map_eccentricity'] ) ) ) )
        beta = numpy.arcsin( 2.0 * y * kz / ( GEDIPY_EASE2_PAR['map_equatorial_radius_m'] * qp ) )

        lam = x / ( GEDIPY_EASE2_PAR['map_equatorial_radius_m'] * kz )
        phi = ( beta + ( ( ( e2 / 3.0 ) + ( ( 31.0 / 180.0 ) * e4 ) + ( ( 517.0 / 5040.0 ) * e6 ) ) * 
              numpy.sin( 2.0 * beta ) ) + ( ( ( ( 23.0 / 360.0 ) * e4) + ( ( 251.0 / 3780.0 ) * e6 ) ) * 
              numpy.sin( 4.0 * beta ) ) + ( ( 761.0 / 45360.0 ) * e6 ) * numpy.sin( 6.0 * beta ) )

        latgrid = numpy.degrees(phi)
        longrid = numpy.degrees(lam)
        while numpy.any(longrid < -180):
            longrid[longrid < -180] += 360
        while numpy.any(longrid > 180):
            longrid[longrid > 180] -= 360

        if bbox:
            mask = (latgrid < bbox[1]) & (latgrid >= bbox[3]) & (longrid < bbox[2]) & (longrid >= bbox[0])
            idx = numpy.ix_(mask.any(1),mask.any(0))
            xmin = GEDIPY_EASE2_PAR['xmin'] + numpy.min(idx[1]) * binsize
            ymax = GEDIPY_EASE2_PAR['ymax'] - numpy.min(idx[0]) * binsize
            latgrid = latgrid[idx]
        else:
            xmin = GEDIPY_EASE2_PAR['xmin']
            ymax = GEDIPY_EASE2_PAR['ymax']

        self.outimage = numpy.zeros((self.profile['count'], latgrid.shape[0], latgrid.shape[1]), 
                                    dtype=self.profile['dtype'])
        mask = numpy.expand_dims((latgrid > self.domain) | (latgrid < -self.domain))
        self.outimage[mask] = self.profile['nodata']
        
        self.profile.update(height=self.outgrid.shape[1], width=self.outgrid.shape[2], 
            transform=Affine(binsize, 0.0, xmin, 0.0, binsize, ymax))

    def zoom_gedi_grid(self, resolution=1000):
        if resolution in (500,100):
            res_factor = numpy.floor(1000 / resolution)
        else:
            print('Only 500 m and 100 m resolutions accepted.')
            exit(1)
        
        self.outgrid = ndimage.zoom(self.outgrid, res_factor, order=1, mode='nearest')
        binsize = GEDIPY_EASE2_PAR['binsize'] / res_factor
        
        self.profile.update(height=self.outgrid.shape[1], width=self.outgrid.shape[2],
            transform=Affine(binsize, 0.0, self.profile[2], 0.0, binsize, self.profile[5])) 
    
    def write_grid(self):
        with rasterio.Env():
            with rasterio.open(self.filename, 'w', **self.profile) as dst:
                dst.write(self.outgrid)

    def add_data(self, longitude, latitude, dataset, func='simple_stats'):
        valid = ~numpy.isnan(longitude) & ~numpy.isnan(latitude) & ~numpy.isnan(dataset)
        if numpy.any(valid):
            x,y = pyproj.transform(self.inproj, self.outproj, longitude[valid], latitude[valid])
            func_method = getattr(userfunctions, func)
            func_method(x, y, dataset[valid], self.outimage, self.xmin, self.ymax, self.binsize)

