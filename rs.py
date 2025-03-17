import os
import numpy as np
from osgeo import gdal
from osgeo import osr
from osgeo import gdalnumeric, ogr
from osgeo.gdalconst import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageDraw

class RSImage(object):
    """
    Image class

    This class is initialized by opening file and load it onto memory.
    Below information is updated after initialization

    self.ncol : Number of columns
    self.nrow : Number of row
    self.nband :Number of band
    self.dtype : Data type
    self.img : Numpy array contains remote sensing data
    """

    def __init__(self, fn):
        self.fn = fn

        # Open remote sensing data
        self.ds = gdal.Open(fn, gdal.GA_ReadOnly)
        self.ncol = self.ds.RasterXSize
        self.nrow = self.ds.RasterYSize
        self.nband = self.ds.RasterCount

        # Determine data type used
        self.band = self.ds.GetRasterBand(1)
        self.dtype = self.band.ReadAsArray(0, 0, 1, 1).dtype

        # Initialize numpy array
        self.img = np.zeros((self.nband, self.nrow, self.ncol), dtype=self.dtype)

        # Read file onto memory
        for i in range(self.nband):
            self.band = self.ds.GetRasterBand(i+1)
            self.img[i, :, :] = self.band.ReadAsArray(0, 0, self.ncol, self.nrow)

        # Compute extent of the image
        self.geotransform = self.ds.GetGeoTransform()
        self.ext_up = self.geotransform[3]
        self.ext_left = self.geotransform[0]
        # Cell size
        self.x_spacing = self.geotransform[1]
        self.y_spacing = self.geotransform[5]
        # Continue computing extent
        self.ext_down = self.ext_up + self.y_spacing * self.nrow
        self.ext_right = self.ext_left + self.x_spacing * self.ncol


    def get_pixel(self, x, y, band=0):
        """
        Return the value of pixel.

        Default is to return the pixel value of first band at (x,y)

        (x,y) is based on the image coordinate where upper left is the origin, and x increase to the right, and y increases downwards.
        This is equivalent to (col, row) pair
        """
        # Check if given (x,y) is inside the range of the image
        if x < 0 or x > self.ncol - 1:
            print("X coordinate is out of range")
            return None

        if y < 0 or y > self.nrow - 1:
            print("Y coordinate is out of range")
            return None

        return self.img[band, y, x]
    
    def show_image(self, band=1, cmap=cm.gist_heat):
        """
        Display image of specified band.
        Default band to display is first band
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)

        cax = ax.imshow(self.img[band-1, :, :], cmap=cmap, interpolation='nearest')
        ax.axis('off')
        cbar = fig.colorbar(cax)

        plt.show()
