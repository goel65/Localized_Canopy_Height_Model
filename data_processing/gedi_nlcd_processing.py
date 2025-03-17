# Once you have the numpy arrays of all the feature sensors (sentinel 1, sentinel 2 and landsat), 
# Start this script to make them ready for training/testing. The order is gedi_nlcd_processing -> landsat_processing -> sentinel2_processing -> sentinel1_processing
# Define the Location dictionary with the following keys:
# Xmin, Xmax, Ymin, Ymax, ncols, nrows, spatial_resolution

import os
import glob
import data_processing.utils.GEDI_functions as gedi
import data_processing.utils.NLCD_functions as nlcd
import rs
import numpy as np

Piute = {
    "Xmax" : -111.76628273950942,
    "Xmin" : -112.5197057381439,
    "Ymax" : 38.51244321480291,
    "Ymin" : 38.145371577379684,
    "spatial_resolution" :  0.00031021562683723573
}

Piute['ncols'] = int((Piute['Xmax'] - Piute['Xmin'])/ Piute['spatial_resolution'])
Piute['nrows'] = int((Piute['Ymax'] - Piute['Ymin'])/ Piute['spatial_resolution'])

Location = Piute

################################## GEDI processing #########################################

gedi_input_dir = 'GEDI_raw_files'
gedi_output_dir = 'prelim_outputs'
gedi_files = glob.glob(os.path.join(gedi_input_dir, '*.h5'))

print('Total number of GEDI files: ', len(gedi_files))
print('\n')

total_lat, total_long, total_rh95 = gedi.get_GEDI_samples(gedi_files, Location)
GEDI_sat = gedi.make_GEDI_raster(total_lat, total_long, total_rh95, Location)
np.save(gedi_output_dir + '/GEDI_sat.npy', np.array(GEDI_sat))
print('GEDI_sat.npy saved')


################################## NLCD processing #########################################

nlcd_input_dir = 'NLCD_raw_files'
nlcd_output_dir = 'prelim_outputs'

nlcd_fn = nlcd_input_dir + '/nlcd_2019_land_cover.tif'
nlcd_raster = rs.RSImage(nlcd_fn)

NLCD_flag = nlcd.make_NLCD_raster(nlcd_raster, Location)
np.save(nlcd_output_dir + '/NLCD_flag.npy', np.array(NLCD_flag))
print('NLCD_flag.npy saved')    


