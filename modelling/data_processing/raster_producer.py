import numpy as np
import glob
import os
from osgeo import gdal
from osgeo import osr

class RasterProducer:
    def __init__(self, location_config):
        """
        Initialize the raster producer with location configuration
        """
        self.location = location_config
        self.projection = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
    
    def compile_data_from_lists(self, files):
        """
        Compile data from multiple files into a single array
        """
        compiled_list = []
        for file in files:
            data = np.load(file)
            compiled_list.append(data)
        compiled_data = np.concatenate(compiled_list, axis=0)
        return compiled_data
    
    def produce_raster(self, coords, values, raster_name):
        """
        Produce a GeoTIFF raster from coordinates and values
        """
        # Initialize empty raster
        raster = np.zeros([self.location['nrows'], self.location['ncols']])
        
        # Fill raster with values at corresponding coordinates
        for coord, value in zip(coords, values):
            row = coord[0]
            col = coord[1]
            raster[row, col] = value
        
        # Get raster parameters from location config
        Xmin = self.location['Xmin']
        Ymax = self.location['Ymax']
        Spatial_Resolution = self.location['spatial_resolution']
        ncols = self.location['ncols']
        nrows = self.location['nrows']
        
        # Create GeoTIFF
        out_format = 'GTiff'
        driver = gdal.GetDriverByName(out_format)
        
        pred_output = driver.Create(raster_name + '.tif', ncols, nrows, 1, gdal.GDT_Float32)
        pred_output.SetGeoTransform([Xmin, Spatial_Resolution, 0, Ymax, 0, -Spatial_Resolution])
        pred_output.SetProjection(self.projection)
        pred_output.GetRasterBand(1).WriteArray(np.array(raster))
        pred_output = None
        
        print(f'Raster - {raster_name} is saved!!')
    
    def validate_batch_data(self, coord, nlcd, lan, sen, sen1):
        """
        Validate that all data arrays in a batch have the same length
        """
        lengths = {
            'coordinates': len(coord),
            'nlcd': len(nlcd),
            'landsat': len(lan),
            'sentinel2': len(sen),
            'sentinel1': len(sen1)
        }
        
        if not all(length == lengths['coordinates'] for length in lengths.values()):
            raise ValueError(f"Data arrays have different lengths: {lengths}")
    
    def process_predictions(self, model, results_dir, coord_files, nlcd_files, lan_features_files, 
                          sen_features_files, sen1_features_files, scaling_params):
        """
        Process predictions in batches and save results
        """
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        max_val, min_val = scaling_params
        
        for coord_file, nlcd_file, lan_file, sen_file, sen1_file in zip(
            coord_files, nlcd_files, lan_features_files, sen_features_files, sen1_features_files):
            
            # Load data
            coord = np.load(coord_file)
            nlcd = np.load(nlcd_file)
            lan = np.load(lan_file)
            sen = np.load(sen_file)
            sen1 = np.load(sen1_file)
            
            # Validate data lengths
            self.validate_batch_data(coord, nlcd, lan, sen, sen1)
            
            # Convert NLCD to one-hot encoding
            nlcd = np.vstack([
                nlcd == 31, nlcd == 41, nlcd == 42, nlcd == 43,
                nlcd == 52, nlcd == 71, nlcd == 81, nlcd == 82,
                nlcd == 90, nlcd == 95
            ]).T * 1
            
            # Generate predictions
            predictions = model.predict([lan, sen, sen1, nlcd]).flatten()
            predictions = predictions * (max_val - min_val) + min_val
            
            # Save predictions
            file_iden = '_'.join(coord_file.split('/')[-1].split('_')[2:])
            np.save(os.path.join(results_dir, 'predictions' + file_iden), predictions)
            print(f'samples {file_iden} have been processed') 