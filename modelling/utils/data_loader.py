import numpy as np
import glob
import pickle
import os
import pandas as pd

class SatelliteDataLoader:
    def __init__(self, base_dir, location='piute'):
        """
        Initialize the data loader with base directory and location
        """
        self.base_dir = base_dir
        self.location = location
        self.scaling_parameters = {'max': 50, 'min': 0}
        
    def check_identical_2d(self, arr1, arr2):
        """
        Check if two 2D arrays are identical
        """
        if arr1.shape != arr2.shape:
            return False
        if (arr1 == arr2).sum() / arr1.shape[0] == arr1.shape[1]:
            return True
        return False

    def validate_coordinates(self, coords_dict):
        """
        Validate that coordinates from different sensors match
        """
        # Check if all coordinate arrays have the same shape
        shapes = [arr.shape for arr in coords_dict.values()]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError("Coordinate arrays have different shapes")
            
        # Check if coordinates are identical
        coords_list = list(coords_dict.values())
        for i in range(len(coords_list)):
            for j in range(i + 1, len(coords_list)):
                if not self.check_identical_2d(coords_list[i], coords_list[j]):
                    raise ValueError(f"Coordinates from sensors {i} and {j} are not identical")
        
        return True
        
    def load_training_data(self):
        """
        Load training data for all satellites and NLCD
        """
        # Load coordinates
        lan_train_coords = np.load(f'{self.base_dir}/landsat/training_files/train_coords.npy')
        sen_train_coords = np.load(f'{self.base_dir}/sentinel2/training_files/train_coords.npy')
        sen1_train_coords = np.load(f'{self.base_dir}/sentinel1/training_files/train_coords.npy')
        
        # Validate coordinates
        coords_dict = {
            'landsat': lan_train_coords,
            'sentinel2': sen_train_coords,
            'sentinel1': sen1_train_coords
        }
        self.validate_coordinates(coords_dict)
        
        # Load features
        lan_train_features = np.load(f'{self.base_dir}/landsat/training_files/train_features.npy')
        sen_train_features = np.load(f'{self.base_dir}/sentinel2/training_files/train_features.npy')
        sen1_train_features = np.load(f'{self.base_dir}/sentinel1/training_files/train_features.npy')
        nlcd_train = np.load(f'{self.base_dir}/sentinel2/training_files/nlcd_train_samples.npy')
        
        # Load labels
        gedi_train = np.load(f'{self.base_dir}/sentinel2/training_files/gedi_train_samples.npy')

        
        # Convert NLCD to one-hot encoding
        nlcd_train = pd.get_dummies(nlcd_train)
        
        return {
            'landsat_features': lan_train_features,
            'sentinel2_features': sen_train_features,
            'sentinel1_features': sen1_train_features,
            'nlcd': nlcd_train,
            'gedi': gedi_train,
            'coordinates': sen_train_coords  # Using sentinel2 coords as reference
        }
    
    def normalize_gedi(self, gedi_data):
        """
        Normalize GEDI data using min-max scaling
        """
        return (gedi_data - self.scaling_parameters['min']) / (self.scaling_parameters['max'] - self.scaling_parameters['min'])
    
    def denormalize_gedi(self, normalized_data):
        """
        Convert normalized predictions back to original scale
        """
        return normalized_data * (self.scaling_parameters['max'] - self.scaling_parameters['min']) + self.scaling_parameters['min']
    
    def _convert_nlcd_to_onehot(self, nlcd_data):
        """
        Convert NLCD classes to one-hot encoding
        """
        classes = [31, 41, 42, 43, 52, 71, 81, 82, 90, 95]
        one_hot = np.zeros((len(nlcd_data), len(classes)))
        for i, cls in enumerate(classes):
            one_hot[:, i] = (nlcd_data == cls).astype(int)
        return one_hot
    
