import os
import glob
import data_processing.utils.SEN1_functions as sen1
import rs
import numpy as np
import pickle

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

GEDI_sat = np.load('prelim_outputs/GEDI_sat.npy')
NLCD_flag = np.load('prelim_outputs/NLCD_flag.npy')

print('Sentinel 1 processing...')

sentinel1_input_dir = 'sentinel1_arrays/'
sentinel1_output_dir = 'prelim_outputs/sentinel1/'

band1_files = glob.glob(f'{sentinel1_input_dir}/band1/*.npy')
band2_files = glob.glob(f'{sentinel1_input_dir}/band2/*.npy')
band1_files.sort()
band2_files.sort()

print(f'Total number of Sentinel 1 files: band 1 - {len(band1_files)} and band 2 - {len(band2_files)}')

######################## CREATE TRAINING AND TEST DICTIONARIES ###################################

# TRAINING DICTIONARY

train_row_no = np.where((GEDI_sat != 0)*(NLCD_flag > 24))[0]
train_col_no = np.where((GEDI_sat != 0)*(NLCD_flag > 24))[1]

dic = sen1.dictionary_from_coords_sen1(band1_files, train_row_no, train_col_no)

dictionary_directory = sentinel1_output_dir + 'sentinel1_dictionaries/'
if not os.path.exists(dictionary_directory):
    os.makedirs(dictionary_directory)

pickle_file_path = dictionary_directory + 'sentinel1_training_points.pkl'
with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump(dic, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

del dic

print('Sentinel 1 training points dictionary saved as sentinel1_training_points.pkl')

# TEST DICTIONARY

test_row_no = np.where((NLCD_flag > 24))[0]
test_col_no = np.where((NLCD_flag > 24))[1]

chunk_size = 500000
rows_list = [test_row_no[i:i + chunk_size] for i in range(0, len(test_row_no), chunk_size)]
cols_list = [test_col_no[i:i + chunk_size] for i in range(0, len(test_col_no), chunk_size)]

for count, (row_num, col_num) in enumerate(zip(rows_list, cols_list)):
    print("The count is", count)
    dic = sen1.dictionary_from_coords_sen1(band1_files, row_num, col_num)   
    
    dictionary_directory = sentinel1_output_dir + 'sentinel1_dictionaries/'
    if not os.path.exists(dictionary_directory):
        os.makedirs(dictionary_directory)

    pickle_file_path = dictionary_directory + f'sentinel1_test_points_{count}.pkl'
    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(dic, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    del dic

print('Sentinel 1 test points dictionaries saved')

######################### EXTRACT FINAL SENTINEL 1 FEATURES DATASET ########################

num_of_days = 87
num_of_bands = 2        

# TRAINING DATASET

pickle_file_path = sentinel1_output_dir + 'sentinel1_dictionaries/sentinel1_training_points.pkl'
with open(pickle_file_path, 'rb') as file:
    sentinel1_train = pickle.load(file) 

start_idx = 0
end_idx = len(sentinel1_train.keys())   

train_features, train_coords, _ = sen1.extracting_features_and_coordinates_sen1(sentinel1_train, start_idx, end_idx, num_of_days, num_of_bands)

gedi_samples = sen1.pixel_values_from_raster_sen1(train_coords, GEDI_sat)
nlcd_samples = sen1.pixel_values_from_raster_sen1(train_coords, NLCD_flag)

data_dir = f'prelim_outputs/sentinel1/training_files/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    
np.save(data_dir + 'train_features', train_features)
np.save(data_dir + 'train_coords', train_coords)
np.save(data_dir + 'nlcd_train_samples', nlcd_samples)
np.save(data_dir + 'gedi_train_samples', gedi_samples)

# TEST DATASET

test_files = glob.glob(f'prelim_outputs/sentinel1/sentinel1_dictionaries/sentinel1_test_points_*')
test_files.sort()

start_arr = [0, 250000]
for i, test_file_path in enumerate(test_files): 
    
    with open(test_file_path, 'rb') as file:
        test_dic = pickle.load(file)
        
    print(f'Number of samples in the dictionary - {len(test_dic.keys())}')
    
    for start_idx in start_arr:
        end_idx = start_idx + 250000
        print(f"Starting index - {start_idx}")
        
        
        test_features, test_coords, end_idx = sen1.extracting_features_and_coordinates_sen1(test_dic, start_idx, end_idx, num_of_days, num_of_bands)
        print(f"Features extracted. Number of samples - {len(test_features)}")
        
        test_nlcd = sen1.pixel_values_from_raster_sen1(test_coords, NLCD_flag)
        print(f"Got the NLCD classes") 
        
        data_dir = f'prelim_outputs/sentinel1/test_files/'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        np.save(f"{data_dir}test_features_{start_idx + i*500000}_{end_idx + i*500000}", test_features)
        np.save(f"{data_dir}test_coords_{start_idx + i*500000}_{end_idx + i*500000}", test_coords)
        np.save(f"{data_dir}nlcd_test_samples_{start_idx + i*500000}_{end_idx + i*500000}", test_nlcd) 
        
    del test_dic

print('Sentinel 1 processing complete')
    






