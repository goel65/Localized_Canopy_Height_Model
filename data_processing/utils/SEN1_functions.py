import numpy as np

################################## Dictionary creation ########################################
def dictionary_from_coords_sen1(band1_files, band2_files, rows, cols):
    
    dic = {}
    
    for j, (b1_file, b2_file) in enumerate(zip(band1_files, band2_files)):
        
        iden1 = b1_file.split('/')[-1].split('_')[1]
        iden2 = b2_file.split('/')[-1].split('_')[1]
        assert(iden1==iden2)
        
        print(f"Processing data of file - {iden1}")
        
        b1 = np.load(b1_file)
        b2 = np.load(b2_file)
        
        for i, (row, col) in enumerate(zip(rows, cols)):

            values = [b1[row,col], b2[row,col]]

            if((row,col) in dic):
                dic[(row,col)].append(j)
                dic[(row,col)].append(values)
            else:
                dic[(row,col)] = []
                dic[(row,col)].append(j)
                dic[(row,col)].append(values)
    
    print(f"Number of samples in the dictionary : {len(dic.keys())}")    
    
    return dic

def extracting_features_and_coordinates_sen1(dic, start_idx, end_idx, num_of_days, num_of_bands):
    
    total_features = []
    total_coords = []
    
    for j, key in enumerate(dic.keys()):
        
        if j%100000 == 0 : print(j)
        
        if(j<start_idx):continue
        if(j==end_idx):break
        
        feature_list = []
        for i, arr in enumerate(dic[key]):
            
            if(i%2 == 0) : continue  ## That is just the date index
            
            scaled_surface_reflectance = arr
            
            features_of_the_day = scaled_surface_reflectance
            feature_list.extend(features_of_the_day)
        
        total_features.append(feature_list)
        total_coords.append(key)
        
    end_idx = j
        
    total_features = np.array(total_features).reshape(len(total_features), num_of_days, num_of_bands)
    total_coords = np.array(total_coords)
    
    return total_features, total_coords, end_idx


def pixel_values_from_raster(coords, raster):
    samples = []
    for coord in coords:
        row = coord[0]
        col = coord[1]
        samples.append(raster[row,col])
        
    samples = np.array(samples)
    return samples