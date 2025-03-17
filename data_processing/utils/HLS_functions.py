import numpy as np

### Use this function after creating the feature arrays from Landsat files. 
def dictionary_from_coords(file_list, rows, cols):
    
    dic = {}
    for j, filename in enumerate(file_list):
        print(filename)
        file = np.load(filename)
        
        for i, (row, col) in enumerate(zip(rows, cols)):
            
            #if(i%250000==0):print(i)
            
            if((row,col) in dic):
                dic[(row,col)].append(j)
                dic[(row,col)].append([file[row,col,:].astype(int)])
            else:
                dic[(row,col)] = []
                dic[(row,col)].append(j)
                dic[(row,col)].append([file[row,col,:].astype(int)])
                
    print(f"Number of samples in the dictionary : {len(dic.keys())}")    
    
    return dic

### Use these functions after getting the dictionaries from the previous function. 

def unpack_Fmask(fmask):
    qa=0
    
    if fmask/2 % 2 == 0:
        qa = 1              ### Clear pixel
    
    elif fmask/2 % 2 == 1:
        q = fmask //  2**7
        
        if q % 2 == 0:
            qa = 0.5        ### Partially clouded
        else:
            qa = 0          ### Completely clouded
            
    return qa

def extracting_features_and_coordinates(dic, start_idx, end_idx, num_of_days, num_of_bands):
    
    total_features = []
    total_coords = []
    
    for j, key in enumerate(dic.keys()):
        
        if j%100000 == 0 : print(j)
        
        if(j<start_idx):continue
        if(j==end_idx):break
        
        feature_list = []
        for i, arr in enumerate(dic[key]):
            
            if(i%2 == 0) : continue  ## That is just the date index
            
            scaled_surface_reflectance = arr[0][:-1] / 10000   ## Scaling them to 0-1
            qa = unpack_Fmask(arr[0][-1]) 
            
            features_of_the_day = np.append(scaled_surface_reflectance, qa)
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


