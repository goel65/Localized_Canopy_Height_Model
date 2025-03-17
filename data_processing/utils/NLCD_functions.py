def make_NLCD_raster(nlcd, Location):
    NLCD_flag = []
    NLCD_flag = [[0 for i in range(Location['ncols'])] for j in range(Location['nrows'])]

    Xmin_g = nlcd.ext_left
    Ymax_g = nlcd.ext_up
    Xmax_g = nlcd.ext_right
    Ymin_g = nlcd.ext_down
    SRg = nlcd.ds.GetGeoTransform()[1]  

    for i in range(Location['nrows']):
        if(i%50==0):print(i)
        for j in range(Location['ncols']): 
            
            Xcord = Location['Xmin'] + (j+0.5)*Location['spatial_resolution']
            Ycord = Location['Ymax'] - (i+0.5)*Location['spatial_resolution']
            
            if(Xcord - Xmin_g < 0 ): continue
            if(Xcord - Xmax_g > 0 ): continue
            if(Ycord - Ymax_g > 0 ): continue
            if(Ycord - Ymin_g < 0 ): continue
            
            x_pix = int((Xcord - Xmin_g)/SRg)
            y_pix = int((Ymax_g - Ycord)/SRg)
            
            height = nlcd.get_pixel(x_pix,y_pix)
            NLCD_flag[i][j] = height

    return NLCD_flag






