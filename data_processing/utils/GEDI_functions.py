import h5py
import numpy as np


def get_GEDI_samples(gedi_files, Location):
    total_lat = []
    total_long = []
    total_rh95 = []

    for file in gedi_files:

        data = h5py.File(file, 'r')
        print(file)

        beamNames = [b for b in data.keys() if b.startswith('BEAM')]

        for beam in beamNames:
            print(beam)

            flag = data[beam + '/quality_flag'][()]
            lat = data[beam + '/lat_highestreturn'][()]
            long = data[beam + '/lon_highestreturn'][()]
            rh = data[beam + '/rh']
            rh95 = rh[:,95]

            cond1 = lat > Location['Ymin']
            cond2 = lat < Location['Ymax']
            cond3 = long > Location['Xmin']
            cond4 = long < Location['Xmax']
            cond5 = (flag==1)
            mask = cond1*cond2*cond3*cond4*cond5

            total_lat.extend(lat[mask])
            total_long.extend(long[mask])
            total_rh95.extend(rh95[mask])
            
    return total_lat, total_long, total_rh95


def make_GEDI_raster(total_lat, total_long, total_rh95, Location):   
    GEDI_sat = [[0 for i in range(Location['ncols'])] for j in range(Location['nrows'])]
    repeat_count = 0
    repeat_pix_acc = []

    for i, (lat, long, rh95) in enumerate(zip(total_lat, total_long, total_rh95)):
        if(i%5000==0):print(i)

        xind = int((long - Location['Xmin'])/Location['spatial_resolution'])
        yind = int((Location['Ymax'] - lat)/Location['spatial_resolution'])

        if(xind<0 or xind>=Location['ncols']): continue
        if(yind<0 or yind>=Location['nrows']): continue

        if(GEDI_sat[yind][xind]!=0):
            repeat_count = repeat_count+1
            repeat_pix_acc.append([yind, xind, GEDI_sat[yind][xind], rh95])
            continue
        GEDI_sat[yind][xind] = rh95

    if repeat_count > 0:
        repeat_pix_acc = np.array(repeat_pix_acc)

        for i in range(repeat_count):
            yind = int(repeat_pix_acc[i,0])
            xind = int(repeat_pix_acc[i,1])
            if(len(np.where((repeat_pix_acc[:,0] == yind)*(repeat_pix_acc[:,1] == xind))) == 1):
                height = (repeat_pix_acc[i,2] + repeat_pix_acc[i,3])/2
                GEDI_sat[yind][xind] = height
            else:
                rep_indices = np.where((repeat_pix_acc[:,0] == yind)*(repeat_pix_acc[:,1] == xind))[0]
                if(i>rep_indices[0]) : continue
                height_sum = repeat_pix_acc[i,2]
                for index in rep_indices:
                    height_sum = height_sum + repeat_pix_acc[index, 3]
                height = height_sum/(len(rep_indices)+1)
                GEDI_sat[yind][xind] = height
        
    return GEDI_sat

