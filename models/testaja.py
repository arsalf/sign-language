import h5py
import glob
import os

# for h5name in glob.glob(os.path.join('models')+'/*'):
#     print(h5name)

# with h5py.File('table_links.h5',mode='w') as h5fw:
#     link_cnt = 0 
#     for h5name in glob.glob(os.path.join('models')+'/*'):        
#         link_cnt += 1
#         h5fw['link'+str(link_cnt)] = h5py.ExternalLink(h5name,'/') 

with h5py.File('table_merge.h5',mode='w') as h5fw:
    row1 = 0
    for h5name in glob.glob('*.h5'):
        h5fr = h5py.File(h5name,'r') 
        dset1 = list(h5fr.keys())[0]
        arr_data = h5fr[dset1][:]
        h5fw.require_dataset('alldata', dtype="f",  shape=(50,5), maxshape=(100, 5) )
        h5fw['alldata'][row1:row1+arr_data.shape[0],:] = arr_data[:]
        row1 += arr_data.shape[0]