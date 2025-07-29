#%% Imports and function definitions
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
import seaborn as sns 

from natsort import natsorted
from time import time
from cellpose import models
from tkinter.filedialog import askdirectory
from skimage.filters import gaussian
from skimage.measure import regionprops_table
def scramble(mask):
    idx = np.unique(mask[mask>0])
    idx_new = idx.copy()
    random.shuffle(idx)
    mask_new = np.zeros_like(mask)
    for n,i in enumerate(idx):
        mask_new[mask==idx[n]] = idx_new[n]
    return mask_new

#%% Load Cellpose model and file selection
model = models.Cellpose(model_type='cyto3',gpu = True)
DirMethod = askdirectory(title='Select Method Directory')
DirImages = [DirMethod+'/Images/'+f for f in os.listdir(DirMethod+'/Images') if os.path.isdir(DirMethod+'/Images/'+f)]

# Columns specific for this type of analysis (time lapse, fluoresecence + brightfield)
Columns_filename = ['Acquisition Name','Well','Time Point #','ID Acquisition','ID Acquisition - Mode','Image Code and extension']

df = pd.DataFrame(columns=Columns_filename, index=[0])
filenames = []
idx = 0
for fold in DirImages:
    for file in os.listdir(fold):
        if file.endswith('.tif') or file.endswith('.tiff'):
            parts = file.split('_')
            if len(parts) == len(Columns_filename):
                df.loc[idx] = parts
                filenames.append(os.path.join(fold, file))
                idx += 1
df['Filename'] = filenames
'''
Not phase images only raw BF images
Diameter 20 pixel good for MB231, test on other cell lines
Add multichannel analysis and DataFrame export
'''

#%% Analysis parameters 
Cellpose_Diameter   = 20              # Diameter for cells (in pixel, is scaled with bin_step)
BKG_sigma_gauss     = Cellpose_Diameter

Properties = [  'label', 
                'area', 
                'centroid', 
                'eccentricity',
                'solidity',
                'equivalent_diameter_area'
              ]
# %%
FLAG_DF = True
FLAG_TIME = True
for acq in df['Acquisition Name'].unique():
    df_acq = df[df['Acquisition Name'] == acq]
    for well in natsorted(df_acq['Well'].unique()):
        df_acq_well = df_acq[df_acq['Well'] == well]
        for time_point in natsorted(df_acq_well['Time Point #'].unique()):
            t0 = time()
            df_tmp = df_acq_well[df_acq_well['Time Point #'] == time_point]
            print(f'Processing {acq} - {well} - Time Point {time_point}...', end=' ')
            img_BF = tifffile.imread(df_tmp[df_tmp['ID Acquisition - Mode']=='Bf']['Filename'].iloc[0])  # Read the first file as a reference for BF
            masks, flows, styles, diams = model.eval(img_BF, diameter=Cellpose_Diameter, channels=[0,0])
            Cell_Number = np.max(masks)
            Confluency = np.sum(masks>0)/masks.size
            print(f'Found {Cell_Number} cells with confluency {Confluency:.2f}')
            params = regionprops_table(masks, properties=Properties)
            
            Acquisition_Mode_list = [mode for mode in df_tmp['ID Acquisition - Mode'].unique() if mode not in ['Ph', 'Bf']]
            for Acquisition_Mode in Acquisition_Mode_list:
                img_mod = tifffile.imread(df_tmp[df_tmp['ID Acquisition - Mode'] == Acquisition_Mode]['Filename'].iloc[0])
                img_mod = img_mod - gaussian(img_mod, sigma=BKG_sigma_gauss,preserve_range=True)  # Apply Gaussian filter to the image
                img_mod[img_mod < 0] = 0  # Ensure no negative values after filtering
                params['Int - Mean'] = []
                params['Int - Median'] = []
                params['Int - Max'] = []
                for cell in range(1, Cell_Number + 1):
                    mask_cell = (masks == cell)
                    int_val = img_mod[mask_cell]
                    params['Int - Mean'].append(np.mean(int_val))
                    params['Int - Median'].append(np.median(int_val))
                    params['Int - Max'].append(np.max(int_val))
            params_df = pd.DataFrame(params)
            params_df['Acquisition Name'] = acq
            params_df['Well'] = well
            params_df['Time Point #'] = time_point
            if FLAG_TIME:
                t1 = time()
                print(f'Time taken: {t1-t0:.2f} seconds')
                t_estimated_time = (t1-t0) * (len(df_acq_well['Time Point #'].unique()))
                t_est_total = t_estimated_time * (len(df_acq['Well'].unique())) * (len(df['Acquisition Name'].unique()))
                print(f'Estimated time for this acquisition: {t_estimated_time/60:.2f} minutes')
                FLAG_TIME = False            
            
            if FLAG_DF:
                df_final = params_df.copy()
                FLAG_DF = False
            else:
                df_final = pd.concat([df_final, params_df], ignore_index=True)    

# %%
Params_Dict = {
    'Diameter': Cellpose_Diameter,
    'BKG_sigma_gauss': BKG_sigma_gauss,
    'Properties': Properties,
    'Columns_filename': Columns_filename,
    'Directory': DirMethod,
}
df_final.to_csv(os.path.join(DirMethod, 'Cellpose_Segmentation_Results.csv'), index=False)
np.save(os.path.join(DirMethod, 'Cellpose_Segmentation_Params.npy'), Params_Dict)
   
# %%
time_points = natsorted(df_final['Time Point #'].unique())
for name in df_final['Acquisition Name'].unique():
    fig, ax = plt.subplots(1,2,figsize=(10, 6))
    df1 = df_final[df_final['Acquisition Name'] == name]
    wells = natsorted(df1['Well'].unique())
    y_all = np.zeros((len(time_points),len(wells)))
    for cnt_wells,well in enumerate(wells):
        mean_int = []
        for time_point in time_points:
            df_tmp = df1[(df1['Well'] == well) & (df1['Time Point #'] == time_point)]
            mean_int.append(df_tmp[df_tmp['Time Point #'] == time_point]['Int - Mean'].median())
            x = [int(tp) for tp in time_points]
        y = mean_int
        y_all[:,cnt_wells] = y
        ax[0].plot(x, y, '-o', label = well)
        ax[0].set_xlabel('Time Point')
        ax[0].set_ylabel('Mean Intensity')
    ax[1].errorbar(x, np.mean(y_all, axis=1), yerr=np.std(y_all, axis=1), fmt='o', color='black', label='Mean ± Std Dev')
    lin_fit = np.polyfit(x, np.mean(y_all, axis=1), 1)
    drop_percent = np.round((1-np.polyval(lin_fit, x[-1])/np.polyval(lin_fit, x[0]))*100,2)
    ax[1].plot(x, np.polyval(lin_fit, x), '--r', label = well + '- Fit')
    ax[1].set_xlabel('Time Point')
    ax[1].set_ylabel('Mean Intensity')
    ax[0].set_title('Single wells')
    ax[1].set_title('Fit results: y = {:.2f}x + {:.2f}'.format(lin_fit[0], lin_fit[1]))
    ax[1].set_ylim(0, np.max(np.mean(y_all, axis=1)) * 1.1)
    ax[0].legend()
    ax[1].legend()
    plt.suptitle(f'Mean Intensity for {name} - Drop: {drop_percent}%')
    plt.tight_layout()
    plt.savefig(os.path.join(DirMethod, f'{name}_Mean_Intensity.png'), dpi=300)
    plt.show()
    
# %%
