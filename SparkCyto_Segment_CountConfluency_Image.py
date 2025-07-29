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
from tkinter.filedialog import askopenfilename
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
filename = askopenfilename(title='Select Image File',filetypes=(("Image files", "*.tif;*.tiff;*.czi"), ("All files", "*.*")))

# Columns specific for this type of analysis (time lapse, fluoresecence + brightfield)
Columns_filename = ['Acquisition Name','Well','ID Acquisition','ID Acquisition - Mode','Image Code and extension']
df = pd.DataFrame(columns=Columns_filename, index=[0])
parts = os.path.basename(filename).split('_')
df.loc[0] = parts
df['Filename'] = filename
Dir = os.path.dirname(filename)
filename_noext = os.path.splitext(os.path.basename(filename))[0]

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
if df['ID Acquisition - Mode']=='Bf':
    img_BF = tifffile.imread(df['Filename'].iloc[0])  # Read the first file as a reference for BF
else:
    print("No brightfield image found in the selected file.")
masks, flows, styles, diams = model.eval(img_BF, diameter=Cellpose_Diameter, channels=[0,0])
Cell_Number = np.max(masks)
Confluency = np.sum(masks>0)/masks.size
df['Cell Number'] = Cell_Number
df['Confluency'] = Confluency
print(f'Found {Cell_Number} cells with confluency {Confluency:.2f}')
params = regionprops_table(masks, properties=Properties)
params_df = pd.DataFrame(params)
df['Area - Median'] = params_df['area'].median()
df['Area - Mean'] = params_df['area'].mean()
df['Eccentricity - Median'] = params_df['eccentricity'].median()
df['Eccentricity - Mean'] = params_df['eccentricity'].mean()
df['Solidity - Median'] = params_df['solidity'].median()
df['Solidity - Mean'] = params_df['solidity'].mean()
df['Equivalent Diameter - Median'] = params_df['equivalent_diameter_area'].median()
df['Equivalent Diameter - Mean'] = params_df['equivalent_diameter_area'].mean()

# %%
Params_Dict = {
    'Diameter': Cellpose_Diameter,
    'BKG_sigma_gauss': BKG_sigma_gauss,
    'Properties': Properties,
    'Columns_filename': Columns_filename,
    'Filename': filename,
}
df.to_csv(os.path.join(Dir, filename_noext+'_Cellpose_Segmentation_Results.csv'), index=False)
np.save(os.path.join(Dir, filename_noext+'_Cellpose_Segmentation_Params.npy'), Params_Dict)
   
# %%
fig, ax = plt.subplots(1,2,figsize=(10, 6))

    
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
sns.kdeplot(data = params_df, x = 'equivalent_diameter_area', y = 'eccentricity',ax=ax, fill=True,)
# %%
