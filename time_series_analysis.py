# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:52:47 2024

@author: farrara
"""
import torch
import re
import pandas as pd
import matplotlib.pyplot as plt
from natsort import natsorted

MODEL_PATH = r'Z:\Alison2\Time Series Videos\Model\AntibioticClassification_TimeLapse_Untreated_240722_1640\AMRClassification_[Ciprofloxacin-Cy3]_231118_1123'

def get_stats(PATH):

    model=torch.load(MODEL_PATH)

    model_stats = []

    for i in range(len(model['pred_labels'])):
        cell = {}
        cell['pred_label'] = model['pred_labels'][i]
        file_name = model['test_stats'][i]['File Name']
        time = re.search(r'_t(\d+)_', file_name)
        cell['time'] = (int(time.group(1))*5)+5
        cell['area'] = model['test_stats'][i]['Cell Area (Pixels)']
        cell['mask'] = model['test_stats'][i]['Mask ID'] # can be used with cv2 to plot overlay
        model_stats.append(cell)
    
    model_stats = pd.DataFrame(model_stats)
    
    return model_stats

df = get_stats(MODEL_PATH)

# Sort the DataFrame naturally by 'time'
df['time'] = pd.Categorical(df['time'], categories=natsorted(df['time'].unique()), ordered=True)
df = df.sort_values('time')

# Calculate mean pred_label over time
mean_pred_label_over_time = df.groupby('time')['pred_label'].mean()
mean_cell_area_over_time = df.groupby('time')['area'].mean()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(mean_pred_label_over_time.index, mean_pred_label_over_time.values, marker='o', linestyle='-')
plt.title('Mean pred_label over time')
plt.xlabel('Time (min)')
plt.ylabel('Mean predicted label, 0 = untreated, 1 = ciprofloxacin')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(mean_cell_area_over_time.index, mean_cell_area_over_time.values, marker='o', linestyle='-')
plt.title('Mean area over time')
plt.xlabel('Time (min)')
plt.ylabel('Cell Area (pixels)')
plt.grid(True)
plt.show()
