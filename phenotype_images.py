"""
Adapted to test only, manual 2 classes
"""

import torch
import timm
import numpy as np
from trainer import Trainer
from file_io import get_metadata, get_cell_images, cache_data, get_training_data, convert_antibiotics_to_strain
import pickle
import re

image_size = (64,64)
resize = False

antibiotic_list = ["Untreated","Ciprofloxacin"]
#antibiotic_list = ["Untreated"]
microscope_list = ["KAP-NIM"]
channel_list = ["mKate"]
cell_list = ["single"]
train_metadata = {"content": "E.Coli MG1655"}
test_metadata = {"content": "E.Coli MG1655",
                 "user_meta3": "BioRepA",
                 "user_meta5": "Experiment5min",
                 "user_meta6": "TimeSeries"}

AKSEG_DIRECTORY = r"\\cmdaq6.nat.physics.ox.ac.uk\AKGroup\Piers_2\BacSeg Database"

USER_INITIAL = "AF"

SAVE_DIR = r"C:\Users\farrara\Desktop\AMR_Pytorch_Classification"
MODEL_FOLDER_NAME = "AntibioticClassification_TimeLapse_CipB_2"

MODEL_PATH = r"C:\Users\farrara\Desktop\AMR_Pytorch_Classification\AMRClassification_[Ciprofloxacin-Cy3]_231118_1123"

akseg_metadata = get_metadata(AKSEG_DIRECTORY,
                              USER_INITIAL,
                              channel_list,
                              antibiotic_list,
                              microscope_list,
                              train_metadata,
                              test_metadata,)


#
if __name__ == '__main__':

    cached_data = cache_data(
        akseg_metadata,
        image_size,
        antibiotic_list,
        channel_list,
        cell_list,
        import_limit = 'None',
        mask_background=True,
        resize=resize)

    times = []
    for file_name in cached_data['file_names']:
        time = re.search(r'_t(\d+)_', file_name)
        time = (int(time.group(1)) * 5) + 5
        times.append(time)
        print(time)
        ## test this

    #images = cached_data['images']
    labels = cached_data['labels'] #e.g. cip, untreated

    time_labels = []

    for i in range(len(labels)):
        time=str(times[i])
        label=str(labels[i])
        time_label=label+'_'+time
        time_labels.append(time_label)

        #time_label =
    #file_names = cached_data['file_names']
    sliced_images = [array[0] for array in cached_data['images']]

    # put sliced_images and time_labels into UMAP