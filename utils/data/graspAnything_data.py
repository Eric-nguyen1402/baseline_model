import glob
import os
import torch
from utils.dataset_processing import grasp, image
from .grasp_data import GraspDatasetBase
import numpy as np
import random
import pickle
from PIL import Image

class GraspAnythingDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Grasp-Anything dataset.
    """
    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: Grasp-Anything Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(GraspAnythingDataset, self).__init__(**kwargs)

        # Join all the file components together with any wildcard
        self.grasp_files = glob.glob(os.path.join(file_path, '*', '*_0.pt'))
        self.grasp_files.sort()
        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]
        
        # Change the wildcard name file from '.pkl' to '.jpg' in each file path (e.g., ab10cd.pkl to ab10cd.jpg)
        self.text_files = [f.replace('_0.pt', '.pkl') for f in self.grasp_files]
        self.rgb_files = [f.replace('.pkl', '.jpg') for f in self.text_files]

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_graspAnything_file(self.grasp_files[idx], scale=self.output_size / 1024.0)
        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))
        return gtbbs

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        # rgb_img = image.Image.from_file(self.rgb_files[idx])
        # rgb_img.rotate(rot)
        # rgb_img.zoom(zoom)
        # rgb_img.resize((self.output_size, self.output_size))
        # if normalise:
        #     rgb_img.normalise()
        #     rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        # return rgb_img.img
        rgb_img = Image.open(self.rgb_files[idx])
        return rgb_img

    def get_text_features(self, idx):
    # Load your text features from the '.pkl' file and preprocess them if needed
    # Example: Load text features using pickle
        try:
            with open(self.text_files[idx], 'rb') as f:
                text_features = pickle.load(f)
        except UnicodeDecodeError:
            # If UnicodeDecodeError occurs, try loading with a specific encoding (e.g., 'latin1')
            with open(self.text_files[idx], 'rb', encoding='latin1') as f:
                text_features = pickle.load(f)

        return text_features


    
