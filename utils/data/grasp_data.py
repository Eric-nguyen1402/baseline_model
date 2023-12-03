import random
import numpy as np
import torch
import torch.utils.data
from utils.dataset_processing import grasp, image
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import torch.nn as nn

class ClipProcessor(nn.Module):
    def __init__(self):
        super(ClipProcessor, self).__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda")
        self.image_transform = Compose([Resize((224, 224)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def forward(self, text, image):
        image_features = self.extract_image_features(image)
        text_features = self.extract_text_features(text)
        return image_features, text_features

    def extract_image_features(self, image):
        # image = (image * 255).astype(np.uint8)
        # image = Image.fromarray(image)

        image = self.image_transform(image).unsqueeze(0).to("cuda")
        image_features = self.clip_model.encode_image(image)
        return image_features

    def extract_text_features(self, text):
        text = clip.tokenize([text]).to("cuda")
        text_features = self.clip_model.encode_text(text)
        return text_features
    
class GraspDatasetBase(torch.utils.data.Dataset):
    """
    An abstract dataset for training networks in a common format.
    """

    def __init__(self, output_size=224, include_depth=False, include_rgb=False, include_text=True,
                 random_rotate=False, random_zoom=False, input_only=False):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param include_text: Whether text features are included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.include_text = include_text
        self.grasp_files = []
        self.clip_processor = ClipProcessor()

        if include_depth is False and include_rgb is False and include_text is False:
            raise ValueError('At least one of Depth, RGB, or Text must be specified.')

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_text_features(self, idx):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if self.random_rotate:
            rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor)
        
        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        
        width_img = np.clip(width_img, 0.0, self.output_size / 2) / (self.output_size / 2)

        x = []

        # if self.include_rgb:
        #     x.append(self.numpy_to_torch(rgb_img))

        # Load text features
        if self.include_text and self.include_rgb:
            text = self.get_text_features(idx)
            rgb_img = self.get_rgb(idx, rot, zoom_factor)
            image_features, text_features = self.clip_processor(text, rgb_img)

        x = torch.cat((image_features, text_features), dim=1)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2 * ang_img))
        sin = self.numpy_to_torch(np.sin(2 * ang_img))
        width = self.numpy_to_torch(width_img)

        return x, (pos, cos, sin, width), idx, rot, zoom_factor

    def __len__(self):
        return len(self.grasp_files)
