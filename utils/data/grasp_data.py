import random
import clip
import numpy as np
import torch
import torch.utils.data
from torchvision import models
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import re

class ClipProcessor(nn.Module):
    def __init__(self):
        super(ClipProcessor, self).__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda")
        # Load pre-trained ResNet-101 model
        resnet18 = models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-2])  # Remove last two layers

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))
        
    def forward(self, text_sentences, related_words_list, image):
        input_image = self.numpy_to_torch(image).unsqueeze(0)
        x = self.resnet18(input_image)
        print(x.shape)

        # Encoding the sentence part of the text
    
        # Assuming text_sentences is a list of strings
        list_of_words = []
        for sentence in text_sentences:
            words = sentence.split()
            list_of_words.extend(words)  # Extending the list with words from each sentence
        tokenized_sentence = clip.tokenize(list_of_words).to("cuda")
        sentence_features = self.clip_model.encode_text(tokenized_sentence)

        # Encoding the related words
        tokenized_related_words = clip.tokenize(related_words_list[0]).to("cuda")
        related_words_features = self.clip_model.encode_text(tokenized_related_words)

        # Combining features if needed for further processing
        combined_features = torch.cat((sentence_features, related_words_features), dim=0)

        print(combined_features.shape)
        # token_text = clip.tokenize([text]).to("cuda")
        # text_features = self.clip_model.encode_text(token_text)
        # print(text_features.shape)

        # x = torch.cat((x, text_features.unsqueeze(0)), dim=1)
        # print(x.shape)

        return x

class GraspDatasetBase(torch.utils.data.Dataset):
    """
    An abstract dataset for training networks in a common format.
    """

    def __init__(self, output_size=224, include_depth=True, include_rgb=False,include_text=True, random_rotate=False,
                 random_zoom=False, input_only=False, seen=True):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
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

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()
    
    def get_text(self, idx):
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

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)


        # Load the text
        if self.include_text:
            text = self.get_text(idx)
            text_sentences, related_words_list = self.split_text_tuples(text)
            

        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor)

        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        width_img = np.clip(width_img, 0.0, self.output_size / 2) / (self.output_size / 2)

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
        elif self.include_rgb and self.include_text:
            x = self.clip_processor(text_sentences, related_words_list, rgb_img)
           

        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2 * ang_img))
        sin = self.numpy_to_torch(np.sin(2 * ang_img))
        width = self.numpy_to_torch(width_img)

        return x, (pos, cos, sin, width), idx, rot, zoom_factor

    def __len__(self):
        return len(self.grasp_files)
    
    def split_text_tuples(self, sentences):
        text_sentences = []
        related_words_list = []
        for i, data in enumerate(sentences):
            if i % 2 == 0:
                text_sentences.append(data)
            else:
                related_words_list.append(data)

        return text_sentences, related_words_list

