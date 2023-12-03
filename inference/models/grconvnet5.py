import torch
import torch.nn as nn
import torch.nn.functional as F
from inference.models.grasp_model import GraspModel, ResidualBlock
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image


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
        image = self.image_transform(image).unsqueeze(0).to("cuda")
        image_features = self.clip_model.encode_image(image)
        return image_features

    def extract_text_features(self, text):
        text = clip.tokenize([text]).to("cuda")
        text_features = self.clip_model.encode_text(text)
        return text_features


class GenerativeResnet(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super(GenerativeResnet, self).__init__()
        # self.clip_processor = ClipProcessor()

        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)

        self.res1 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res2 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res3 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res4 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res5 = ResidualBlock(channel_size * 4, channel_size * 4)

        self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 2)

        self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, combined_features):
        # # Extract features using CLIP processor
        # image_features, text_features = self.clip_processor(x_text, x_image)
        # # Combine image and text features here
        # combined_features = torch.cat((image_features, text_features), dim=1)

        # Image processing
        x_image = F.relu(self.bn1(self.conv1(combined_features)))
        x_image = F.relu(self.bn2(self.conv2(x_image)))
        x_image = F.relu(self.bn3(self.conv3(x_image)))
        x_image = self.res1(x_image)
        x_image = self.res2(x_image)
        x_image = self.res3(x_image)
        x_image = self.res4(x_image)
        x_image = self.res5(x_image)
        x_image = F.relu(self.bn4(self.conv4(x_image)))
        x_image = F.relu(self.bn5(self.conv5(x_image)))
        x_image = self.conv6(x_image)

        # Final convolutional layer
        x = F.relu(self.conv6(x))

        # Output layers
        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output
