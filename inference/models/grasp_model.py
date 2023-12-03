import torch.nn as nn
import torch.nn.functional as F


class GraspModel(nn.Module):
    """
    An abstract model for grasp network in a common format.
    """

    def __init__(self):
        super(GraspModel, self).__init__()

    def forward(self, x_in):
        raise NotImplementedError()

    def compute_loss(self, image, text, yc):
        y_pos, y_height, y_width, y_theta = yc
        pos_pred, height_pred, width_pred, theta_pred = self(image, text)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        h_loss = F.smooth_l1_loss(height_pred, y_height)
        w_loss = F.smooth_l1_loss(width_pred, y_width)
        theta_loss = F.smooth_l1_loss(theta_pred, y_theta)

        return {
            'loss': p_loss + h_loss + w_loss + theta_loss,
            'losses': {
                'p_loss': p_loss,
                'h_loss': h_loss,
                'w_loss': w_loss,
                'theta_loss': theta_loss
            },
            'pred': {
                'pos': pos_pred,
                'height': height_pred,
                'width': width_pred,
                'theta': theta_pred
            }
        }

    def predict(self, xc):
        pos_pred, height_pred, width_pred, theta_pred = self(xc)
        return {
            'pos': pos_pred,
            'height': height_pred,
            'width': width_pred,
            'theta': theta_pred
        }

class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in
