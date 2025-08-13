from torch import nn
from . import layers as ly
import torch


class Parameters:
    def __init__(self, param_dict):
        self.CNN_w_regularizer = param_dict["CNN_w_regularizer"]
        self.RNN_w_regularizer = param_dict["RNN_w_regularizer"]
        self.CNN_batch_size = param_dict["CNN_batch_size"]
        self.RNN_batch_size = param_dict["RNN_batch_size"]
        self.CNN_drop_rate = param_dict["CNN_drop_rate"]
        self.RNN_drop_rate = param_dict["RNN_drop_rate"]
        self.epochs = param_dict["epochs"]
        self.gpu = param_dict["gpu"]
        self.model_filepath = param_dict["model_filepath"] + "/net.h5"
        self.num_clinical = param_dict["num_clinical"]
        self.image_shape = param_dict["image_shape"]
        self.final_layer_size = param_dict["final_layer_size"]
        self.optimizer = param_dict["optimizer"]


class CNN(nn.Module):
    def __init__(self, image_channels, clin_data_channels, droprate):
        super().__init__()

        # Image Section
        self.image_section = CNN_Image_Section(image_channels, droprate)

        # Data Layers, fully connected
        self.fc_clin1 = ly.FullConnBlock(clin_data_channels, 64, droprate=droprate)
        self.fc_clin2 = ly.FullConnBlock(64, 20, droprate=droprate)

        # Final Dense Layer
        self.dense1 = nn.Linear(40, 5)
        self.dense2 = nn.Linear(5, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        image, clin_data = x

        image = self.image_section(image)

        clin_data = self.fc_clin1(clin_data)
        clin_data = self.fc_clin2(clin_data)

        x = torch.cat((image, clin_data), dim=1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x


class CNN_Image_Section(nn.Module):
    def __init__(self, image_channels, droprate):
        super().__init__()
        # Initial Convolutional Blocks
        self.conv1 = ly.ConvBlock(
            image_channels,
            192,
            (11, 13, 11),
            stride=(4, 4, 4),
            droprate=droprate,
            pool=False,
        )
        self.conv2 = ly.ConvBlock(192, 384, (5, 6, 5), droprate=droprate, pool=False)

        # Midflow Block
        self.midflow = ly.MidFlowBlock(384, droprate)

        # Split Convolutional Block
        self.splitconv = ly.SplitConvBlock(384, 192, 96, 1, droprate)

        # Fully Connected Block
        self.fc_image = ly.FullConnBlock(227136, 20, droprate=droprate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.midflow(x)
        x = self.splitconv(x)
        x = torch.flatten(x, 1)
        x = self.fc_image(x)

        return x
