import torch
import torch.nn as nn
"""
Discriminator:

convolutional layer with in_channels=3, out_channels=128, kernel_size=4, stride=2
convolutional layer with in_channels=128, out_channels=256, kernel_size=4, stride=2
batch norm
convolutional layer with in_channels=256, out_channels=512, kernel_size=4, stride=2
batch norm
convolutional layer with in_channels=512, out_channels=1024, kernel_size=4, stride=2
batch norm
convolutional layer with in_channels=1024, out_channels=1, kernel_size=4, stride=1
Instead of Relu we LeakyReLu throughout the discriminator (we use a negative slope value of 0.2).

The output of your discriminator should be a single value score corresponding to each input sample. See torch.nn.LeakyReLU.
"""
dtype = torch.FloatTensor
#dtype = torch.cuda.FloatTensor

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        # Unflatten(batch_size, 1, 28, 28),
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, padding=1, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            #nn.MaxPool2d(kernel_size=4, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            #nn.MaxPool2d(kernel_size=4, stride=2),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            #nn.MaxPool2d(kernel_size=4, stride=2),

            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, padding=0, stride=1)
        )
        """
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=2)

        self.conv4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2)
        self.bn3 = nn.batchnorm2c(1024)
        self.relu3 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=4, stride=2)

        self.conv5 = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1)
        """

        # Flatten()
        #self.linear1 = nn.Linear(4*4*64, 4*4*64)
        #self.relu4 = nn.LeakyReLU(inplace=True, negative_slope=0.01)
        #self.linear2 = nn.Linear(4*4*64, 1)


        ##########       END      ##########

    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        # x = Unflatten(batch_size, 1, 28, 28)
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.bn1(x)
        # x = self.relu1(x)
        # x = self.maxpool1(x)
        #
        # x = self.conv3(x)
        # x = self.bn2(x)
        # x = self.relu2(x)
        # x = self.maxpool2(x)
        #
        # x = self.conv4(x)
        # x = self.bn3(x)
        # x = self.relu3(x)
        # x = self.maxpool3(x)
        #
        # x = self.conv5(x)

        #x = self.linear1(x)
        #x = self.relu4(x)
        #x = self.linear2(x)
        x = self.main(x)
        ##########       END      ##########

        return x
"""
Generator:

Note: In the generator, you will need to use transposed convolution (sometimes known as fractionally-strided convolution or deconvolution). This function is implemented in pytorch as torch.nn.ConvTranspose2d.

transpose convolution with in_channels=NOISE_DIM, out_channels=1024, kernel_size=4, stride=1
batch norm
transpose convolution with in_channels=1024, out_channels=512, kernel_size=4, stride=2
batch norm
transpose convolution with in_channels=512, out_channels=256, kernel_size=4, stride=2
batch norm
transpose convolution with in_channels=256, out_channels=128, kernel_size=4, stride=2
batch norm
transpose convolution with in_channels=128, out_channels=3, kernel_size=4, stride=2
"""

class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.noise_dim, out_channels=1024, kernel_size=4, padding=0, stride=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            #nn.MaxPool2d(kernel_size=4, stride=2),

            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            #nn.MaxPool2d(kernel_size=4, stride=2),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            #nn.MaxPool2d(kernel_size=4, stride=2),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, padding=1, stride=2),
            #nn.LeakyReLU(inplace=True, negative_slope=0.01)
        )

        ##########       END      ##########

    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        # x.view(-1, self.noise_dim, 1, 1)

        x = self.main(x)

        ##########       END      ##########

        return x
