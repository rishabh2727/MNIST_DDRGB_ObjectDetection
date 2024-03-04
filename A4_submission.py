import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

####  IMAGE CLASSIFICATION MODEL  ####
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()
        out_channels = int(in_channels/2)

        self.layer1 = nn.Sequential(
                      nn.Conv2d(in_channels, out_channels, kernel_size=1,stride = 1, padding=0, bias=False),
                      nn.BatchNorm2d(out_channels),
                      nn.LeakyReLU())
        
        self.layer2 = nn.Sequential(
                      nn.Conv2d(out_channels, in_channels, kernel_size=3,stride = 1, padding=1, bias=False),
                      nn.BatchNorm2d(in_channels),
                      nn.LeakyReLU())
    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out

class MNISTDDRGBDarknet53(nn.Module):
    def __init__(self, block, num_classes):
        super(MNISTDDRGBDarknet53, self).__init__()
        self.num_classes = num_classes
        
        self.conv1 = nn.Sequential(
                      nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
                      nn.BatchNorm2d(32),
                      nn.LeakyReLU())
       
        self.conv2 = nn.Sequential(
                      nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(64),
                      nn.LeakyReLU())
                      
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)    ## Residual block after every two conv layers

        self.conv3 = nn.Sequential(
                      nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(128),
                      nn.LeakyReLU())
        
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)     ## Residual block after every two conv layers

        self.conv4 = nn.Sequential(
                      nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(256),
                      nn.LeakyReLU())
        
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)

        self.conv5 = nn.Sequential(
                      nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(512),
                      nn.LeakyReLU())
       
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        
        self.conv6 = nn.Sequential(
                      nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(1024),
                      nn.LeakyReLU())
       
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)


        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual_block1(x)
        x = self.conv3(x)
        x = self.residual_block2(x)
        x = self.conv4(x)
        x = self.residual_block3(x)
        x = self.conv5(x)
        x = self.residual_block4(x)
        x = self.conv6(x)
        x = self.residual_block5(x)
        x = self.global_avg_pool(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)

def MNISTDDDark(num_classes):
    return MNISTDDRGBDarknet53(DarkResidualBlock,num_classes)

class CustomModel:
    def __init__(self,pth):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(self.device)
        model = MNISTDDDark(28)
        model.load_state_dict(torch.load(pth, map_location=self.device))
        self.Darknet53 = model.to(self.device)
    def predict(self,image):
        self.Darknet53.eval()
        gray = image.reshape(64, 64, 3)
        with torch.no_grad():
            gray_tensor = torch.from_numpy(gray.astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).to(self.device)
            oh = self.Darknet53(gray_tensor)
            oh_class = oh[:, :20].contiguous().view(-1, 10)
            oh_box = oh[:, 20:]

            # Sort the tensor by ascending order
            pred_class = oh_class.argmax(1).cpu().numpy()
            pred_box = oh_box.long().cpu().numpy()[0].reshape(2,4)
            # pred_seg = oh_seg.argmax(1).cpu().numpy().reshape(64, 64)  

        return pred_class,pred_box
    

#####   IMAGE SEGMENTATION MODEL   #####

## REFERENCE  :  https://github.com/milesial/Pytorch-UNet/tree/master

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):        ### downscaling with maxpool
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):            ### upscaling 
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

        
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



def detect_and_segment(images):
    """

    :param np.ndarray images: N x 12288 array containing N 64x64x3 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """

    N = images.shape[0]
    

    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.empty((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.empty((N, 2, 4), dtype=np.float64)
    # pred_seg: Your predicted segmentation for the image, shape [N, 4096]
    pred_seg = np.empty((N, 4096), dtype=np.int32)

    # add your code here to fill in pred_class and pred_bboxes
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Image Classification and bboxes
    images_1 = images
    model = CustomModel("weightsclass.pth")

    # Image Segmentation
    images = images.reshape([images.shape[0], 64, 64, 3])
    images = np.transpose(images, (0, 3, 1, 2))
    model2 = UNet(3,11).to(device)
    model2.load_state_dict(torch.load("weightsseg.pth", map_location=device))
    model2 = model2.to(device)

    for i in range(N):
        label,box=model.predict(images_1[i,:])

        box[0,2] = box[0,0] + 28
        box[0,3] = box[0,1] + 28
        box[1,2] = box[1,0] + 28
        box[1,3] = box[1,1] + 28
        pred_class[i,:]=label
        pred_bboxes[i,:]=box

        image_seg = torch.as_tensor(images[i]).float()
        logit = model2(image_seg.to(device).view(-1,3,64,64))
        pred = logit.argmax(1).view(-1).long().cpu().numpy()
        pred_seg[i,:] = pred
    

    return pred_class, pred_bboxes, pred_seg
