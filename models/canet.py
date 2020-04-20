from .resnet import ResNetExtractor

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import models


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                kernel_size, stride, padding, dilation):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=True),
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )
    
    def forward(self, x):
        return self.conv(x)

class DenseComparisonModule(nn.Module):
    def __init__(self, extractor_channels, in_channels, out_channels):
        super().__init__()
        self.layer1 = Conv2dBlock(in_channels=extractor_channels, 
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=2,
                                  dilation=2)
        self.layer2 = Conv2dBlock(in_channels=in_channels*2,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=2,
                                  dilation=2)
     
    def forward(self, support, annotation, query):
        h, w = support.shape[-2:][0], support.shape[-2:][1]
        # Query Image Part 
        query = self.layer1(query)

        # Support Image Part 
        support = self.layer1(support)

        annotation = F.interpolate(annotation, support.shape[-2:], mode='bilinear',align_corners=True)
        
        pool_area = nn.AvgPool2d(kernel_size=support.shape[-2:])
        area = pool_area(annotation) * h * w + 0.0001

        concat = annotation * support
        pool = nn.AvgPool2d(kernel_size=support.shape[-2:])
        concat = pool(concat) * h * w / area 
        
        concat = F.interpolate(concat, query.shape[-2:], mode='bilinear', align_corners=True)

        result = torch.cat([query, concat], dim=1)
        result = self.layer2(result)
        return result

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=True),
            nn.ReLU()    
        )

    def forward(self, x):
        return self.res(x)

class IterativeOptimizationModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = Conv2dBlock(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   dilation=1)

        self.resblock1 = ResBlock(in_channels=in_channels*2,
                                  out_channels=out_channels)
        self.resblock2 = ResBlock(in_channels=in_channels,
                                  out_channels=out_channels)
        self.resblock3 = ResBlock(in_channels=in_channels,
                                  out_channels=out_channels)

        # ASPP block: includes 4 layers
        self.aspp_layer1 = Conv2dBlock(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0, dilation=1)
        self.aspp_layer2 = Conv2dBlock(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=6,
                                       dilation=6)
        self.aspp_layer3 = Conv2dBlock(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=12,
                                       dilation=12)
        self.aspp_layer4 = Conv2dBlock(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=18,
                                       dilation=18)
        self.aspp_layer5 = Conv2dBlock(in_channels=in_channels*5,
                                       out_channels=out_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       dilation=1)

    def forward(self, x, out):
        # x is the result of DCM
        concat = torch.cat([x, out], dim=1)
        x = x + self.resblock1(concat)
        x = x + self.resblock2(x)
        x = x + self.resblock3(x)

        out = torch.cat([out, self.aspp_layer1(x), self.aspp_layer2(x), 
                        self.aspp_layer3(x), self.aspp_layer4(x)], dim=1)
        out = self.aspp_layer5(out)
        return out

class CANet(nn.Module):

    def __init__(self, num_class, extractor, t=4):
        super().__init__()
        self.t = t
        self.extractor = extractor
        #512 --> 1024
        self.DCM = DenseComparisonModule(extractor_channels=512, 
                                in_channels=256, out_channels=256)

        self.IOM = IterativeOptimizationModule(in_channels=256,
                                        out_channels=256)

        self.conv1x1 = Conv2dBlock(in_channels=256,
                                   out_channels=256,
                                   kernel_size=1, 
                                   stride=1,
                                   padding=0,
                                   dilation=1)
        self.final_layer = nn.Conv2d(256, num_class, kernel_size=1,
                                    stride=1, bias=True)

    def forward(self, support, annotation, query):
        query_extraction = self.extractor(query)
        support_extraction = self.extractor(support)
        
        x = self.DCM(support_extraction, annotation, query_extraction)
        
        out = self.conv1x1(x)

        for i in range(self.t):
            out = self.IOM(x, out)

        final_out = self.final_layer(out)
        return final_out 
                                        

def test():
    query_rgb = torch.rand(1,3,854,480).cpu()
    support_rgb = torch.rand(1,3,854,480).cpu()
    support_mask = torch.rand(1,1,854,480).cpu()

    extractor = ResNetExtractor('resnet50').cpu()
    print(extractor)


if __name__ == "__main__":
    test()