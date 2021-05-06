import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn import init
import torchvision

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

# def weights_init_classifier(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         init.normal_(m.weight.data, 0, 0.001)
#         if m.bias:
#             init.zeros_(m.bias.data)


class ResNet50TA_rgb(nn.Module):
    """提取rgb特征，可调节attention生成函数"""
    def __init__(self, **kwargs):
        super(ResNet50TA_rgb, self).__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.att_gen = 'softmax' # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048 # feature dimension
        self.middle_dim = 256 # middle layer dimension
        #self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,4]) # 7,4 cooresponds to 224, 112 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        a = F.relu(self.attention_conv(x))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0,2,1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        x = F.avg_pool2d(x, x.size()[2:])
        if self. att_gen=='softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen=='sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
        else: 
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        x = x.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.feat_dim)
        att_x = torch.mul(x,a)
        att_x = torch.sum(att_x,1)
        
        feat1 = att_x.view(b,self.feat_dim)
        return feat1


class ResNet50TA_ir(nn.Module):
    """提取ir特征，可调节attention生成函数"""
    def __init__(self, **kwargs):
        super(ResNet50TA_ir, self).__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.att_gen = 'softmax' # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048 # feature dimension
        self.middle_dim = 256 # middle layer dimension
        #self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,4]) # 7,4 cooresponds to 224, 112 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        a = F.relu(self.attention_conv(x))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0,2,1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        x = F.avg_pool2d(x, x.size()[2:])
        if self. att_gen=='softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen=='sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
        else: 
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        x = x.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.feat_dim)
        att_x = torch.mul(x,a)
        att_x = torch.sum(att_x,1)
        
        feat2 = att_x.view(b,self.feat_dim)
        return feat2

class model_embedding(nn.Module):
    """共用全连接"""
    def __init__(self,**kwargs):
        super(model_embedding,self).__init__()
        self.feat_dim = 2048
        self.embedding = nn.Linear(self.feat_dim, 1024)
    def forward(self,x):

        feat = F.relu(x)
        feat_embedding = self.embedding(feat)
        feat_embedding = F.relu(feat_embedding)
        return feat_embedding

class binetwork(nn.Module):
    def __init__(self,num_classes,**kwargs):
        super(binetwork,self).__init__()
        self.classifier = nn.Linear(1024,num_classes)
        self.feature1 = ResNet50TA_rgb()
        self.feature2 = ResNet50TA_ir()
        self.embedding = model_embedding()
        self.l2norm = Normalize(2)


    def forward(self,x1,x2,modal=0):
        """三种模式，模式0用于训练，模式1、2提取rgb/ir图片信息"""
        if modal ==0:
            feat1 = self.feature1(x1)
            feat2 = self.feature2(x2)
            feat = torch.cat((feat1,feat2),0)
            # print(feat.shape)
        elif modal == 1:
            # ex rgb
            feat = self.feature1(x1)
        elif modal == 2:
            feat = self.feature2(x2)


        feat_embedding = self.embedding(feat)
        result = self.classifier(feat_embedding)

        if self.training:
            return feat_embedding, result
        else:
            return self.l2norm(feat_embedding)