# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @File     : model.py
# @Software : PyCharm


from torch import nn
from torch.nn import init
from torchvision import models


def weight_init_kamming(module):
    class_name = module.__class__.__name__
    if class_name.find('Conv') != -1:
        init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
    elif class_name.find('Linear') != -1:
        init.kaiming_normal_(module.weight, a=0, mode='fan_out')
        init.constant_(module.bias.data, 0.0)
    elif class_name.find('BatchNorm') != -1:
        init.normal_(module.weight.data, 1.0, 0.02)
        init.constant_(module.bias.data, 0.0)


def weight_init_classifier(module):
    class_name = module.__class__.__name__
    if class_name.find('Linear') != -1:
        init.normal_(module.weight.data, std=0.001)
        init.constant_(module.bias.data, 0.0)


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]

        if dropout:
            add_block += [nn.Dropout(p=0.5)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weight_init_kamming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weight_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


class FTNet(nn.Module):
    def __init__(self, class_num):
        super(FTNet, self).__init__()
        ft_model = models.resnet50(pretrained=True)
        # 更改预训练模型renet的avgpool module
        ft_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 删除预训练模型resnet的fc module
        del ft_model._modules['fc']
        self.model = ft_model
        self.classifier = ClassBlock(2048, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.ft_model.avgpool(x)
        # reshape
        x = x.view(-1, x.size(1))  # [?, 2048]
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    ft_net = FTNet(100)
    print(ft_net)
