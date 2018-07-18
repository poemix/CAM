# -*- coding: utf-8 -*-

# @Env      : windows python3.5
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @File     : grad_cam.py
# @Software : PyCharm


import cv2
import glob
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from pytorch_cam.model import FTNet


# def preprocess_image(img):
#     means = [0.485, 0.456, 0.406]
#     stds = [0.229, 0.224, 0.225]
#
#     preprocessed_img = img.copy()[:, :, ::-1]
#     for i in range(3):
#         preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
#         preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
#     preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
#     preprocessed_img = torch.from_numpy(preprocessed_img)
#     preprocessed_img.unsqueeze_(0)
#     input = Variable(preprocessed_img, requires_grad=True)
#     return input


class MyData(Dataset):
    def __init__(self, img_path, transformer=None, loader=default_loader):
        self.img_name = glob.glob('{}/*.png'.format(img_path))
        self.transformer = transformer
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        img = self.loader(img_name)

        if self.transformer is not None:
            try:
                img = self.transformer(img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img_name, img


class GradCAM:
    def __init__(self, model, use_cuda=False):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        self.gradients = []
        if self.cuda:
            self.model = model.cuda()

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def forward(self, x):
        # model结构[ft_model(resnet50), classifier]
        x = self.model.model.conv1(x)
        x = self.model.model.bn1(x)
        x = self.model.model.relu(x)
        x = self.model.model.maxpool(x)
        x = self.model.model.layer1(x)
        x = self.model.model.layer2(x)
        x = self.model.model.layer3(x)
        x = self.model.model.layer4(x)

        features = x  # 要提取的feature map [?, 2048, 7, 7]
        # 后向传播时，求在特征图上的梯度，并用变量来存储它方便后期获取
        features.register_hook(self.save_gradient)

        x = self.model.model.avgpool(x)
        # reshape
        x = x.view(-1, x.size(1))  # [?, 2048]
        outputs = self.model.classifier(x)

        return features, outputs

    def __call__(self, x, index=None):
        # x.shape=[1, 3, 224, 224]
        features, outputs = self.forward(x)

        # If None, returns the map for the highest scoring class.
        # Otherwise, targets the requested index.
        if index is None:
            index = np.argmax(outputs.cpu().data.numpy(), axis=1)

        # 图片数，类别数
        n_sample, n_class = outputs.size()
        one_hot = np.zeros((n_sample, n_class), dtype=np.float32)
        one_hot[:, index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)

        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * outputs)
        else:
            one_hot = torch.sum(one_hot * outputs)

        # 梯度清零
        self.model.model.zero_grad()
        self.model.classifier.zero_grad()

        # 提取的特征图
        targets = features.cpu().data.numpy()  # shape=[?, 2048, 7, 7]

        # 后向传播，求在提取的特征图上的梯度
        one_hot.backward()
        grads = self.gradients[-1].cpu().data.numpy()  # shape=[?, 2048, 7, 7]

        # 梯度求平均，求出的权重是 提取的特征图上通道的权重，有2048个通道就有2048个权重
        weights = np.mean(grads, axis=(2, 3))  # [?, 2048]
        # reshape 仅仅是便于计算，能将权重和特征图相乘
        weights = np.reshape(weights, [n_sample, -1, 1, 1])  # [?, 2048, 1, 1]

        # 特征图在通道上加权平均
        cams = np.multiply(weights, targets)  # shape=[?, 2048, 7, 7]
        cams = np.sum(cams, axis=1)  # shape=[?, 7, 7]

        # ReLu激活
        cams = np.maximum(cams, 0)
        return cams


if __name__ == '__main__':
    use_cuda = False
    height, width = 244, 244

    transformer = transforms.Compose([
        transforms.Resize((height, width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])
    data_set = MyData(img_path='../examples', transformer=transformer)
    data_loader = DataLoader(dataset=data_set, batch_size=2, shuffle=False, num_workers=2)
    model = FTNet(1036)
    grad_cam = GradCAM(model=model, use_cuda=use_cuda)

    for data in data_loader:
        # 一个batch_size大小数目的图片
        x_name, x = data
        cams = grad_cam(x)

        for i in range(cams.shape[0]):
            cam = cv2.resize(cams[i], (width, height))
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

            heatmap = heatmap.astype(np.float32)
            heatmap /= 255
            cam = heatmap + cv2.resize(cv2.imread(x_name[i]).astype(np.float32), (width, height))/255
            cam = cam / np.max(cam)
            split = x_name[i].replace('\\', '/').split('/')
            pre = '/'.join(split[:-1])
            sufix = 'cam-{}'.format(split[-1])
            cv2.imwrite('{}'.format(sufix), np.uint8(255 * cam))
