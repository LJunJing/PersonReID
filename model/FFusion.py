import math
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet50
import torch.nn.functional as F
import numpy as np

from model.backbones.Transformer import deit_small_patch16_224 as Deit


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x) + self.identity(x))

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1+in_ch2, out_ch) # conv之后修改了x的维度，从 in_ch1+in_ch2 -> out_ch

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.up(x1) # [B,C,H,W] -> [B,C,2*H,2*W]

        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

# def save_img(img,filename):
#     """
#     将tensor转为numpy再保存为RGB图片，适用于[1,3,h,w]格式的tensor
#     img：要保存的tensor
#     filename：保存的文件名
#     """
#     output = img.permute(0,2,3,1)
#     output = output.cpu().detach().numpy().squeeze()
#     output = Image.fromarray(np.uint8(output)).convert('RGB')
#     output.save('results/{}'.format(filename))


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out

class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFusion_block, self).__init__()

        # channel attention for F_g, use SE Block
        self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)

        # spatial attention for cnn branch
        g_in = g
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in

        # channel attetion for transformer branch
        x_in = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * x_in
        fuse = self.residual(torch.cat([g, x, bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse

class FFusion_cnn(nn.Module):
    def __init__(self, num_classes, dropout=0.2, pretrained=True):
        super(FFusion_cnn, self).__init__()

        self.cnn = resnet50()
        self.model_name = 'resnet50'
        if pretrained:
            self.cnn.load_state_dict(torch.load('pretrained/resnet50-19c8e357.pth'))
        self.cnn.fc = nn.Identity()
        self.cnn.layer4 = nn.Identity()

        self.drop = nn.Dropout2d(dropout)

        self.classifier = nn.Linear(1024, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck2 = nn.BatchNorm1d(1024)
        self.bottleneck2.bias.requires_grad_(False)
        self.bottleneck2.apply(weights_init_kaiming)

    def name(self):
        return self.model_name

    def forward(self, x, labels=None):
        """
        :param x: [32,3,224,224]
        :param labels:
        :return:
        """
        x_c = self.cnn.conv1(x)
        x_c = self.cnn.bn1(x_c)
        x_c = self.cnn.relu(x_c)
        x_c = self.cnn.maxpool(x_c)  # x_c: [32,64,64,32]
        # print("x_c",x_c.shape)

        x_c_2 = self.cnn.layer1(x_c)
        x_c_2 = self.drop(x_c_2)  # x_c_2:[32,256,64,32]
        # print("x_c_2", x_c_2.shape)

        x_c_1 = self.cnn.layer2(x_c_2)
        x_c_1 = self.drop(x_c_1)  # x_c_1:[32,512,32,16]
        # print("x_c_1", x_c_1.shape)

        x_c_0 = self.cnn.layer3(x_c_1)
        x_c_0 = self.drop(x_c_0)  # x_c_0:[32,1024,16,8]
        # print("x_c_0", x_c_0.shape)

        # map_x = F.interpolate(self.final_x(x_c_0), scale_factor=16, mode='bilinear', align_corners=True)   # f0(第一个BiFusion Module的输出)

        out = nn.functional.avg_pool2d(x_c_0, x_c_0.shape[2:4])
        out = out.view(out.shape[0], -1)  # out:[32, 1024]
        out = self.bottleneck2(out)  # out:[32, 1024]

        if self.training:
            cls_score = self.classifier(out)  # out:[32, 751]
            return cls_score, out  # global feature for triplet loss
        else:
            return out

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape,
                                                                                                self.state_dict()[
                                                                                                    k].shape))

class FFusion_deit(nn.Module):
    def __init__(self, num_classes, dropout=0.2, pretrained=True):
        super(FFusion_deit, self).__init__()

        self.trans = Deit()
        self.model_name = 'deit'

        self.drop = nn.Dropout2d(dropout)

        self.classifier = nn.Linear(384, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(384)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def name(self):
        return self.model_name

    def forward(self, x, labels=None):
        """
        :param x: [32, 3, 256, 128]
        :param labels:
        :return:
        """
        x_t = self.trans(x)[:, 0]  # [32,384]
        feat = self.bottleneck(x_t)  # [32,384]

        if self.training:
            cls_score = self.classifier(feat)  # [32,751]
            return cls_score, x_t  # global feature for triplet loss
        else:
            return feat

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape,
                                                                                                self.state_dict()[
                                                                                                    k].shape))

class FFusion(nn.Module):
    def __init__(self, num_classes, dropout=0.2, pretrained=True, normal_init=True):
        super(FFusion, self).__init__()
        self.model_name = 'ffusion'

        self.cnn = resnet50()
        if pretrained:
            self.cnn.load_state_dict(torch.load('pretrained/resnet50-19c8e357.pth'))
        self.cnn.fc = nn.Identity()
        self.cnn.layer4 = nn.Identity()

        self.trans = Deit()

        self.bottleneck = nn.BatchNorm1d(384)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.up1 = Up(in_ch1=384, out_ch=128)
        self.up2 = Up(128, 64)

        self.final_x = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.final = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 384, 3, bn=False, relu=False)
        )

        self.up_c = BiFusion_block(ch_1=1024, ch_2=384, r_2=4, ch_int=256, ch_out=256, drop_rate=dropout / 2)

        self.up_c_1_1 = BiFusion_block(ch_1=512, ch_2=128, r_2=2, ch_int=128, ch_out=128, drop_rate=dropout / 2)
        self.up_c_1_2 = Up(in_ch1=256, out_ch=128, in_ch2=128, attn=True)

        self.up_c_2_1 = BiFusion_block(ch_1=256, ch_2=64, r_2=1, ch_int=64, ch_out=64, drop_rate=dropout / 2)
        self.up_c_2_2 = Up(128, 64, 64, attn=True)

        self.drop = nn.Dropout2d(dropout)

        self.classifier = nn.Linear(384, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        if normal_init:
            self.init_weights()

    def name(self):
        return self.model_name

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)

    def forward(self, x, labels=None):
        """
        x:[32,3,256,128]
        """

        # bottom-up path [Transformer Branch]
        x_t = self.trans(x)  # [32,128,384]
        # feat = self.bottleneck(x_t)  # [32,128,384]
        # print("x_t", x_t)

        x_t = torch.transpose(x_t, 1, 2)
        x_t_0 = x_t.view(x_t.shape[0], -1, 16, 8)  # [32,384,16,8]
        x_t_0 = self.drop(x_t_0)  # [32,384,16,8]
        # print("x_t",x_t_0.shape)

        x_t_1 = self.up1(x_t_0)  # t1:[32,128,32,16]
        x_t_1 = self.drop(x_t_1)
        # print("x_t_1",x_t_1.shape)

        x_t_2 = self.up2(x_t_1)  # t2:[32,64,64,32]  transformer pred supervise here.
        x_t_2 = self.drop(x_t_2)
        # print("x_t_2", x_t_2.shape)

        # top-down path [CNN Branch]
        x_c = self.cnn.conv1(x)
        x_c = self.cnn.bn1(x_c)
        x_c = self.cnn.relu(x_c)
        x_c = self.cnn.maxpool(x_c)  # x_c: [32,64,64,32]

        x_c_2 = self.cnn.layer1(x_c)
        x_c_2 = self.drop(x_c_2)  # x_c_2:[32,256,64,32]
        # print("x_c_2",x_c_2.shape)

        x_c_1 = self.cnn.layer2(x_c_2)
        x_c_1 = self.drop(x_c_1)  # x_c_1:[32,512,32,16]

        x_c_0 = self.cnn.layer3(x_c_1)
        x_c_0 = self.drop(x_c_0)  # x_c_0:[32,1024,16,8]

        # joint path [BiFusion Module]
        x_f = self.up_c(x_c_0, x_t_0)  # ^f0 = f0 = up_c(g0,t0):[32,256,16,8]

        x_f_1_1 = self.up_c_1_1(x_c_1, x_t_1)  # f1 = up_c_1_1(g1,t1):[32,128,32,16]
        x_f_1 = self.up_c_1_2(x_f, x_f_1_1)  # ^f1 = up_c_1_2(^f0,f1):[32,128,32,16]

        x_f_2_1 = self.up_c_2_1(x_c_2, x_t_2)  # f2 = up_c_2_1(g2,t2):[32,64,64,32]
        x_f_2 = self.up_c_2_2(x_f_1, x_f_2_1)  # ^f2 = up_c_2_2(^f1,f2):[32,64,64,32]  joint predict low supervise here

        # # decoder part
        # map_x = F.interpolate(self.final_x(x_f), scale_factor=16, mode='bilinear', align_corners=True)  # f0(第一个BiFusion Module的输出)
        # map_1 = F.interpolate(self.final_1(x_t_2), scale_factor=4, mode='bilinear', align_corners=True)  # t2
        # map_2 = F.interpolate(self.final_2(x_f_2), scale_factor=4, mode='bilinear', align_corners=True)  # ^f2
        # print("map_x", map_x.shape)
        # print("map_1", map_1.shape)
        # print("map_2", map_2.shape)
        # return map_x, map_1, map_2

        s_f = self.final_x(x_f)[:, :, 0, 0]  # [32,751,16,8] -> [32,751]
        s_t_2 = self.final_1(x_t_2)[:, :, 0, 0]  # [32,751,64,32] -> [32,751]
        s_f_2 = self.final_2(x_f_2)[:, :, 0, 0]  # [32,751,64,32] -> [32,751]

        # print("x_f_2", x_f_2.shape)  # [32,64,64,32]
        # x_f_2 = x_f_2.view(x_f_2.shape[0], 384, -1)[:, :, 0]  # [32,384,16,8]
        x_f_2 = self.final(x_f_2)[:, :, 0, 0]
        # x_f_2= torch.transpose(x_f_2, 1, 2)[:, :, 0]

        if self.training:
            # return s_f, s_t_2, s_f_2, x_f, x_t_2, x_f_2  # global feature for triplet loss
            return s_f_2, x_f_2
        else:
            return x_f_2

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))

def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb

def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)