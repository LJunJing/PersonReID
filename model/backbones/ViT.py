import math
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embedding_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)

        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.linear_project = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embedding_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        # 如果H和W分别等于了输入图片的高和宽，则x的大小不符合模型的要求
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        #flatten: [B,C,H,W] -> [B,C,H*W]
        #transpose: [B,C,H*W] -> [B,H*W,C]
        x = self.linear_project(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim*3, bias=False)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        :param x: [batch_size, num_patches+1, total_embedding_dim]
        :return:
        """
        B, N, C = x.shape
        #qkv: -> [batch_size, num_patches+1, 3*total_embedding_dim]
        #reshape: -> [batch_size, num_patches+1, 3, num_heads, embedding_dim_per_head]
        #permute: -> [3, batch_size, num_heads, num_patched+1, embedding_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        #q、k、v: [batch_size, num_heads, num_patches+1, embedding_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        d_k = k.size(-1)
        scale_attn = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weight = F.softmax(scale_attn, dim=-1)
        #attention_weight和v点乘
        attn = torch.matmul(attn_weight, v).transpose(1, 2).reshape(B, N, C)
        attn = self.proj(attn)
        attn = self.proj_drop(attn)

        return attn


# class Mlp(nn.Module):
#     def __init__(self,in_features, hidden_features=None, out_features=None, dropout=0):
#         super(Mlp, self).__init__()
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drouput = nn.Dropout(p=dropout)
#
#     def forward(self, x):
#         return self.fc2(self.drouput(F.relu(self.fc1(x))))

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == None or self.training == False:
            return x
        keep_prob = 1-self.drop_prob
        shape = (x.shape[0], ) + (1, ) * (x.ndim-1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device = x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class Transformer(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop_path=0.):
        super(Transformer, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.mlp(self.norm(x)))
        return x


class ViT(nn.Module):
    # def __init__(self, embedding_dim=768, distilled=False, dropout=0, depth=12):
    def __init__(self,img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embedding_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, dropout=0,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):

        super(ViT, self).__init__()
        self.num_features = self.embed_dim = embedding_dim
        self.num_tokens = 2 if distilled else 1

        self.patch_embed = PatchEmbed()
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embedding_dim)) if distilled else None
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches+self.num_tokens, embedding_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        self.transformer = nn.Sequential(*[
            Transformer(dim=embedding_dim)
            for i in range(depth)
        ])
        self.norm = None or partial(nn.LayerNorm, eps=1e-6)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embedding_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        """
        :param x: [B, 3, 224, 224]
        :return:
        """
        #[B,C,H,W] -> [B,num_patched,embed_dim]
        x = self.patch_embed(x)  # [B,3,224,224] -> [B,14,14,768] -> [B, 196, 768]

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # [1,1,768] -> [B,1,768]

        if self.dist_token is not None:
            dist_token = self.dist_token.expand(x.shapa[0], -1, -1)
            x = torch.cat((cls_token, dist_token, x), dim=1)
        else:
            x = torch.cat((cls_token, x), dim=1)

        x = self.pos_drop(x + self.pos_embedding)
        x = self.transformer(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x, label=None):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

# vit = ViT()
# print(vit)