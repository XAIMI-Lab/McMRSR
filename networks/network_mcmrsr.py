import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import functools
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) dc_mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention dc_mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, nf, kernel_size=3, stride=1, padding=1, dilation=1, act='relu'):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.conv2(self.act(self.conv1(x)))

        return out + x

class SAB(nn.Module):
    def __init__(self, nf, use_residual=True, learnable=True):
        super(SAB, self).__init__()

        self.learnable = learnable
        self.norm_layer = nn.InstanceNorm2d(nf, affine=False)

        if self.learnable:
            self.conv_shared = nn.Sequential(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True),
                                             nn.ReLU(inplace=True))
            self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

            self.use_residual = use_residual

            # initialization
            self.conv_gamma.weight.data.zero_()
            self.conv_beta.weight.data.zero_()
            self.conv_gamma.bias.data.zero_()
            self.conv_beta.bias.data.zero_()

    def forward(self, lr, ref):
        ref_normed = self.norm_layer(ref)
        if self.learnable:
            style = self.conv_shared(torch.cat([lr, ref], dim=1))
            gamma = self.conv_gamma(style)
            beta = self.conv_beta(style)

        b, c, h, w = lr.size()
        lr = lr.view(b, c, h * w)
        lr_mean = torch.mean(lr, dim=-1, keepdim=True).unsqueeze(3)
        lr_std = torch.std(lr, dim=-1, keepdim=True).unsqueeze(3)

        if self.learnable:
            if self.use_residual:
                gamma = gamma + lr_std
                beta = beta + lr_mean
            else:
                gamma = 1 + gamma
        else:
            gamma = lr_std
            beta = lr_mean

        out = ref_normed * gamma + beta

        return out

class JRFAB(nn.Module):
    def __init__(self, nf):
        super(JRFAB, self).__init__()
        self.conv_down_a = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv_up_a = nn.ConvTranspose2d(nf, nf, 3, 2, 1, 1, bias=True)
        self.conv_down_b = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv_up_b = nn.ConvTranspose2d(nf, nf, 3, 2, 1, 1, bias=True)
        self.conv_cat = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, lr, ref):
        res_a = self.act(self.conv_down_a(ref)) - lr
        out_a = self.act(self.conv_up_a(res_a)) + ref

        res_b = lr - self.act(self.conv_down_b(ref))
        out_b = self.act(self.conv_up_b(res_b + lr))

        out = self.act(self.conv_cat(torch.cat([out_a, out_b], dim=1)))

        return out

class Conv2D(nn.Module):
    def __init__(self, in_chl, nf, n_blks=[1, 1, 1], act='relu'):
        super(Conv2D, self).__init__()

        block = functools.partial(ResidualBlock, nf=nf)
        self.conv_L1 = nn.Conv2d(in_chl, nf, 3, 1, 1, bias=True)
        self.blk_L1 = make_layer(block, n_layers=n_blks[0])

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        fea_L1 = self.blk_L1(self.act(self.conv_L1(x)))

        return fea_L1

class MAB(nn.Module):
    def __init__(self, nf, out_chl, n_blks, upscale=4):
        super(MAB, self).__init__()
        block = functools.partial(ResidualBlock, nf=nf)

        ### spatial adaptation block ##
        self.SAB = SAB(nf, use_residual=True, learnable=True)
        ### joint residual feature aggregation block ##
        self.JRFAB = JRFAB(nf)

        self.blk_x1 = make_layer(block, n_blks[3])
        self.blk_x2 = make_layer(block, n_blks[4])
        self.blk_x4 = make_layer(functools.partial(ResidualBlock, nf=nf), n_blks[5])
        self.merge = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)

        self.conv_out = nn.Conv2d(nf, out_chl, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, tar_lr, F_M, upscale):

        if upscale == 2:
            warp_ref_x1 = self.SAB(tar_lr, F_M[1])
            fea_x1 = self.act(self.merge(torch.cat([warp_ref_x1, tar_lr], dim=1)))
            fea_x1 = self.blk_x1(fea_x1)
            fea_x1_up = F.interpolate(fea_x1, scale_factor=2, mode='bilinear', align_corners=False)

            warp_ref_x2 = self.SAB(fea_x1_up, F_M[0])
            fea_x2 = self.JRFAB(fea_x1, warp_ref_x2)
            fea_x2 = self.blk_x2(fea_x2)

            out = self.conv_out(fea_x2)

        elif upscale == 4:
            warp_ref_x1 = self.SAB(tar_lr, F_M[2])
            fea_x1 = self.act(self.merge(torch.cat([warp_ref_x1, tar_lr], dim=1)))
            fea_x1 = self.blk_x1(fea_x1)
            fea_x1_up = F.interpolate(fea_x1, scale_factor=2, mode='bilinear', align_corners=False)

            warp_ref_x2 = self.SAB(fea_x1_up, F_M[1])
            fea_x2 = self.JRFAB(fea_x1, warp_ref_x2)
            fea_x2 = self.blk_x2(fea_x2)
            fea_x2_up = F.interpolate(fea_x2, scale_factor=2, mode='bilinear', align_corners=False)

            warp_ref_x4 = self.SAB(fea_x2_up, F_M[0])
            fea_x4 = self.JRFAB(fea_x2, warp_ref_x4)
            fea_x4 = self.blk_x4(fea_x4)

            out = self.conv_out(fea_x4)

        return out


class McMRSR(nn.Module):
    r""" McMRSR
        A PyTorch impl of : `Transformer-empowered Multi-scale Contextual Matching and Aggregation for Multi-contrast MRI Super-resolution`, based on SwinIR.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 2
        embed_dim (int): Patch embedding dimension. Default: 60
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/4 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=2,
                 embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1.,
                 **kwargs):
        super(McMRSR, self).__init__()
        num_in_ch = in_chans

        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.window_size = window_size
        n_blks = [2, 2, 2]
        n_blks_dec = [2, 2, 2, 12, 8, 4]
        self.lr_block_size = 8
        self.ref_down_block_size = 1.5
        self.dilations = [1, 2, 3]
        self.num_nbr = 1
        self.psize = 3

        self.MAB = MAB(embed_dim, num_in_ch, n_blks=n_blks_dec, upscale=self.upscale)

        #####################################################################################################
        ################################### 1, Tar/Ref LR feature extraction ###################################
        self.conv2d = Conv2D(in_chl=num_in_ch, nf=embed_dim, n_blks=n_blks)

        #####################################################################################################
        ################################### 2, Reference feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.conv_second = nn.Conv2d(embed_dim, embed_dim, 3, 2, 1)
        self.conv_third = nn.Conv2d(embed_dim, embed_dim, 3, 2, 1)

        #####################################################################################################
        ################################### 3, deep feature extraction (STG) ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction (Conv2d)
        self.conv_after_RSTB = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, C*k*k, H*W]
        # dim: scalar > 0
        # index: [N, Hi, Wi]
        views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]  # views = [N, 1, -1]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1  # expanse = [-1, C*k*k, -1]
        index = index.clone().view(views).expand(expanse)  # [N, Hi, Wi] -> [N, 1, Hi*Wi] - > [N, C*k*k, Hi*Wi]
        return torch.gather(input, dim, index)  # [N, C*k*k, Hi*Wi]

    def search(self, lr, reflr, ks=3, pd=1, stride=1, dilations=[1, 2, 4]):
        # lr: [N, p*p, C, k_y, k_x]
        # reflr: [N, C, Hr, Wr]

        N, C, Hr, Wr = reflr.size()
        _, _, _, k_y, k_x = lr.size()
        x, y = k_x // 2, k_y // 2
        corr_sum = 0
        for i, dilation in enumerate(dilations):
            reflr_patches = F.unfold(reflr, kernel_size=(ks, ks), padding=dilation, stride=stride, dilation=dilation)  # [N, C*ks*ks, Hr*Wr]
            lr_patches = lr[:, :, :, y - dilation: y + dilation + 1: dilation,
                                     x - dilation: x + dilation + 1: dilation]  # [N, p*p, C, ks, ks]
            lr_patches = lr_patches.contiguous().view(N, -1, C * ks * ks)  # [N, p*p, C*ks*ks]

            lr_patches = F.normalize(lr_patches, dim=2)
            reflr_patches = F.normalize(reflr_patches, dim=1)
            corr = torch.bmm(lr_patches, reflr_patches)  # [N, p*p, Hr*Wr]
            corr_sum = corr_sum + corr

        sorted_corr, ind_l = torch.topk(corr_sum, self.num_nbr, dim=-1, largest=True, sorted=True)  # [N, p*p, num_nbr]

        return sorted_corr, ind_l

    def make_grid(self, idx_x1, idx_y1, diameter_x, diameter_y, s):
        idx_x1 = idx_x1 * s
        idx_y1 = idx_y1 * s
        idx_x1 = idx_x1.view(-1, 1).repeat(1, diameter_x * s)
        idx_y1 = idx_y1.view(-1, 1).repeat(1, diameter_y * s)
        idx_x1 = idx_x1 + torch.arange(0, diameter_x * s, dtype=torch.long, device=idx_x1.device).view(1, -1)
        idx_y1 = idx_y1 + torch.arange(0, diameter_y * s, dtype=torch.long, device=idx_y1.device).view(1, -1)

        ind_y_l = []
        ind_x_l = []
        for i in range(idx_x1.size(0)):
            grid_y, grid_x = torch.meshgrid(idx_y1[i], idx_x1[i])
            ind_y_l.append(grid_y.contiguous().view(-1))
            ind_x_l.append(grid_x.contiguous().view(-1))
        ind_y = torch.cat(ind_y_l)
        ind_x = torch.cat(ind_x_l)

        return ind_y, ind_x

    def search_org(self, lr, reflr, ks=3, pd=1, stride=1):
        # lr: [N, C, H, W]
        # reflr: [N, C, Hr, Wr]

        batch, c, H, W = lr.size()
        _, _, Hr, Wr = reflr.size()

        reflr_unfold = F.unfold(reflr, kernel_size=(ks, ks), padding=0, stride=stride)  # [N, C*k*k, Hr*Wr]
        lr_unfold = F.unfold(lr, kernel_size=(ks, ks), padding=0, stride=stride)
        lr_unfold = lr_unfold.permute(0, 2, 1)  # [N, H*W, C*k*k]

        lr_unfold = F.normalize(lr_unfold, dim=2)
        reflr_unfold = F.normalize(reflr_unfold, dim=1)

        corr = torch.bmm(lr_unfold, reflr_unfold)  # [N, H*W, Hr*Wr]
        corr = corr.view(batch, H-2, W-2, (Hr-2)*(Wr-2))
        sorted_corr, ind_l = torch.topk(corr, self.num_nbr, dim=-1, largest=True, sorted=True)  # [N, H, W, num_nbr]

        return sorted_corr, ind_l

    def transfer(self, fea, index, soft_att, ks=3, pd=1, stride=1):
        # fea: [N, C, H, W]
        # index: [N, Hi, Wi]
        # soft_att: [N, 1, Hi, Wi]
        scale = stride

        fea_unfold = F.unfold(fea, kernel_size=(ks, ks), padding=0, stride=stride)  # [N, C*k*k, H*W]
        out_unfold = self.bis(fea_unfold, 2, index)  # [N, C*k*k, Hi*Wi]
        divisor = torch.ones_like(out_unfold)

        _, Hi, Wi = index.size()
        out_fold = F.fold(out_unfold, output_size=(Hi*scale, Wi*scale), kernel_size=(ks, ks), padding=pd, stride=stride)
        divisor = F.fold(divisor, output_size=(Hi*scale, Wi*scale), kernel_size=(ks, ks), padding=pd, stride=stride)
        soft_att_resize = F.interpolate(soft_att, size=(Hi*scale, Wi*scale), mode='bilinear',align_corners=True)
        out_fold = out_fold / divisor * soft_att_resize
        # out_fold = out_fold / (ks*ks) * soft_att_resize
        return out_fold

    def forward_features_RSTB(self,x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    ## Multi-scale Contextual Matching based on MASA-SR##
    ## https://github.com/dvlab-research/MASA-SRhttps://github.com/dvlab-research/MASA-SR ##
    def contextual_matching(self, tar_lr, ref_lr, ref_feature, upscale):

        bs, _, h, w = tar_lr.size()
        px = w // self.lr_block_size
        py = h // self.lr_block_size
        k_x = w // px
        k_y = h // py
        _, _, h, w = ref_lr.size()
        diameter_x = 2 * int(w // (2 * px) * self.ref_down_block_size) + 1
        diameter_y = 2 * int(h // (2 * py) * self.ref_down_block_size) + 1

        ### begin ####
        N, C, H, W = tar_lr.size()
        _, _, Hr, Wr = ref_lr.size()

        lr_patches = F.pad(tar_lr, pad=(1, 1, 1, 1), mode='replicate')
        lr_patches = F.unfold(lr_patches, kernel_size=(k_y + 2, k_x + 2), padding=(0, 0), stride=(k_y, k_x))  # [N, C*(k_y+2)*(k_x+2), py*px]
        lr_patches = lr_patches.view(N, C, k_y + 2, k_x + 2, py * px).permute(0, 4, 1, 2, 3)  # [N, py*px, C, k_y+2, k_x+2]

        ## find the corresponding ref patch for each lr patch
        sorted_corr, ind_l = self.search(lr_patches, ref_lr, ks=3, pd=1, stride=1, dilations=self.dilations)

        ## crop corresponding ref patches
        index = ind_l[:, :, 0]  # [N, py*px]
        idx_x = index % Wr
        idx_y = index // Wr
        idx_x1 = idx_x - diameter_x // 2 - 1
        idx_x2 = idx_x + diameter_x // 2 + 1
        idx_y1 = idx_y - diameter_y // 2 - 1
        idx_y2 = idx_y + diameter_y // 2 + 1

        mask = (idx_x1 < 0).long()
        idx_x1 = idx_x1 * (1 - mask)
        idx_x2 = idx_x2 * (1 - mask) + (diameter_x + 1) * mask

        mask = (idx_x2 > Wr - 1).long()
        idx_x2 = idx_x2 * (1 - mask) + (Wr - 1) * mask
        idx_x1 = idx_x1 * (1 - mask) + (idx_x2 - (diameter_x + 1)) * mask

        mask = (idx_y1 < 0).long()
        idx_y1 = idx_y1 * (1 - mask)
        idx_y2 = idx_y2 * (1 - mask) + (diameter_y + 1) * mask

        mask = (idx_y2 > Hr - 1).long()
        idx_y2 = idx_y2 * (1 - mask) + (Hr - 1) * mask
        idx_y1 = idx_y1 * (1 - mask) + (idx_y2 - (diameter_y + 1)) * mask

        ind_y_x1, ind_x_x1 = self.make_grid(idx_x1, idx_y1, diameter_x + 2, diameter_y + 2, 1)
        ind_y_x2, ind_x_x2 = self.make_grid(idx_x1, idx_y1, diameter_x + 2, diameter_y + 2, 2)
        ind_y_x4, ind_x_x4 = self.make_grid(idx_x1, idx_y1, diameter_x + 2, diameter_y + 2, 4)

        ind_b = torch.repeat_interleave(torch.arange(0, N, dtype=torch.long, device=idx_x1.device), py * px * (diameter_y + 2) * (diameter_x + 2))
        ind_b_x2 = torch.repeat_interleave(torch.arange(0, N, dtype=torch.long, device=idx_x1.device), py * px * ((diameter_y + 2) * 2) * ((diameter_x + 2) * 2))
        ind_b_x4 = torch.repeat_interleave(torch.arange(0, N, dtype=torch.long, device=idx_x1.device), py * px * ((diameter_y + 2) * 4) * ((diameter_x + 2) * 4))

        if upscale == 2:
            reflr_patches = ref_lr[ind_b, :, ind_y_x1, ind_x_x1].view(N * py * px, diameter_y + 2, diameter_x + 2, C).permute(0, 3, 1, 2).contiguous()  # [N*py*px, C, (radius_y+1)*2, (radius_x+1)*2]
            ref_patches_x1 = ref_feature[1][ind_b, :, ind_y_x1, ind_x_x1].view(N * py * px, diameter_y + 2, diameter_x + 2, C).permute(0, 3, 1, 2).contiguous()
            ref_patches_x2 = ref_feature[0][ind_b_x2, :, ind_y_x2, ind_x_x2].view(N * py * px, (diameter_y + 2) * 2, (diameter_x + 2) * 2, C).permute(0, 3, 1, 2).contiguous()

            ## calculate correlation between lr patches and their corresponding ref patches
            lr_patches = lr_patches.contiguous().view(N * py * px, C, k_y + 2, k_x + 2)
            corr_all_l, index_all_l = self.search_org(lr_patches, reflr_patches, ks=self.psize, pd=self.psize // 2, stride=1)

            index_all = index_all_l[:, :, :, 0]  # [N*p*p, k_y, k_x]
            soft_att_all = corr_all_l[:, :, :, 0:1].permute(0, 3, 1, 2)  # [N*p*p, 1, k_y, k_x]

            warp_ref_patches_x1 = self.transfer(ref_patches_x1, index_all, soft_att_all, ks=self.psize, pd=self.psize // 2, stride=1)  # [N*py*px, C, k_y, k_x]
            warp_ref_patches_x2 = self.transfer(ref_patches_x2, index_all, soft_att_all, ks=self.psize * 2, pd=self.psize // 2 * 2, stride=2)  # [N*py*px, C, k_y*2, k_x*2]

            warp_ref_patches_x1 = warp_ref_patches_x1.view(N, py, px, C, H // py, W // px).permute(0, 3, 1, 4, 2, 5).contiguous()  # [N, C, py, H//py, px, W//px]
            warp_ref_patches_x1 = warp_ref_patches_x1.view(N, C, H, W)

            warp_ref_patches_x2 = warp_ref_patches_x2.view(N, py, px, C, H // py * 2, W // px * 2).permute(0, 3, 1, 4, 2, 5).contiguous()  # [N, C, py, H//py*2, px, W//px*2]
            warp_ref_patches_x2 = warp_ref_patches_x2.view(N, C, H * 2, W * 2)

            F_M = [warp_ref_patches_x2, warp_ref_patches_x1]

        elif upscale == 4:
            reflr_patches = ref_lr[ind_b, :, ind_y_x1, ind_x_x1].view(N * py * px, diameter_y + 2, diameter_x + 2, C).permute(0, 3, 1, 2).contiguous()  # [N*py*px, C, (radius_y+1)*2, (radius_x+1)*2]
            ref_patches_x1 = ref_feature[2][ind_b, :, ind_y_x1, ind_x_x1].view(N * py * px, diameter_y + 2, diameter_x + 2, C).permute(0, 3, 1, 2).contiguous()
            ref_patches_x2 = ref_feature[1][ind_b_x2, :, ind_y_x2, ind_x_x2].view(N * py * px, (diameter_y + 2) * 2, (diameter_x + 2) * 2, C).permute(0, 3, 1, 2).contiguous()
            ref_patches_x4 = ref_feature[0][ind_b_x4, :, ind_y_x4, ind_x_x4].view(N * py * px, (diameter_y + 2) * 4, (diameter_x + 2) * 4, C).permute(0, 3, 1, 2).contiguous()

            ## calculate correlation between lr patches and their corresponding ref patches
            lr_patches = lr_patches.contiguous().view(N * py * px, C, k_y + 2, k_x + 2)
            corr_all_l, index_all_l = self.search_org(lr_patches, reflr_patches, ks=self.psize, pd=self.psize // 2, stride=1)

            index_all = index_all_l[:, :, :, 0]  # [N*p*p, k_y, k_x]
            soft_att_all = corr_all_l[:, :, :, 0:1].permute(0, 3, 1, 2)  # [N*p*p, 1, k_y, k_x]

            warp_ref_patches_x1 = self.transfer(ref_patches_x1, index_all, soft_att_all, ks=self.psize, pd=self.psize // 2, stride=1)  # [N*py*px, C, k_y, k_x]
            warp_ref_patches_x2 = self.transfer(ref_patches_x2, index_all, soft_att_all, ks=self.psize * 2, pd=self.psize // 2 * 2, stride=2)  # [N*py*px, C, k_y*2, k_x*2]
            warp_ref_patches_x4 = self.transfer(ref_patches_x4, index_all, soft_att_all, ks=self.psize * 4, pd=self.psize // 2 * 4, stride=4)  # [N*py*px, C, k_y*4, k_x*4]

            warp_ref_patches_x1 = warp_ref_patches_x1.view(N, py, px, C, H // py, W // px).permute(0, 3, 1, 4, 2, 5).contiguous()  # [N, C, py, H//py, px, W//px]
            warp_ref_patches_x1 = warp_ref_patches_x1.view(N, C, H, W)

            warp_ref_patches_x2 = warp_ref_patches_x2.view(N, py, px, C, H // py * 2, W // px * 2).permute(0, 3, 1, 4, 2, 5).contiguous()  # [N, C, py, H//py*2, px, W//px*2]
            warp_ref_patches_x2 = warp_ref_patches_x2.view(N, C, H * 2, W * 2)

            warp_ref_patches_x4 = warp_ref_patches_x4.view(N, py, px, C, H // py * 4, W // px * 4).permute(0, 3, 1, 4, 2, 5).contiguous()  # [N, C, py, H//py*4, px, W//px*4]
            warp_ref_patches_x4 = warp_ref_patches_x4.view(N, C, H * 4, W * 4)

            F_M = [warp_ref_patches_x4, warp_ref_patches_x2, warp_ref_patches_x1]
        ##### end #####

        return F_M

    def forward(self, tar, reflr, ref):
        #### tar_lr #####
        #### Conv2D #####
        tar_lr = self.conv2d(tar)
        ###### STG ######
        tar_lr = self.conv_after_RSTB(self.forward_features_RSTB(tar_lr)) + tar_lr

        #### ref_lr #####
        #### Conv2D #####
        ref_lr = self.conv2d(reflr)
        ###### STG ######
        ref_lr = self.conv_after_RSTB(self.forward_features_RSTB(ref_lr)) + ref_lr

        #### Ref Pyramid begin ####
        #### Conv2D & STG#####
        ref_0 = self.conv_first(ref)
        ref_0 = self.conv_after_RSTB(self.forward_features_RSTB(ref_0)) + ref_0
        ref_1 = self.conv_second(ref_0)
        ref_1 = self.conv_after_RSTB(self.forward_features_RSTB(ref_1)) + ref_1
        ref_2 = self.conv_third(ref_1)
        ref_2 = self.conv_after_RSTB(self.forward_features_RSTB(ref_2)) + ref_2
        ref_feature = [ref_0, ref_1, ref_2]
        #### Ref Pyramid end ####

        ## Multi-scale Contextual Matching ##
        F_M = self.contextual_matching(tar_lr, ref_lr, ref_feature, self.upscale)

        ## Multi-Scale Aggregation Block (MAB) ##
        Tar_Rec_SR = self.MAB(tar_lr, F_M, self.upscale)

        return Tar_Rec_SR

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = 64
    width = 64
    model = McMRSR(upscale=upscale, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2)
    num_param = sum([p.numel() for p in model.parameters() if p.requires_grad])

    print(model)
    print('Number of parameters: {}'.format(num_param))
    x = torch.randn((1, 2, height, width))
    x_T1_lr = torch.randn((1, 2, height, width))
    x_T1 = torch.randn((1, 2, height*upscale, width*upscale))
    x = model(x,x_T1_lr,x_T1)
    print(x.shape)

