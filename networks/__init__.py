import torch.nn as nn
from networks.network_mcmrsr import SwinIR

import pdb


def set_gpu(network, gpu_ids):
    network.to(gpu_ids[0])
    network = nn.DataParallel(network, device_ids=gpu_ids)

    return network

def get_swin(opts):

    if opts.net_G == 'McMRSR':
        network = SwinIR(upscale=opts.upscale, img_size=(opts.height, opts.width),
                   window_size=opts.window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
    num_param = sum([p.numel() for p in network.parameters() if p.requires_grad])
    print('Number of parameters: {}'.format(num_param))
    return set_gpu(network, opts.gpu_ids)
