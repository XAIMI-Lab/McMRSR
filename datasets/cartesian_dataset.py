import os
import scipy.io as sio
import torch.utils.data as data
from datasets.utilizes import *
from models.utils import fft2, ifft2, to_tensor

class MRIDataset_Cartesian(data.Dataset):
    def __init__(self, opts, mode):
        self.mode = mode
        if self.mode == 'TRAIN':
            self.data_dir_flair = os.path.join(opts.data_root, 'train')
            self.sample_list = open(os.path.join(opts.list_dir, self.mode + '.txt')).readlines()
            self.seed = None

        if self.mode == 'VALI':
            self.data_dir_flair = os.path.join(opts.data_root, 'vali')
            self.sample_list = open(os.path.join(opts.list_dir, self.mode + '.txt')).readlines()
            self.seed = 1234

        if self.mode == 'TEST':
            self.data_dir_flair = os.path.join(opts.data_root, 'test')
            self.sample_list = open(os.path.join(opts.list_dir, self.mode + '.txt')).readlines()
            self.seed = 5678

        self.data_dir_flair = os.path.join(self.data_dir_flair)
        self.mask_path = opts.mask_path
    def __getitem__(self, idx):

        mask = sio.loadmat(self.mask_path)['lr_mask']
        mask = mask[np.newaxis,:,:]
        mask = np.concatenate([mask, mask], axis=0)
        mask = torch.from_numpy(mask.astype(np.float32))

        slice_name = self.sample_list[idx].strip('\n')
        T2_img = sio.loadmat(os.path.join(self.data_dir_flair, slice_name))
        #=======
        T2_256_img_real = T2_img['T2'].real
        T2_256_img_real = T2_256_img_real[np.newaxis, :, :]
        T2_256_img_imag = T2_img['T2'].imag
        T2_256_img_imag = T2_256_img_imag[np.newaxis, :, :]
        T2_256_img = np.concatenate([T2_256_img_real, T2_256_img_imag], axis=0)  # 2,w,h
        T2_256_img = to_tensor(T2_256_img).float()
        # =======
        T2_256_img_k = T2_256_img.permute(1, 2, 0)
        T2_256_img_k_ks = fft2(T2_256_img_k)
        T2_256_img_ks = T2_256_img_k_ks.permute(2, 0, 1)
        #=======
        T2_128_img_real = T2_img['T2_128'].real#ZF
        T2_128_img_real = T2_128_img_real[np.newaxis,:, :]
        T2_128_img_imag = T2_img['T2_128'].imag
        T2_128_img_imag = T2_128_img_imag[np.newaxis,:, :]
        T2_128_img = np.concatenate([T2_128_img_real, T2_128_img_imag], axis=0)#2,w,h
        T2_128_img = to_tensor(T2_128_img).float()
        #=======
        T2_64_real = T2_img['T2_64'].real
        T2_64_real = T2_64_real[np.newaxis, :, :]
        T2_64_imag = T2_img['T2_64'].imag
        T2_64_imag = T2_64_imag[np.newaxis, :, :]
        T2_64_img = np.concatenate([T2_64_real, T2_64_imag], axis=0)
        T2_64_img = to_tensor(T2_64_img).float()



        T1_256_real = T2_img['T1'].real
        T1_256_real = T1_256_real[np.newaxis, :, :]
        T1_256_imag = T2_img['T1'].imag
        T1_256_imag = T1_256_imag[np.newaxis, :, :]
        T1_256_img = np.concatenate([T1_256_real, T1_256_imag], axis=0)
        T1_256_img = to_tensor(T1_256_img).float()
        # =======T1 128
        T1_128_real = T2_img['T1_128'].real
        T1_128_real = T1_128_real[np.newaxis, :, :]
        T1_128_imag = T2_img['T1_128'].imag
        T1_128_imag = T1_128_imag[np.newaxis, :, :]
        T1_128_img = np.concatenate([T1_128_real, T1_128_imag], axis=0)
        T1_128_img = to_tensor(T1_128_img).float()
        # =======T1 64
        T1_64_real = T2_img['T1_64'].real
        T1_64_real = T1_64_real[np.newaxis, :, :]
        T1_64_imag = T2_img['T1_64'].imag
        T1_64_imag = T1_64_imag[np.newaxis, :, :]
        T1_64_img = np.concatenate([T1_64_real, T1_64_imag], axis=0)
        T1_64_img = to_tensor(T1_64_img).float()
        #=======

# ---------------------over------
        return {'ref_image_full': T1_256_img,
                'ref_image_sub': T1_64_img, #  if 2x, change it to T1_128_img
                'tag_image_full': T2_256_img,
                'tag_kspace_full':T2_256_img_ks,
                'tag_image_sub': T2_64_img, # if 2x, change it to T2_128_img
                'tag_kspace_mask2d': mask  #  dc dc_mask
                }

    def __len__(self):
        return len(self.sample_list)

if __name__ == '__main__':
    a = 1
