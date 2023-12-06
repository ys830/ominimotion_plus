import os
import glob
import json
import imageio
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import multiprocessing as mp
from util import normalize_coords, gen_grid_np
import random


def get_sample_weights(flow_stats):
    sample_weights = {}
    for k in flow_stats.keys():
        sample_weights[k] = {}
        total_num = np.array(list(flow_stats[k].values())).sum()
        for j in flow_stats[k].keys():
            sample_weights[k][j] = 1. * flow_stats[k][j] / total_num
    return sample_weights


class RAFTExhaustiveDataset(Dataset):
    def __init__(self, args, max_interval=None):
        self.args = args
        self.seq_dir = args.data_dir
        self.seq_name = os.path.basename(self.seq_dir.rstrip('/'))
        self.img_dir = os.path.join(self.seq_dir, 'color')
        img_names = sorted(os.listdir(self.img_dir))
        self.num_imgs = min(self.args.num_imgs, len(img_names))
        self.img_names = img_names[:self.num_imgs]

        h, w, _ = imageio.imread(os.path.join(self.img_dir, img_names[0])).shape
        self.h, self.w = h, w
        self.num_pts = self.args.num_pts
        self.grid = gen_grid_np(self.h, self.w)

    def __len__(self):
        return self.num_imgs * 100000

    def __getitem__(self, idx):

        id1 = idx % self.num_imgs
        img_name1 = self.img_names[id1]

        # read image, flow and confidence
        img1 = imageio.imread(os.path.join(self.img_dir, img_name1)) / 255.

        coord1 = self.grid

        mask = np.ones_like(img1[:,:,0]).astype(bool)
        select_ids = np.random.choice(mask.sum(), self.num_pts, replace=(mask.sum() < self.num_pts))

        pts1 = torch.from_numpy(coord1[mask][select_ids]).float()

        gt_rgb1 = torch.from_numpy(img1[mask][select_ids]).float()

        # check here
        # pair_weight = np.cos((frame_interval - 1.) / max_interval * np.pi / 2)
        pair_weight = 1.0
        covisible_mask = (torch.from_numpy(mask).unsqueeze(-1))[mask][select_ids].float()
        weights = torch.ones_like(covisible_mask) * pair_weight

        data = {'ids1': id1,
                'pts1': pts1,  # [n_pts, 2]
                'gt_rgb1': gt_rgb1,  # [n_pts, 3]
                'weights': weights,  # [n_pts, 1]
                'covisible_mask':  covisible_mask,  # [n_pts, 1]
                }
        return data               