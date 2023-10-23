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
        self.flow_dir = os.path.join(self.seq_dir, 'raft_exhaustive')
        self.txt_dir = os.path.join(self.seq_dir, 'txtFile')
        img_names = sorted(os.listdir(self.img_dir))
        self.num_imgs = min(self.args.num_imgs, len(img_names))
        self.img_names = img_names[:self.num_imgs]

        h, w, _ = imageio.imread(os.path.join(self.img_dir, img_names[0])).shape
        self.h, self.w = h, w
        max_interval = self.num_imgs - 1 if not max_interval else max_interval
        self.max_interval = mp.Value('i', max_interval)
        self.num_pts = self.args.num_pts
        self.grid = gen_grid_np(self.h, self.w)
        flow_stats = json.load(open(os.path.join(self.seq_dir, 'flow_stats.json')))
        self.sample_weights = get_sample_weights(flow_stats)
        with open(os.path.join(self.txt_dir,'frame_numbers.txt'), 'r') as file:
            lines = file.readlines()
        self.keyframe = [int(line.strip()) for line in lines]

    def __len__(self):
        return self.num_imgs * 100000

    def set_max_interval(self, max_interval):
        self.max_interval.value = min(max_interval, self.num_imgs - 1)

    def increase_max_interval_by(self, increment):
        curr_max_interval = self.max_interval.value
        self.max_interval.value = min(curr_max_interval + increment, self.num_imgs - 1)

    def __getitem__(self, idx):
        cached_flow_pred_dir = os.path.join('out', '{}_{}'.format(self.args.expname, self.seq_name), 'flow')
        cached_flow_pred_files = sorted(glob.glob(os.path.join(cached_flow_pred_dir, '*')))
        flow_error_file = os.path.join(os.path.dirname(cached_flow_pred_dir), 'flow_error.txt')
        if os.path.exists(flow_error_file):
            flow_error = np.loadtxt(flow_error_file)
            id1_sample_weights = flow_error / np.sum(flow_error)
            id1 = np.random.choice(self.num_imgs, p=id1_sample_weights)
        else:
            id1 = idx % self.num_imgs
        img_name1 = self.img_names[id1]

        if id1 in self.keyframe:
            max_interval = min(self.max_interval.value, self.num_imgs - 1)
            img2_candidates = sorted(list(self.sample_weights[img_name1].keys()))
            # img2_candidates = img2_candidates[max(id1 - max_interval, 0):min(id1 + max_interval, self.num_imgs - 1)] #获取id1对应的前后总共40帧

            # # sample more often from i-1 and i+1
            id2s = np.array([self.img_names.index(n) for n in img2_candidates])

            sample_weights = np.array([self.sample_weights[img_name1][i] for i in img2_candidates])
            sample_weights /= np.sum(sample_weights) #这将样本权重归一化为概率分布，以便它们表示概率值，可以用于随机采样。
            sample_weights[np.abs(id2s - id1) <= 1] = 0.5
            sample_weights /= np.sum(sample_weights)

            img_name2 = np.random.choice(img2_candidates, p=sample_weights)
            id2 = self.img_names.index(img_name2)
            frame_interval = abs(id1 - id2)

            # read image, flow and confidence
            img1 = imageio.imread(os.path.join(self.img_dir, img_name1)) / 255.
            img2 = imageio.imread(os.path.join(self.img_dir, img_name2)) / 255.

            flow_file = os.path.join(self.flow_dir, '{}_{}.npy'.format(img_name1, img_name2))
            flow = np.load(flow_file) #flow.shape = [480,853,2]
            mask_file = flow_file.replace('raft_exhaustive', 'raft_masks').replace('.npy', '.png')
            masks = imageio.imread(mask_file) / 255.

            coord1 = self.grid
            coord2 = self.grid + flow

            cycle_consistency_mask = masks[..., 0] > 0
            occlusion_mask = masks[..., 1] > 0

            if frame_interval == 1:
                mask = np.ones_like(cycle_consistency_mask)
            else:
                mask = cycle_consistency_mask | occlusion_mask #被遮挡的区域和循环一致的区域都取（只要有一个为true就是true）

            if mask.sum() == 0:
                invalid = True
                mask = np.ones_like(cycle_consistency_mask)
            else:
                invalid = False

            if len(cached_flow_pred_files) > 0 and self.args.use_error_map:
                cached_flow_pred_file = cached_flow_pred_files[id1] #定义out文件夹中flow存储的路径
                assert img_name1 + '_' in cached_flow_pred_file
                sup_flow_file = os.path.join(self.flow_dir, os.path.basename(cached_flow_pred_file)) #supervise 监督的flow
                pred_flow = np.load(cached_flow_pred_file)
                sup_flow = np.load(sup_flow_file)

                # 总之，这段代码计算两个光流场之间的误差，并使用高斯模糊平滑误差图像，
                # 然后根据 mask 选择特定区域的误差值，以便进一步分析或处理这些误差值。
                error_map = np.linalg.norm(pred_flow - sup_flow, axis=-1)
                error_map = cv2.GaussianBlur(error_map, (5, 5), 0)
                error_selected = error_map[mask]

                prob = error_selected / np.sum(error_selected)
                select_ids_error = np.random.choice(mask.sum(), self.num_pts, replace=(mask.sum() < self.num_pts), p=prob)
                select_ids_random = np.random.choice(mask.sum(), self.num_pts, replace=(mask.sum() < self.num_pts))
                select_ids = np.random.choice(np.concatenate([select_ids_error, select_ids_random]), self.num_pts,
                                            replace=False)
            else:
                if self.args.use_count_map:
                    count_map = imageio.imread(os.path.join(self.seq_dir, 'count_maps', img_name1.replace('.jpg', '.png')))
                    pixel_sample_weight = 1 / np.sqrt(count_map + 1.)
                    pixel_sample_weight = pixel_sample_weight[mask]
                    pixel_sample_weight /= pixel_sample_weight.sum()
                    #从根据权重计算的像素中选择 self.num_pts 个像素的索引
                    select_ids = np.random.choice(mask.sum(), self.num_pts, replace=(mask.sum() < self.num_pts),
                                                p=pixel_sample_weight)
                else:
                    select_ids = np.random.choice(mask.sum(), self.num_pts, replace=(mask.sum() < self.num_pts))

            # pair_weight = np.cos((frame_interval - 1.) / max_interval * np.pi / 2)
            pair_weight = 1.0

            pts1 = torch.from_numpy(coord1[mask][select_ids]).float()
            pts2 = torch.from_numpy(coord2[mask][select_ids]).float()
            pts2_normed = normalize_coords(pts2, self.h, self.w)[None, None]

            covisible_mask = torch.from_numpy(cycle_consistency_mask[mask][select_ids]).float()[..., None]
            weights = torch.ones_like(covisible_mask) * pair_weight
    
            gt_rgb1 = torch.from_numpy(img1[mask][select_ids]).float()
            gt_rgb2 = F.grid_sample(torch.from_numpy(img2).float().permute(2, 0, 1)[None], pts2_normed,
                                    align_corners=True).squeeze().T

            if invalid:
                weights = torch.zeros_like(weights)

            if np.random.choice([0, 1]):
                id1, id2, pts1, pts2, gt_rgb1, gt_rgb2 = id2, id1, pts2, pts1, gt_rgb2, gt_rgb1
                weights[covisible_mask == 0.] = 0

            data = {'ids1': id1,
                    'ids2': id2,
                    'pts1': pts1,  # [n_pts, 2]
                    'pts2': pts2,  # [n_pts, 2]
                    'gt_rgb1': gt_rgb1,  # [n_pts, 3]
                    'gt_rgb2': gt_rgb2,
                    'weights': weights,  # [n_pts, 1]
                    'covisible_mask': covisible_mask,  # [n_pts, 1]
                    'is_keyframe': True
                    }
            return data               
        else:
            max_interval = min(self.max_interval.value, self.num_imgs - 1)
            id2 = id1
            while id2 == id1:
                id2 = random.randint(0, self.num_imgs-1)
            img_name2 = self.img_names[id2]
            frame_interval = abs(id1 - id2)   


            # read image, flow and confidence
            img1 = imageio.imread(os.path.join(self.img_dir, img_name1)) / 255.
            img2 = imageio.imread(os.path.join(self.img_dir, img_name2)) / 255.

            coord1 = self.grid
            coord2 = self.grid

            if len(cached_flow_pred_files) > 0 and self.args.use_error_map:
                cached_flow_pred_file = cached_flow_pred_files[id1] #定义out文件夹中flow存储的路径
                assert img_name1 + '_' in cached_flow_pred_file
                sup_flow_file = os.path.join(self.flow_dir, os.path.basename(cached_flow_pred_file)) #supervise 监督的flow
                pred_flow = np.load(cached_flow_pred_file)
                sup_flow = np.load(sup_flow_file)

                # 总之，这段代码计算两个光流场之间的误差，并使用高斯模糊平滑误差图像，
                # 然后根据 mask 选择特定区域的误差值，以便进一步分析或处理这些误差值。
                error_map = np.linalg.norm(pred_flow - sup_flow, axis=-1)
                error_map = cv2.GaussianBlur(error_map, (5, 5), 0)
                error_selected = error_map[mask]

                prob = error_selected / np.sum(error_selected)
                select_ids_error = np.random.choice(mask.sum(), self.num_pts, replace=(mask.sum() < self.num_pts), p=prob)
                select_ids_random = np.random.choice(mask.sum(), self.num_pts, replace=(mask.sum() < self.num_pts))
                select_ids = np.random.choice(np.concatenate([select_ids_error, select_ids_random]), self.num_pts,
                                            replace=False)
            else:
                mask = np.ones_like(img1[:,:,0]).astype(bool)
                select_ids = np.random.choice(mask.sum(), self.num_pts, replace=(mask.sum() < self.num_pts))

            pts1 = torch.from_numpy(coord1[mask][select_ids]).float()
            pts2 = torch.from_numpy(coord2[mask][select_ids]).float()
            pts2_normed = normalize_coords(pts2, self.h, self.w)[None, None]
    
            gt_rgb1 = torch.from_numpy(img1[mask][select_ids]).float()
            gt_rgb2 = F.grid_sample(torch.from_numpy(img2).float().permute(2, 0, 1)[None], pts2_normed,
                                    align_corners=True).squeeze().T
            # check here
            # pair_weight = np.cos((frame_interval - 1.) / max_interval * np.pi / 2)
            pair_weight = 1.0
            covisible_mask = (torch.from_numpy(mask).unsqueeze(-1))[mask][select_ids].float()
            weights = torch.ones_like(covisible_mask) * pair_weight

            data = {'ids1': id1,
                    'ids2': id2,
                    'pts1': pts1,  # [n_pts, 2]
                    'pts2': pts2,  # [n_pts, 2]
                    'gt_rgb1': gt_rgb1,  # [n_pts, 3]
                    'gt_rgb2': gt_rgb2,
                    'weights': weights,  # [n_pts, 1]
                    'covisible_mask':  covisible_mask,  # [n_pts, 1]
                    'is_keyframe': False
                    }
            return data               