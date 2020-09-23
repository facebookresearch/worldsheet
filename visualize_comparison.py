import matplotlib; matplotlib.use('Agg')  # NoQA
import os

import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_results(results, descrption='', image_name='', is_comparison=False):
    h = plt.figure(figsize=(20, 15 if is_comparison else 10))
    row_num = 3 if is_comparison else 2

    max_depth_0 = np.max(results['depth_0']) * 1.05
    max_depth_1 = np.max(results['depth_1']) * 1.05

    plt.subplot(row_num, 4, 1)
    plt.imshow(results['orig_img_0'])
    plt.title('ground-truth View 0 RGB\n{}'.format(image_name))

    plt.subplot(row_num, 4, 2)
    plt.imshow(results['depth_0'], vmin=0, vmax=max_depth_0)
    plt.title('ground-truth View 0 depth')

    plt.subplot(row_num, 4, 3)
    plt.imshow(results['orig_img_1'])
    plt.title('ground-truth View 1 RGB')

    plt.subplot(row_num, 4, 4)
    plt.imshow(results['depth_1'], vmin=0, vmax=max_depth_1)
    plt.title('ground-truth View 1 depth')

    plt.subplot(row_num, 4, 5)
    rgb_0_rec = results['rgba_0_rec'][..., :3].copy()
    rgb_0_rec = np.clip(rgb_0_rec, 0, 1)
    alpha = (results['rgba_0_rec'][..., 3] > 0.6)
    rgb_0_rec[..., 0] = np.maximum(rgb_0_rec[..., 0], 1 - alpha)
    rgb_0_rec[..., 1] = np.minimum(rgb_0_rec[..., 1], alpha)
    rgb_0_rec[..., 2] = np.minimum(rgb_0_rec[..., 2], alpha)
    plt.imshow(rgb_0_rec)
    plt.title('exp: {}'.format(descrption))

    plt.subplot(row_num, 4, 6)
    plt.imshow(results['depth_0_rec'], vmin=0, vmax=max_depth_0)
    plt.title('predicted View 0 depth')

    plt.subplot(row_num, 4, 7)
    rgb_1_rec = results['rgba_1_rec'][..., :3].copy()
    rgb_1_rec = np.clip(rgb_1_rec, 0, 1)
    plt.imshow(rgb_1_rec)
    plt.title('predicted View 1 RGB')

    plt.subplot(row_num, 4, 8)
    plt.imshow(results['depth_1_rec'], vmin=0, vmax=max_depth_1)
    plt.title('predicted View 1 depth')
    return h


def plot_comparison(results, descrption=''):
    max_depth_0 = np.max(results['depth_0']) * 1.05
    max_depth_1 = np.max(results['depth_1']) * 1.05

    plt.subplot(3, 4, 9)
    rgb_0_rec = results['rgba_0_rec'][..., :3].copy()
    rgb_0_rec = np.clip(rgb_0_rec, 0, 1)
    alpha = (results['rgba_0_rec'][..., 3] > 0.6)
    rgb_0_rec[..., 0] = np.maximum(rgb_0_rec[..., 0], 1 - alpha)
    rgb_0_rec[..., 1] = np.minimum(rgb_0_rec[..., 1], alpha)
    rgb_0_rec[..., 2] = np.minimum(rgb_0_rec[..., 2], alpha)
    plt.imshow(rgb_0_rec)
    plt.title('exp: {}'.format(descrption))

    plt.subplot(3, 4, 10)
    plt.imshow(results['depth_0_rec'], vmin=0, vmax=max_depth_0)
    plt.title('predicted View 0 depth')

    plt.subplot(3, 4, 11)
    rgb_1_rec = results['rgba_1_rec'][..., :3].copy()
    rgb_1_rec = np.clip(rgb_1_rec, 0, 1)
    plt.imshow(rgb_1_rec)
    plt.title('predicted View 1 RGB')

    plt.subplot(3, 4, 12)
    plt.imshow(results['depth_1_rec'], vmin=0, vmax=max_depth_1)
    plt.title('predicted View 1 depth')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--exp1", type=str, required=True)
    parser.add_argument("--exp2", type=str, required=True)
    parser.add_argument(
        "--visualization_root", type=str,
        default='/private/home/ronghanghu/workspace/mmf_nr/save/visualization/'
    )
    args = parser.parse_args()

    split = args.split
    exp_name_1 = args.exp1
    exp_name_2 = args.exp2
    visualization_root = args.visualization_root

    description_exp1 = exp_name_1
    description_exp2 = exp_name_2
    save_dir_exp1 = os.path.join(visualization_root, f'{exp_name_1}/{split}/')
    save_dir_exp2 = os.path.join(visualization_root, f'{exp_name_2}/{split}/')

    save_visualization_dir = os.path.join(
        visualization_root, f'comparison--{exp_name_1}--{exp_name_2}/{split}/'
    )

    result_files_exp1 = sorted(glob(os.path.join(save_dir_exp1, '*.npz')))
    np.random.seed(3)
    np.random.shuffle(result_files_exp1)
    os.makedirs(save_visualization_dir, exist_ok=True)

    for file_exp1 in tqdm(result_files_exp1):
        image_name = os.path.basename(file_exp1).split('.')[0]
        d = np.load(file_exp1)
        results_exp1 = dict(d)
        h = plot_results(
            results_exp1, description_exp1, image_name, is_comparison=True
        )
        d.close()

        file_exp2 = file_exp1.replace(save_dir_exp1, save_dir_exp2)
        d = np.load(file_exp2)
        results_exp2 = dict(d)
        plot_comparison(results_exp2, description_exp2)
        d.close()

        plt.savefig(
            os.path.join(save_visualization_dir, image_name + '.png'),
            bbox_inches='tight'
        )
        plt.close(h)
