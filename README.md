# Worldsheet: Wrapping the World in a 3D Sheet for View Synthesis from a Single Image

This repository contains the code for the following paper:

* R. Hu, N. Ravi, A. Berg, D. Pathak, *Worldsheet: Wrapping the World in a 3D Sheet for View Synthesis from a Single Image*. in ICCV, 2021. ([PDF](https://arxiv.org/pdf/2012.09854.pdf))
```
@inproceedings{hu2021worldsheet,
  title={Worldsheet: Wrapping the World in a 3D Sheet for View Synthesis from a Single Image},
  author={Hu, Ronghang and Ravi, Nikhila and Berg, Alex and Pathak, Deepak},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

Project Page: https://worldsheet.github.io/

## Installation

Our Worldsheet implementation is based on [MMF](https://mmf.sh/) and [PyTorch3D](https://pytorch3d.org/). This repository is adapted from the MMF repository (https://github.com/facebookresearch/mmf).

This code is designed to be run on GPU, CPU training/inference is not supported. 

1. Create a new conda environment:
```
conda create -n worldsheet python=3.8
conda activate worldsheet
``` 

2. Download this repository or clone with Git, and then enter the root directory of the repository
`git clone https://github.com/facebookresearch/worldsheet.git && cd worldsheet`

3. Install the MMF dependencies: `pip install -r requirements.txt`

4. Install PyTorch3D as follows (we used v0.2.5):

```
# Install using conda
conda install -c pytorch3d pytorch3d=0.2.5

# Or install from GitHub directly
git clone https://github.com/facebookresearch/pytorch3d.git && cd pytorch3d
git checkout v0.2.5
rm -rf build/ **/*.so
FORCE_CUDA=1 pip install -e .
cd ..

# or pip install from github 
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.2.5"
```

5. Extra dependencies

```
pip install scikit-image
```

## Train and evaluate on the Matterport3D and Replica datasets

In this work, we use the same Matterport3D and Replica datasets as in [SynSin](https://github.com/facebookresearch/synsin), based on the [Habitat](https://aihabitat.org/) environment. **In our codebase and config files, these two datasets are referred to as `synsin_habitat` (`synsin_mp3d` and `synsin_replica`)** (note that here the `synsin_` prefix only refers to the datasets used in SynSin; the underlying model being trained and evaluated is our Worldsheet model, not SynSin).

### Extract the image frames

In our project, we extract those training, validation, and test image frames and camera matrices using the [SynSin](https://github.com/facebookresearch/synsin) codebase for direct comparisons with SynSin and other previous work.

Please install our modified SynSin codebase from `synsin_for_data_and_eval` branch of this repository to extract the Matterport3D and Replica image frames:
```
git clone https://github.com/facebookresearch/worldsheet.git -b synsin_for_data_and_eval synsin && cd synsin
```
and install `habitat-sim` and `habitat-api` as additional SynSin dependencies following the official [SynSin installation instructions](https://github.com/ronghanghu/synsin/blob/master/INSTALL.md). For convenience, we provide the corresponding versions of `habitat-sim` and `habitat-api` for SynSin in `habitat-sim-for-synsin` and `habitat-sim-for-synsin` branches of this repository.

After installing the SynSin codebase from `synsin_for_data_and_eval` branch, set up Matterport3D and Replica datasets following the [instructions](https://github.com/ronghanghu/synsin/blob/master/MP3D.md) in the SynSin codebase, and run the following to save the image frames to disk (you can change `MP3D_SAVE_IMAGE_DIR` to a location on your machine).
```
# this is where Matterport3D and Replica image frames will be extracted
export MP3D_SAVE_IMAGE_DIR=/checkpoint/ronghanghu/neural_rendering_datasets

# clone the SynSin repo from `synsin_for_data_and_eval` branch
git clone https://github.com/facebookresearch/worldsheet.git -b synsin_for_data_and_eval ../synsin
cd ../synsin

# Matterport3D train
DEBUG="" python evaluation/dump_train_to_mmf.py \
     --result_folder ${MP3D_SAVE_IMAGE_DIR}/synsin_mp3d/train \
     --old_model modelcheckpoints/mp3d/synsin.pth \
     --batch_size 8 --num_workers 10  --images_before_reset 1000
# Matterport3D val
DEBUG="" python evaluation/dump_val_to_mmf.py \
     --result_folder ${MP3D_SAVE_IMAGE_DIR}/synsin_mp3d/val \
     --old_model modelcheckpoints/mp3d/synsin.pth \
     --batch_size 8 --num_workers 10  --images_before_reset 200
# Matterport3D test
DEBUG="" python evaluation/dump_test_to_mmf.py \
     --result_folder ${MP3D_SAVE_IMAGE_DIR}/synsin_mp3d/test \
     --old_model modelcheckpoints/mp3d/synsin.pth \
     --batch_size 8 --num_workers 10  --images_before_reset 200
# Replica test
DEBUG="" python evaluation/dump_test_to_mmf.py \
     --result_folder ${MP3D_SAVE_IMAGE_DIR}/synsin_replica/test \
     --old_model modelcheckpoints/mp3d/synsin.pth \
     --batch_size 8 --num_workers 10  --images_before_reset 200 \
     --dataset replica
# Matterport3D val with 20-degree angle change
DEBUG="" python evaluation/dump_val_to_mmf.py \
     --result_folder ${MP3D_SAVE_IMAGE_DIR}/synsin_mp3d/val_jitter_angle20 \
     --old_model modelcheckpoints/mp3d/synsin.pth \
     --batch_size 8 --num_workers 10 --images_before_reset 200 \
     --render_ids 0 --jitter_quaternions_angle 20
# Matterport3D test with 20-degree angle change
DEBUG="" python evaluation/dump_test_to_mmf.py \
     --result_folder ${MP3D_SAVE_IMAGE_DIR}/synsin_mp3d/test_jitter_angle20 \
     --old_model modelcheckpoints/mp3d/synsin.pth \
     --batch_size 8 --num_workers 10 --images_before_reset 200 \
     --render_ids 0 --jitter_quaternions_angle 20

cd ../worldsheet  # assuming `synsin` repo and `worldsheet` repo are under the same parent directory
```

### Training

Run the following to perform training and evaluation. In our experiments, we use a single machine with 4 NVIDIA V100-32GB GPUs.
```
# set to the same path as in image frame extraction above
export MP3D_SAVE_IMAGE_DIR=/checkpoint/ronghanghu/neural_rendering_datasets

# train the scene mesh prediction in Worldsheet
./run_mp3d_and_replica/train_mp3d.sh mp3d_nodepth_perceptual_l1laplacian

# train the inpainter with frozen scene mesh prediction
./run_mp3d_and_replica/train_mp3d.sh mp3d_nodepth_perceptual_l1laplacian_inpaintGonly_freezemesh
```

### Pretrained models

Instead of performing the training above, one can also directly download the pretrained models via
```
./run_mp3d_and_replica/download_pretrained_models.sh
```
and run the evaluation below.

### Evaluation

The evaluation scripts below will print the performance (PSNR, SSIM, Perc-Sim) on different test data.

Evaluate on the default test sets with the same camera changes as the training data (Table 1):
```
# set to the same path as in image frame extraction above
export MP3D_SAVE_IMAGE_DIR=/checkpoint/ronghanghu/neural_rendering_datasets

# Matterport3D, without inpainter (Table 1 line 6)
./run_mp3d_and_replica/eval_mp3d_test_iter.sh mp3d_nodepth_perceptual_l1laplacian 40000

# Matterport3D, full model (Table 1 line 7)
./run_mp3d_and_replica/eval_mp3d_test_iter.sh mp3d_nodepth_perceptual_l1laplacian_inpaintGonly_freezemesh 40000

# Replica, full model (Table 1 line 7)
./run_mp3d_and_replica/eval_replica_test_iter.sh mp3d_nodepth_perceptual_l1laplacian_inpaintGonly_freezemesh 40000
```

Evaluate the full model on 2X camera changes (Table 2):
```
# set to the same path as in image frame extraction above
export MP3D_SAVE_IMAGE_DIR=/checkpoint/ronghanghu/neural_rendering_datasets

# Matterport3D, without inpainter (Table 2 line 4)
./run_mp3d_and_replica/eval_mp3d_test_jitter_angle20_iter.sh mp3d_nodepth_perceptual_l1laplacian 40000

# Matterport3D, full model (Table 2 line 5)
./run_mp3d_and_replica/eval_mp3d_test_jitter_angle20_iter.sh mp3d_nodepth_perceptual_l1laplacian_inpaintGonly_freezemesh 40000

# Replica, full model (Table 2 line 5)
./run_mp3d_and_replica/eval_replica_test_jitter_angle20_iter.sh mp3d_nodepth_perceptual_l1laplacian_inpaintGonly_freezemesh 40000
```

### Visualization

One can visualize the model predictions using the script `run_mp3d_and_replica/visualize_mp3d_val_iter.sh` to visualize the Matterport3D validation set (and this script can be modified to visualize other splits). For example, run the following to visualize the predictions from the full model:
```
export MP3D_SAVE_IMAGE_DIR=/checkpoint/ronghanghu/neural_rendering_datasets

./run_mp3d_and_replica/visualize_mp3d_val_iter.sh mp3d_nodepth_perceptual_l1laplacian_inpaintGonly_freezemesh 40000
```
Then, you can inspect the predictions using the notebook `run_mp3d_and_replica/visualize_predictions.ipynb`.

## Train and evaluate on the RealEstate10K dataset

In this work, we use the same [RealEstate10K](https://google.github.io/realestate10k/) dataset as in [SynSin](https://github.com/facebookresearch/synsin).

### Setting up the RealEstate10K dataset

Please set up the dataset following the [instructions](https://github.com/facebookresearch/synsin/blob/master/REALESTATE.md) in SynSin. The scripts below assumes this dataset has been downloaded to `/checkpoint/ronghanghu/neural_rendering_datasets/realestate10K/RealEstate10K/frames/`. You can modify its path in `mmf/configs/datasets/synsin_realestate10k/defaults.yaml`.

### Training

Run the following to perform the training and evaluation. In our experiments, we use a single machine with 4 NVIDIA V100-32GB GPUs.

```
# train 33x33 mesh
./run_realestate10k/train.sh realestate10k_dscale2_lowerL1_200

# initialize 65x65 mesh from trained 33x33 mesh
python ./run_realestate10k/init_65x65_from_33x33.py \
    --input ./save/synsin_realestate10k/realestate10k_dscale2_lowerL1_200/models/model_50000.ckpt \
    --output ./save/synsin_realestate10k/realestate10k_dscale2_stride4ft_lowerL1_200/init.ckpt

# train 65x65 mesh
./run_realestate10k/train.sh realestate10k_dscale2_stride4ft_lowerL1_200
```

### Pretrained models

Instead of performing the training above, one can also directly download the pretrained models via
```
./run_realestate10k/download_pretrained_models.sh
```
and run the evaluation below.

### Evaluation

Note: as mentioned in the paper, following the evaluation protocol of SynSin on RealEstate10K, the best metrics of two separate predictions based on each view were reported for single-view methods. We follow this evaluation protocol for consistency with SynSin on RealEstate10K in Table 3. We also report averaged metrics over all predictions in the supplemental.

The script below evaluates the performance on RealEstate10K with **averaged metrics** over all predictions, as reported in the supplemental Table C.1:
```
# Evaluate 33x33 mesh (Supplemental Table C.1 line 6)
./run_realestate10k/eval_test_iter.sh realestate10k_dscale2_lowerL1_200 50000

# Evaluate 65x65 mesh (Supplemental Table C.1 line 7)
./run_realestate10k/eval_test_iter.sh realestate10k_dscale2_stride4ft_lowerL1_200 50000
```

To evaluate with the SynSin protocol using the **best metrics of two separate predictions** as in Table 3, one needs to first save the predicted novel views as PNG files, and then use the SynSin codebase for evaluation. Please install our modified SynSin codebase from `synsin_for_data_and_eval` branch of this repository following the Matterport3D and Replica instructions above. Then evaluate as follows:
```
# Save prediction PNGs for 33x33 mesh
./run_realestate10k/write_pred_pngs_test_iter.sh realestate10k_dscale2_lowerL1_200 50000

# Save prediction PNGs for 65x65 mesh
./run_realestate10k/write_pred_pngs_test_iter.sh realestate10k_dscale2_stride4ft_lowerL1_200 50000

cd ../synsin  # assuming `synsin` repo and `worldsheet` repo under the same directory

# Evaluate 33x33 mesh (Table 3 line 9)
python evaluation/evaluate_realestate10k_all.py \
    --take_every_other \
    --folder ../worldsheet/save/prediction_synsin_realestate10k/realestate10k_dscale2_lowerL1_200/50000/realestate10k_test

# Evaluate 65x65 mesh (Table 3 line 10)
python evaluation/evaluate_realestate10k_all.py \
    --take_every_other \
    --folder ../worldsheet/save/prediction_synsin_realestate10k/realestate10k_dscale2_stride4ft_lowerL1_200/50000/realestate10k_test
```
(The `--take_every_other` flag above performs best-of-two-prediction evaluation; without this flag, it should give the average-over-all-prediction results as in Supplemental Table C.1.)

### Visualization

One can visualize the model's predictions using the script `run_realestate10k/eval_val_iter.sh` for the RealEstate10K validation set (`run_realestate10k/visualize_test_iter.sh` for the test set). For example, run the following to visualize the predictions from the 65x65 mesh:
```
./run_realestate10k/visualize_val_iter.sh realestate10k_dscale2_stride4ft_lowerL1_200 50000
```
Then, you can inspect the predictions using notebook `run_realestate10k/visualize_predictions.ipynb`.

We also provide a notebook for interactive predictions in `run_realestate10k/make_interactive_videos.ipynb`, where one can walk through the scene and generate a continuous video of the predicted novel views.

## The structure of Worldsheet codebase

Worldsheet is implemented as a [MMF](https://mmf.sh/) model. This codebase largely follows the structure of typical MMF models and datasets.

The Worldsheet model is defined under the MMF model name `mesh_renderer` in the following files:
- model definition: `mmf/models/mesh_renderer.py`
- mesh and rendering utilities, losses, and metrics: `mmf/neural_rendering/`
- config base: `mmf/configs/models/mesh_renderer/defaults.yaml`

The experimental config files for the Matterport and Replica experiments are in the following files:
- Habitat dataset definition: `mmf/datasets/builders/synsin_habitat/`
- Habitat dataset config base: `mmf/configs/datasets/synsin_habitat/defaults.yaml`
- experimental configs: `projects/neural_rendering/configs/synsin_habitat/`

The experimental config files for the RealEstate10K experiments are in the following files:
- RealEstate10K dataset definition: `mmf/datasets/builders/synsin_realestate10k/`
- RealEstate10K dataset config base: `mmf/configs/datasets/synsin_realestate10k/defaults.yaml`
- experimental configs: `projects/neural_rendering/configs/synsin_realestate10k/`

## Acknowledgements

This repository is modified from the [MMF](https://mmf.sh/) library from Facebook AI Research. A large part of the codebase has been modified from the [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) codebase. Our PSNR, SSIM, and Perc-Sim evaluation scripts are modified from the [SynSin](https://github.com/facebookresearch/synsin) codebase and we also use SynSin for image frame extraction on Matterport3D and Replica.
