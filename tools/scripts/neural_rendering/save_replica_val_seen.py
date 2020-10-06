import gzip
import os
import skimage.io

import habitat
import habitat.datasets.pointnav.pointnav_dataset as habitat_dataset
import numpy as np
import quaternion
import tqdm
from habitat.config.default import get_config


def make_config(
    config, gpu_id, split, data_path, sensors, resolution, scenes_dir
):
    config = get_config(config)
    config.defrost()
    config.TASK.NAME = "Nav-v0"
    config.TASK.MEASUREMENTS = []
    config.DATASET.SPLIT = split
    config.DATASET.POINTNAVV1.DATA_PATH = data_path
    config.DATASET.SCENES_DIR = scenes_dir
    config.HEIGHT = resolution
    config.WIDTH = resolution
    for sensor in sensors:
        config.SIMULATOR[sensor]["HEIGHT"] = resolution
        config.SIMULATOR[sensor]["WIDTH"] = resolution

    config.TASK.HEIGHT = resolution
    config.TASK.WIDTH = resolution
    config.SIMULATOR.TURN_ANGLE = 15
    config.SIMULATOR.FORWARD_STEP_SIZE = 0.1  # in metres
    config.SIMULATOR.AGENT_0.SENSORS = sensors
    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False

    config.SIMULATOR.DEPTH_SENSOR.HFOV = 90

    config.ENVIRONMENT.MAX_EPISODE_STEPS = 2 ** 32
    config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id
    return config


def get_pytorch3d_camera_RT(position, rotation):
    rotation = quaternion.as_rotation_matrix(rotation)

    Pinv = np.eye(4, dtype=np.float32)
    Pinv[0:3, 0:3] = rotation
    Pinv[0:3, 3] = position
    P = np.linalg.inv(Pinv)

    # change from Habitat coordinates to PyTorch3D coordinates
    P[0] *= -1  # flip X axis
    P[2] *= -1  # flip Z axis

    R = P[0:3, 0:3].T  # to row major
    T = P[0:3, 3]

    return R, T


def jitter_quaternions(original_quaternion, angle=30.0):
    original_euler = quaternion.as_euler_angles(original_quaternion)
    euler_angles = np.array(
        [
            (np.random.rand() - 0.5) * np.pi * angle / 180.0 + original_euler[0],
            (np.random.rand() - 0.5) * np.pi * angle / 180.0 + original_euler[1],
            (np.random.rand() - 0.5) * np.pi * angle / 180.0 + original_euler[2],
        ]
    )
    quaternions = quaternion.from_euler_angles(euler_angles)

    return quaternions


class RandomImageGenerator(object):
    def __init__(self, split, gpu_id, opts):
        print("gpu_id", gpu_id)
        resolution = opts["W"]
        sensors = ["RGB_SENSOR", "DEPTH_SENSOR"]
        data_path = opts["data_path"]
        unique_dataset_name = opts["dataset"]

        config = make_config(
            opts["config"],
            gpu_id,
            split,
            data_path,
            sensors,
            resolution,
            opts["scenes_dir"],
        )
        data_dir = os.path.join(
            "/private/home/ronghanghu/workspace/synsin/data/scene_episodes/",
            unique_dataset_name + "_" + split
        )
        self.dataset_name = config.DATASET.TYPE
        assert os.path.exists(data_dir)
        data_path = os.path.join(data_dir, "dataset_one_ep_per_scene.json.gz")

        # Load in data and update the location to the proper location (else
        # get a weird, uninformative, error -- Affine2Dtransform())
        dataset = habitat_dataset.PointNavDatasetV1()
        with gzip.open(data_path, "rt") as f:
            dataset.from_json(f.read())

            for i in range(0, len(dataset.episodes)):
                dataset.episodes[i].scene_id = dataset.episodes[i].scene_id.replace(
                    '/checkpoint/erikwijmans/data/mp3d/', opts["scenes_dir"] + '/mp3d/'
                )
                dataset.episodes[i].scene_id = dataset.episodes[i].scene_id.replace(
                    '/checkpoint/ow045820/data/replica/', opts["scenes_dir"] + '/replica/'
                )

        config.TASK.SENSORS = ["POINTGOAL_SENSOR"]
        config.freeze()

        self.env = habitat.Env(config=config, dataset=dataset)
        self.env_sim = self.env.sim
        self.env.seed(50)
        self.env_sim.seed(50)

        self.config = config
        self.opts = opts

    def get_sample(self, num_views_per_sample, max_loc_change, max_angle_change):
        rgbs = []
        depths = []
        camera_Rs = []
        camera_Ts = []
        agent_positions = []
        agent_rotations = []

        orig_location = self.env_sim.sample_navigable_point()
        orig_angle = np.random.uniform(0, 2 * np.pi)
        orig_rotation = [0, np.sin(orig_angle / 2), 0, np.cos(orig_angle / 2)]
        for i in range(0, num_views_per_sample):
            # random change in position
            position = orig_location.copy()
            position[0] = position[0] + (np.random.rand() - 0.5) * max_loc_change

            # random change in location
            rotation = jitter_quaternions(
                quaternion.from_float_array(orig_rotation),
                angle=max_angle_change
            )
            rotation = quaternion.as_float_array(
                rotation
            ).tolist()

            # get observation
            obs = self.env_sim.get_observations_at(
                position=position,
                rotation=rotation,
                keep_agent_at_new_pose=True
            )
            agent_state = self.env_sim.get_agent_state().sensor_states["depth"]
            R, T = get_pytorch3d_camera_RT(agent_state.position, agent_state.rotation)

            depths.append(obs["depth"].copy())
            rgbs.append(obs["rgb"].copy())
            camera_Rs.append(R)
            camera_Ts.append(T)
            agent_positions.append(agent_state.position)
            agent_rotations.append(agent_state.rotation)

        return {
            "rgbs": rgbs,
            "depths": depths,
            "camera_Rs": camera_Rs,
            "camera_Ts": camera_Ts,
            "agent_positions": agent_positions,
            "agent_rotations": agent_rotations,
        }


if __name__ == "__main__":
    IMAGE_SIZE = 256
    GPU_ID = 0
    SPLIT = 'train'

    # generate a different set of images as val_seen on the same scenes as train, with a different random seed
    SAVE_IM_DIR = '/checkpoint/ronghanghu/neural_rendering_datasets/replica/val_seen/images'
    SAVE_DATA_DIR = '/checkpoint/ronghanghu/neural_rendering_datasets/replica/val_seen/data'

    NUM_SAMPLE_PER_SCENE = 2000
    NUM_VIEW_PER_SAMPLE = 5
    MAX_LOC_CHANGE = 0.32
    MAX_ANGLE_CHANGE = 20

    HABITAT_ROOT = '/private/home/ronghanghu/workspace/habitat-api/'
    opts = {
        "W": IMAGE_SIZE,
        "config": os.path.join(HABITAT_ROOT, "configs/tasks/pointnav_rgbd.yaml"),
        "data_path": os.path.join(HABITAT_ROOT, "data/datasets/pointnav/replica/v1/{}/{}.json.gz".format(SPLIT, SPLIT)),
        "dataset": "replica",
        "scenes_dir": os.path.join(HABITAT_ROOT, "data"),
    }

    generator = RandomImageGenerator(
        split=SPLIT,
        gpu_id=GPU_ID,
        opts=opts
    )

    os.makedirs(SAVE_IM_DIR, exist_ok=True)
    os.makedirs(SAVE_DATA_DIR, exist_ok=True)
    num_scenes = len(generator.env.episodes)
    for n_scene in range(num_scenes):
        print('saving scene {} / {}'.format(n_scene + 1, num_scenes))
        generator.env.reset()
        assert generator.env.current_episode.episode_id == n_scene

        for n_sample in tqdm.tqdm(range(NUM_SAMPLE_PER_SCENE)):
            data = generator.get_sample(NUM_VIEW_PER_SAMPLE, MAX_LOC_CHANGE, MAX_ANGLE_CHANGE)

            rgbs = data.pop("rgbs")
            for n_im in range(len(rgbs)):
                im_filename = 'scene_{:04d}_sample_{:08d}_im_{:04d}.png'.format(
                    n_scene, n_sample, n_im)
                skimage.io.imsave(os.path.join(SAVE_IM_DIR, im_filename), rgbs[n_im])

            npy_filename = 'scene_{:04d}_sample_{:08d}.npz'.format(n_scene, n_sample)
            np.savez_compressed(
                os.path.join(SAVE_DATA_DIR, npy_filename),
                depths=np.stack(data["depths"]),
                camera_Rs=np.stack(data["camera_Rs"]),
                camera_Ts=np.stack(data["camera_Ts"]),
                agent_positions=data["agent_positions"],
                agent_rotations=data["agent_rotations"]
            )
