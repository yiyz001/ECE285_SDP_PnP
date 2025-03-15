import os
import pickle
import numpy as np
from tqdm import tqdm
import scipy
from core.manifold_utils import hat_map


def read_poses(npz_file):
    """
    Load SE(3) poses from npz file

    :param npz_file: path to npz file
    :return: 3d poses belongs to SE(3)
    """
    arr = np.load(npz_file)
    num_robots = len(arr)
    poses = []
    for i in range(num_robots):
        poses.append(arr['arr_{}'.format(i)])
    return poses


def load_data():
    """
    Load data from KITTI dataset

    :return: camera observations, landmark positions in 3d, camera calibration matrix
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = current_dir + "/data/kitti/00/"

    # load object feature measurements
    obj_feat_file = os.path.join(data_path, 'obj_features_messages')
    with open(obj_feat_file, 'rb') as f:
        obj_feat_messages = pickle.load(f)

    # load gt landmarks
    gt_obj_file = os.path.join(data_path, 'obj_features_gt')
    with open(gt_obj_file, 'rb') as f:
        gt_objects = pickle.load(f)

    # load calibration matrix
    calib_file = os.path.join(data_path, 'calib')
    with open(calib_file, 'rb') as f:
        calib = pickle.load(f)

    return obj_feat_messages, gt_objects, calib


def select_data(features, landmarks, K):
    """
    The PnP algorithm requires at least 4 points to work, however, when we only have 4 points, Omega matrix
    might be singular, thus we select frame with at least 5 points.

    :param features: 2d camera observations
    :param landmarks: 3d world frame landmarks
    :param K: 3d camera calibration matrix

    :return: selected homogeneous camera observations, selected world frame landmarks, selected frame index
    """
    inv_K = np.linalg.inv(K)

    selected_frames = []
    selected_observations = []
    selected_landmarks = []

    pbar = tqdm(total=len(features))
    for i in range(len(features)):
        if len(features[i]) > 4:
            np_obsv = np.ndarray((len(features[i]), 3, 1))
            np_lmark = np.ndarray((len(features[i]), 3, 1))
            j = 0
            selected_frames.append(i)
            for key, value in features[i].items():
                np_obsv[j] = (inv_K @ np.array((value[0], value[1], 1))).reshape(3, 1)
                np_lmark[j] = landmarks[key].reshape(3, 1)
                j += 1

            selected_observations.append(np_obsv)
            selected_landmarks.append(np_lmark)

        pbar.update(1)
    pbar.close()

    return selected_observations, selected_landmarks, selected_frames


def construct_data_matrix(features, landmarks):
    """
    Construct data matrix P and Omega

    :param features: homogeneous camera observations
    :param landmarks: 3d world frame landmark positions

    :return: data matrix P, data matrix Omega
    """
    total_frames = len(features)
    e_3 = np.zeros((3, 1))
    e_3[2] = 1

    # Data container
    P = []
    Omega = []

    # pbar = tqdm(total=total_frames)
    for i in range(total_frames):
        # Observations and landmarks
        u = features[i]
        m = landmarks[i]

        # Construct Q matrix
        sqrt_Q = u @ e_3.T[None, :, :] - np.eye(3)[None, :, :]
        Q_i = sqrt_Q.transpose(0, 2, 1) @ sqrt_Q

        # Construct A matrix
        A_i = np.kron(np.eye(3)[None, :, :], m.transpose(0, 2, 1))

        # Construct P matrix
        QA_i = Q_i @ A_i
        P_i = -1.0 * np.linalg.inv(np.sum(Q_i, axis=0)) @ np.sum(QA_i, axis=0)
        P.append(P_i)

        # Construct Omega matrix
        ApP_i = A_i + P_i[None, :, :]
        Omega.append(np.sum(ApP_i.transpose(0, 2, 1) @ Q_i @ ApP_i, axis=0))

        # pbar.update(1)
    # pbar.close()

    return P, Omega


def generate_data(num_obsv: int) -> (list, np.ndarray, np.ndarray):
    """
    Generate observation and landmark pairs

    :param num_obsv: number of observations

    :return: ground-truth [rotation.T, translation, camera frame landmark], homogeneous observations, landmarks
    """
    u = np.random.uniform(1, 3) * np.random.randn(num_obsv, 3)
    u[:, 2] = np.abs(u[:, 2])

    theta = np.random.uniform(-np.pi, np.pi, 3)
    Rt = scipy.linalg.expm(hat_map(theta[None, :])).squeeze()
    p = np.random.randn(3)

    m = np.sum(Rt.T[None, :, :] * (u - p[None, :])[:, None, :], axis=-1)

    return [Rt, p, u], u / u[:, 2][:, None], m


def generate_noisy_data(num_obsv: int, noise_level: float) -> (list, np.ndarray, np.ndarray):
    """
    Generate observation and landmark pairs

    :param num_obsv: number of observations
    :param noise_level: covariance for the Gaussian noise

    :return: ground-truth [rotation.T, translation, camera frame landmark], homogeneous observations, landmarks
    """
    u = np.random.uniform(1, 3) * np.random.randn(num_obsv, 3)
    u[:, 2] = np.abs(u[:, 2])
    u_w_noise = u.copy()
    u_w_noise[:, :2] = u_w_noise[:, :2] + np.sqrt(noise_level) * np.random.randn(num_obsv, 2)

    theta = np.random.uniform(-np.pi, np.pi, 3)
    Rt = scipy.linalg.expm(hat_map(theta[None, :])).squeeze()
    p = np.random.randn(3)

    m = np.sum(Rt.T[None, :, :] * (u - p[None, :])[:, None, :], axis=-1)

    return [Rt, p, u], u_w_noise / u[:, 2][:, None], m


def kitti_data():
    """
    Load data from KITTI dataset

    :return: homogeneous observations, world frame landmarks
    """
    # Load raw data from KITTI
    homo_optical_z, lmark_m, calib_stereo = load_data()
    calib_mat = calib_stereo["K_mono"]

    # Obtain paired data
    obsv, lmark, _ = select_data(homo_optical_z, lmark_m, calib_mat)

    return obsv, lmark
