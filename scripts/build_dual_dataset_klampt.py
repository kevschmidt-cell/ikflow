#!/usr/bin/env python3
import argparse
import os
from time import time
import numpy as np
import torch
from tqdm import trange
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET

from ikflow.config import DATASET_DIR
from ikflow.utils import get_dataset_directory, safe_mkdir, print_tensor_stats
from jrl.utils import T_to_posevec, posevec_to_T

# Deine Wrapper-Classes importieren
from jrl.robots import DualIiwa7


TEST_SET_SIZE = 15000


# ---------------------------------------------------
# Hilfsfunktion: Offsets aus URDF laden
# ---------------------------------------------------
def get_offsets_from_urdf(urdf_path, parent="obj_com", children=("obj_p_01", "obj_p_02")):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    offsets = {}
    for joint in root.findall("joint"):
        if joint.get("type") != "fixed":
            continue

        parent_link = joint.find("parent").get("link")
        child_link = joint.find("child").get("link")

        if parent_link != parent or child_link not in children:
            continue

        origin = joint.find("origin")
        xyz = [float(v) for v in origin.get("xyz").split()]
        rpy = [float(v) for v in origin.get("rpy").split()]

        T = np.eye(4)
        T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
        T[:3, 3] = xyz
        offsets[child_link] = {
            "offset": T,
            "offset_inv": np.linalg.inv(T),
        }

    if not offsets:
        raise RuntimeError(f"Keine Offsets für {children} in {urdf_path} gefunden!")
    return offsets


# ---------------------------------------------------
# Sampling
# ---------------------------------------------------
from tqdm import tqdm

def sample_dual_joint_angles_and_object_pose(
    left_robot, right_robot, n: int, T_left_offset, T_right_offset,
    only_non_self_colliding=True,
    batch_size=1000
):
    """
    Samplet n Joint-Konfigurationen für linken Arm + rechte IK
    + dazugehörige Objekt-Pose. Optimiert für Geschwindigkeit.
    """
    q_left_all = []
    q_right_all = []
    poses_obj_all = []

    pbar = tqdm(total=n, desc="Valid dual-arm samples", unit="cfg")

    while len(q_left_all) < n:
        # linker Arm: Batch sampeln
        q_left_batch, poses_left_batch = left_robot.sample_joint_angles_and_poses(
            batch_size,
            only_non_self_colliding=only_non_self_colliding,
            tqdm_enabled=False,
            return_torch=False,
        )

        for qL, poseL in zip(q_left_batch, poses_left_batch):
            T_left = posevec_to_T(poseL)
            T_obj = T_left @ np.linalg.inv(T_left_offset)
            T_right_desired = T_obj @ T_right_offset
            pose_right = np.array(T_to_posevec(T_right_desired))

            ik_result = right_robot.inverse_kinematics_klampt(pose_right)
            if ik_result is None:
                continue  # IK failed → skip

            q_left_all.append(qL)
            q_right_all.append(ik_result)
            poses_obj_all.append(T_to_posevec(T_obj))
            pbar.update(1)

            if len(q_left_all) >= n:
                break

    pbar.close()
    return np.array(q_left_all), np.array(q_right_all), np.array(poses_obj_all)


# ---------------------------------------------------
# Dataset speichern
# ---------------------------------------------------
def save_dual_dataset(
    left_robot,
    right_robot,
    dataset_directory,
    training_set_size,
    test_set_size,
    T_left_offset=np.eye(4),
    T_right_offset=np.eye(4),
    only_non_self_colliding=False,
):
    safe_mkdir(dataset_directory)

    qL_tr, qR_tr, poses_tr = sample_dual_joint_angles_and_object_pose(
        left_robot, right_robot, training_set_size, T_left_offset, T_right_offset,
        only_non_self_colliding=only_non_self_colliding,
    )
    qL_te, qR_te, poses_te = sample_dual_joint_angles_and_object_pose(
        left_robot, right_robot, test_set_size, T_left_offset, T_right_offset,
        only_non_self_colliding=only_non_self_colliding,
    )

    q_tr = np.hstack([qL_tr, qR_tr])
    q_te = np.hstack([qL_te, qR_te])

    torch.save(
        {"q": torch.tensor(q_tr, dtype=torch.float32),
         "pose": torch.tensor(poses_tr, dtype=torch.float32)},
        os.path.join(dataset_directory, "train.pt"),
    )
    torch.save(
        {"q": torch.tensor(q_te, dtype=torch.float32),
         "pose": torch.tensor(poses_te, dtype=torch.float32)},
        os.path.join(dataset_directory, "test.pt"),
    )

    with open(os.path.join(dataset_directory, "info.txt"), "w") as f:
        f.write(f"Dual dataset for {left_robot.name} + {right_robot.name}\n")
        f.write(f"Requested training set size: {training_set_size}\n")
        f.write(f"Actual training set size: {len(q_tr)}\n")
        f.write(f"Requested test set size: {test_set_size}\n")
        f.write(f"Actual test set size: {len(q_te)}\n")
        print_tensor_stats(torch.tensor(q_tr), writable=f, name="q_tr")


# ---------------------------------------------------
# Main
# ---------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_set_size", type=int, default=int(1e5))
    parser.add_argument("--test_set_size", type=int, default=15000)  # <-- hier
    parser.add_argument("--batch_size", type=int, default=5000)       # <-- hier
    parser.add_argument("--only_non_self_colliding", action="store_true")
    parser.add_argument("--urdf_path", type=str, required=True,
                        help="Pfad zu deinem Objekt-URDF mit obj_p_01/obj_p_02")
    args = parser.parse_args()


    # Roboter aus deiner Wrapper-Klasse
    dual_robot = DualIiwa7()
    left_robot = dual_robot.left_robot
    right_robot = dual_robot.right_robot

    # Offsets laden
    offsets = get_offsets_from_urdf(args.urdf_path)
    print("Offsets geladen:", list(offsets.keys()))

    T_left_offset = offsets["obj_p_01"]["offset"]
    T_right_offset = offsets["obj_p_02"]["offset"]

    dset_directory = get_dataset_directory("dual_iiwa7")
    print(f"Speichere unter: {dset_directory}")

    t0 = time()
    save_dual_dataset(
        left_robot,
        right_robot,
        dset_directory,
        args.training_set_size,
        TEST_SET_SIZE,
        T_left_offset,
        T_right_offset,
        only_non_self_colliding=args.only_non_self_colliding,
    )
    print(f"Fertig in {time() - t0:.2f}s")

