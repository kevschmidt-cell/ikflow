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

import numpy as np
from multiprocessing import Pool, cpu_count
from jrl.utils import posevec_to_T, T_to_posevec
from concurrent.futures import ThreadPoolExecutor, as_completed

def sample_dual_joint_angles_and_object_pose_parallel(
    left_robot, right_robot, n: int, T_left_offset, T_right_offset,
    only_non_self_colliding=True, n_threads=8
):
    # linker Arm
    q_left, poses_left = left_robot.sample_joint_angles_and_poses(
        n,
        only_non_self_colliding=only_non_self_colliding,
        tqdm_enabled=True,
        return_torch=False,
    )

    q_right = np.zeros((n, right_robot.ndof))
    poses_obj = np.zeros((n, 7))

    def compute_right_ik(i, T_left_pose):
        T_left = posevec_to_T(T_left_pose)
        T_obj = T_left @ np.linalg.inv(T_left_offset)
        T_right_desired = T_obj @ T_right_offset

        ik_result = right_robot.inverse_kinematics_klampt(T_right_desired)
        if ik_result is None:
            return None
        ok, qR = ik_result
        if not ok:
            return None
        return i, qR, T_to_posevec(T_obj)

    # Threads starten
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(compute_right_ik, i, poses_left[i]): i for i in range(n)}

        for future in as_completed(futures):
            res = future.result()
            if res is None:
                continue
            i, qR, pose_obj = res
            q_right[i] = qR
            poses_obj[i] = pose_obj

    return q_left, q_right, poses_obj




# ---------------------------------------------------
# Dataset speichern
# ---------------------------------------------------
# ---------------------------------------------------
# Dataset speichern (mit parallelem Sampling)
# ---------------------------------------------------
def save_dual_dataset_parallel(
    left_robot,
    right_robot,
    dataset_directory,
    training_set_size,
    test_set_size,
    T_left_offset=np.eye(4),
    T_right_offset=np.eye(4),
    batch_size=1000,
):
    safe_mkdir(dataset_directory)

    print(f"Sampling training set ({training_set_size} Samples)...")
    qL_tr, qR_tr, poses_tr = sample_dual_joint_angles_and_object_pose_parallel(
        left_robot,
        right_robot,
        training_set_size,
        T_left_offset,
        T_right_offset,
        only_non_self_colliding=args.only_non_self_colliding,
        n_threads=8,  # optional
    )

    print(f"Sampling test set ({test_set_size} Samples)...")
    qL_te, qR_te, poses_te = sample_dual_joint_angles_and_object_pose_parallel(
        left_robot, right_robot, test_set_size, T_left_offset, T_right_offset, batch_size=batch_size
    )

    # Beide Arme zusammenfügen
    q_tr = np.hstack([qL_tr, qR_tr])
    q_te = np.hstack([qL_te, qR_te])

    # Speichern
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
        f.write(f"Training set size: {len(q_tr)}\n")
        f.write(f"Test set size: {len(q_te)}\n")
        print_tensor_stats(torch.tensor(q_tr), writable=f, name="q_tr")

    print("Dataset erfolgreich gespeichert!")

# ---------------------------------------------------
# Main
# ---------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_set_size", type=int, default=int(1e5))
    parser.add_argument("--test_set_size", type=int, default=TEST_SET_SIZE)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--urdf_path", type=str, required=True,
                        help="Pfad zu deinem Objekt-URDF mit obj_p_01/obj_p_02")
    args = parser.parse_args()

    # Dual-Roboter Instanz
    dual_robot = DualIiwa7()
    left_robot = dual_robot.left_robot
    right_robot = dual_robot.right_robot

    # Offsets aus URDF laden
    offsets = get_offsets_from_urdf(args.urdf_path)
    print("Offsets geladen:", list(offsets.keys()))

    T_left_offset = offsets["obj_p_01"]["offset"]
    T_right_offset = offsets["obj_p_02"]["offset"]

    # Dataset-Verzeichnis erstellen
    dset_directory = get_dataset_directory("dual_iiwa7")
    print(f"Speichere unter: {dset_directory}")

    # Sampling starten
    t0 = time()
    save_dual_dataset_parallel(
        left_robot,
        right_robot,
        dset_directory,
        training_set_size=args.training_set_size,
        test_set_size=args.test_set_size,
        T_left_offset=T_left_offset,
        T_right_offset=T_right_offset,
        batch_size=args.batch_size,
    )
    print(f"Fertig in {time() - t0:.2f}s")

