import torch
import argparse
import os
from jrl.robots import get_robot
import numpy as np

def main(robot_name, dataset_path):
    print(f"Filtering val set for robot: {robot_name}")
    robot = get_robot(robot_name)

    samples_te_path = os.path.join(dataset_path, "samples_tr.pt__tag0=non-self-colliding")
    poses_te_path = os.path.join(dataset_path, "endpoints_tr.pt__tag0=non-self-colliding")

    samples_te = torch.load(samples_te_path, map_location='cpu')
    poses_te = torch.load(poses_te_path, map_location='cpu')

    assert samples_te.shape[0] == poses_te.shape[0], "Mismatch in number of samples and poses"

    filtered_qs = []
    filtered_poses = []

    for q, pose in zip(samples_te, poses_te):
        if not robot.config_self_collides(q):
            filtered_qs.append(q)
            filtered_poses.append(pose)

    samples_te_filtered = torch.stack(filtered_qs)
    poses_te_filtered = torch.stack(filtered_poses)

    # Überschreiben der Dateien
    torch.save(samples_te_filtered, samples_te_path)
    torch.save(poses_te_filtered, poses_te_path)


    print("Filtered val set saved as:")
    print(" →", os.path.join(dataset_path, "samples_te_filtered.pt"))
    print(" →", os.path.join(dataset_path, "poses_te_filtered.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_name", required=True, type=str, help="e.g. iiwa7_L")
    parser.add_argument("--dataset_path", required=True, type=str, help="Pfad zum Ordner mit samples_te.pt & poses_te.pt")
    args = parser.parse_args()
    main(args.robot_name, args.dataset_path)
