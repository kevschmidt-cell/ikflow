import os
import torch
import numpy as np
from tqdm import tqdm
from jrl.robots import get_robot
from ikflow.utils import get_dataset_filepaths

BATCH_SIZE = 512  # je nach RAM
MAX_KEEP = int(2e7)  # wie viele gültige Samples du behalten willst

def main(robot_name, dataset_path):
    robot = get_robot(robot_name)
    print(f"Filtering training set for robot: {robot.name}")
    
    # Lade Dateipfade
    samples_tr_path, poses_tr_path, _, _, _ = get_dataset_filepaths(dataset_path, tags=["non-self-colliding"])
    samples_tr = torch.load(samples_tr_path, map_location="cpu").numpy()
    poses_tr   = torch.load(poses_tr_path, map_location="cpu").numpy()

    assert len(samples_tr) == len(poses_tr), "samples_tr und poses_tr haben unterschiedliche Längen!"

    print(f"Original: {samples_tr.shape[0]} samples")

    valid_samples = []
    valid_poses = []
    total_checked = 0
    total_valid = 0

    for i in tqdm(range(0, samples_tr.shape[0], BATCH_SIZE)):
        batch_samples = samples_tr[i:i + BATCH_SIZE]
        batch_poses = poses_tr[i:i + BATCH_SIZE]
        for x, pose in zip(batch_samples, batch_poses):
            if robot.config_self_collides(x):
                continue
            valid_samples.append(x)
            valid_poses.append(pose)
            total_valid += 1
            if total_valid >= MAX_KEEP:
                break
        total_checked += len(batch_samples)
        if total_valid >= MAX_KEEP:
            break

    valid_samples = np.stack(valid_samples)
    valid_poses = np.stack(valid_poses)

    print(f"Gefiltert: {valid_samples.shape[0]} gültige Konfigurationen von {total_checked} überprüft.")

    # Speichern
    samples_out_path = os.path.join(dataset_path, "samples_tr_filtered.pt")
    poses_out_path   = os.path.join(dataset_path, "poses_tr_filtered.pt")
    torch.save(torch.tensor(valid_samples, dtype=torch.float32), samples_out_path)
    torch.save(torch.tensor(valid_poses, dtype=torch.float32), poses_out_path)

    print(f"Gespeichert unter:\n  {samples_out_path}\n  {poses_out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()

    main(args.robot_name, args.dataset_path)
