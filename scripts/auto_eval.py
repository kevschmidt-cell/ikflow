import subprocess
import yaml
import re
import os
import csv
from tqdm import tqdm

VERSIONS = list(range(0, 63))
RUN_ID = "j08o8cjr"
MODEL_NAME = "iiwa7_left_arm"
YAML_PATH = "/home/kevin/dev/ikflow/ikflow/model_descriptions.yaml"
CSV_PATH = "results.csv"

def run(cmd):
    print(">>", " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr)
        raise RuntimeError("Command failed")
    return out.stdout

def cleanup_old_pkls():
    """Löscht alle .pkl Dateien im aktuellen Ordner."""
    removed = 0
    for f in os.listdir("."):
        if f.endswith(".pkl"):
            os.remove(f)
            removed += 1
    if removed > 0:
        print(f"Deleted {removed} old .pkl files.")

def update_yaml(pkl_path):
    with open(YAML_PATH, "r") as f:
        data = yaml.safe_load(f)
    data[MODEL_NAME]["model_weights_url"] = pkl_path
    with open(YAML_PATH, "w") as f:
        yaml.dump(data, f)

def extract_l2_error(evaluate_output):
    m = re.search(r"Average positional error.*?:\s*([0-9.]+)", evaluate_output)
    if m:
        return float(m.group(1))
    else:
        raise ValueError("L2 error not found in output")

def main():
    # Header anlegen, falls CSV nicht existiert
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["version", "model_file", "l2_error_mm"])

    for v in tqdm(VERSIONS, desc="Processing versions", unit="version"):
        print("\n================================================")
        print(f"PROCESSING VERSION v{v}")
        print("================================================")

        # 0. Alte .pkl löschen
        cleanup_old_pkls()

        # 1. Download & convert model
        out = run([
            "python",
            "scripts/download_model_from_wandb_checkpoint.py",
            f"--wandb_run_id={RUN_ID}",
            f"--version={v}"
        ])

        # 2. Neueste PKL suchen
        pkl_files = [f for f in os.listdir(".") if f.endswith(".pkl")]
        if not pkl_files:
            print(f"No .pkl found for version v{v}, skipping.")
            continue

        latest_pkl = max(pkl_files, key=os.path.getctime)
        print(f"Using PKL: {latest_pkl}")

        # 3. YAML updaten
        update_yaml(os.path.abspath(latest_pkl))

        # 4. Evaluate
        eval_out = run([
            "python", "scripts/evaluate.py",
            "--testset_size=500",
            f"--model_name={MODEL_NAME}",
            "--solutions_per_pose=20",
            "--do_refinement"
        ])

        # 5. L2-Wert extrahieren
        l2 = extract_l2_error(eval_out)
        print("L2 error:", l2)

        # 6. Ergebnis direkt in CSV schreiben
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([v, latest_pkl, l2])

    print("\nDONE! Results saved in", CSV_PATH)

if __name__ == "__main__":
    main()

