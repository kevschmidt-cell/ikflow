import xml.etree.ElementTree as ET
import numpy as np

def rpy_to_matrix(rpy):
    roll, pitch, yaw = rpy
    cx, cy, cz = np.cos([roll, pitch, yaw])
    sx, sy, sz = np.sin([roll, pitch, yaw])
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx

def xyz_rpy_to_matrix(xyz, rpy):
    T = np.eye(4)
    T[:3,:3] = rpy_to_matrix(rpy)
    T[:3,3] = xyz
    return T

def get_offsets_from_urdf(urdf_path, parent="obj_com", children=("obj_p_01","obj_p_02")):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    offsets = {}
    for joint in root.findall("joint"):
        if joint.get("type") != "fixed":
            continue

        parent_link = joint.find("parent").get("link").strip()
        child_link = joint.find("child").get("link").strip()

        if parent_link != parent or child_link not in children:
            continue

        origin = joint.find("origin")
        if origin is None:
            xyz = [0.0,0.0,0.0]
            rpy = [0.0,0.0,0.0]
        else:
            xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]

        T = xyz_rpy_to_matrix(np.array(xyz), np.array(rpy))
        T_inv = np.linalg.inv(T)

        offsets[child_link] = {
            "offset": T,
            "offset_inv": T_inv
        }
        print(f"Found joint {joint.get('name')}: parent={parent_link}, child={child_link}, T=\n{T}\n")

    print("Offsets keys found:", list(offsets.keys()))
    return offsets

if __name__ == "__main__":
    urdf_path = "urdfs/object/se3_object.urdf"  # Passe den Pfad ggf. an
    offsets = get_offsets_from_urdf(urdf_path)
    print(offsets["obj_p_01"])
    print(offsets["obj_p_02"])
