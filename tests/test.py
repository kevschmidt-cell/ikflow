from klampt import WorldModel, vis, Geometry3D
from klampt.math import vectorops
from klampt.model import coordinates
from jrl.robots import Iiwa7_L, Iiwa7_R
import numpy as np

robot = Iiwa7_R()
q, poses = robot.sample_joint_angles_and_poses(1)
q = q[0]
pose = poses[0]

# Klampt-Welt laden
world = WorldModel()
world.loadElement(robot.urdf_filepath)
klampt_robot = world.robot(0)
print("type q:", type(q))
print("q shape:", np.shape(q))
print("q:", q)

print("klampt_robot.getConfig() length:", len(klampt_robot.getConfig()))
print("klampt_robot.numDrivers():", klampt_robot.numDrivers())



print("Endeffektor-Link-Name:", robot.end_effector_link_name)
print("Anzahl DOFs Klampt-Roboter:", klampt_robot.numDrivers())
print("Länge q:", len(q))
print("q.tolist() =", q.tolist())
print("Type of q.tolist() =", type(q.tolist()))
print("Length of q.tolist() =", len(q.tolist()))
for i, val in enumerate(q.tolist()):
    print(f"q[{i}] = {val}, type: {type(val)}")

full_q = klampt_robot.getConfig()
print("full_q type:", type(full_q))
print("full_q:", full_q)
print("full_q length:", len(full_q))

full_q[2:9] = q.tolist()

klampt_robot.setConfig(full_q)

print("Set config erfolgreich")


# Marker an die tatsächlich genutzte Endeffektor-Position setzen
ee_link = klampt_robot.link(robot.end_effector_link_name)
print("Link index:", klampt_robot.link(robot.end_effector_link_name).index)

R, t = ee_link.getTransform()
print("t =", t)
link8 = klampt_robot.link(8)
print("Link 7 index:", link8.getName())
print("Link 7 transform:", link8.getTransform())

vis.add("world", world)
vis.addText("ee_pose", f"EE Pos:\n{t}")
vis.setAttribute("ee_pose", "position", t)  # Position des Texts im Raum
vis.show()
input("Drücke Enter zum Schließen...")
vis.kill()
