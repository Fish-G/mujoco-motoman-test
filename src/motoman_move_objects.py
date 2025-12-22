import re
import typing

import json
import mujoco_viewer
from dm_control import mjcf
from dm_control import mujoco
from init_scene import asset_dict

def load_objects(world, object_json:str):
    scene_dict = None
    with open(object_json, 'r') as f:
        scene_dict = json.load(f)
    if scene_dict is None:
        print("Failed to read object_json")
        return

    objects = scene_dict["objects"]
    print(objects)

    for obj in objects:
        # print(obj)
        body = world.worldbody.add("body", name=f"obj_{obj["id"]}", pos=f"{obj["bottom_left_pos"]["x"]/50} {obj["bottom_left_pos"]["y"]/50} {obj["bottom_left_pos"]["z"]/50}")

        body.add("joint", type="free")
        body.add("geom", type="box", size=f"{obj["dimensions"]["width_x"]/50} {obj["dimensions"]["depth_y"]/50} {obj["dimensions"]["height_z"]/50}", rgba="1 0 0 1")




def init(robot_xml: str, assets_dir: str, scene_xml: str, scene_objects: str, gui=True):
    world = mjcf.from_path(scene_xml)
    world.attach(mjcf.from_path(robot_xml))

    load_objects(world, scene_objects)


    assets = asset_dict(assets_dir)

    xml_str = re.sub('-[a-f0-9]+.stl', '.stl', world.to_xml_string())

    print(xml_str)

    world = mujoco.MjModel.from_xml_string(xml=xml_str,assets=assets)
    data = mujoco.MjData(world)

    viewer = mujoco_viewer.MujocoViewer(world, data) if gui else None

    return world, data, viewer


def main():
    robot_xml = "../motoman/motoman.xml"
    scene_xml = "../scenes/two_table_1.xml"
    assets_dir = "../motoman/meshes"

    scene_objects = "../scenes/robot_experiment_small.json"

    world, data, viewer = init(robot_xml,assets_dir, scene_xml, scene_objects)

    while viewer.is_alive:
        mujoco.mj_step(world,data)
        viewer.render()
    viewer.close()

    print("t")


if __name__ == '__main__':
    main()
