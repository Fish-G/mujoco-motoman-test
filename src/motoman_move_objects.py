import glob
import math
import re
import time
import typing
import random
import json
import mujoco_viewer
from cloudinit.subp import target_path
from dm_control import mjcf
from dm_control import mujoco
from init_scene import asset_dict
import numpy as np
import transformations as tfm

def asset_dict(assets_dir):
    ASSETS = dict()
    for fname in glob.glob(assets_dir + '/*.obj'):
        with open(fname, 'rb') as f:
            ASSETS[fname] = f.read()
    return ASSETS


def tgt_obj_coordinates(obj):
    scale_factor = 0.03
    translate = np.array([0.75, 0, 0.17])
    pos = scale_factor * np.array([obj["bottom_left_pos"]["y"] + 0.5*obj["dimensions"]["depth_y"], obj["bottom_left_pos"]["x"] + 0.5*obj["dimensions"]["width_x"], obj["bottom_left_pos"]["z"] + 0.5*obj["dimensions"]["height_z"]])

    shape = scale_factor * np.array([0.5*obj["dimensions"]["width_x"], 0.5*obj["dimensions"]["depth_y"], 0.5*obj["dimensions"]["height_z"]], dtype=np.float64)

    pos += translate

    return pos, shape


def load_objects(world, object_json: str):
    scene_dict = None
    with open(object_json, 'r') as f:
        scene_dict = json.load(f)
    if scene_dict is None:
        print("Failed to read object_json")
        return

    objects = scene_dict["objects"]
    print(objects)

    for i, obj in enumerate(objects):
        # print(obj)
        pos, shape = tgt_obj_coordinates(obj)
        print(pos, shape)
        target_body = world.worldbody.add("body", name=f"obj_{obj["id"]}_tgt", pos=f"{pos[0]} {pos[1]} {pos[2]}")

        # body.add("joint", type="free")
        color = f"{random.random()} {random.random()} {random.random()}"
        target_body.add("geom", type="box", size=f"{shape[0]} {shape[1]} {shape[2]}", rgba=f"{color} 0.2", quat="0.707107 0 0 0.707107")


        obj_body = world.worldbody.add("body", name=f"obj_{obj["id"]}", pos=f"{-0.6 + i/10} 1 0.2")
        obj_body.add("geom", type="box", size=f"{shape[0]} {shape[1]} {shape[2]}", rgba=f"{color} 1")
        obj_body.add("joint", type="free")

def init(robot_xml: str, assets_dir: str, scene_xml: str, scene_objects: str, gui=True):
    world = mjcf.from_path(scene_xml)
    world.attach(mjcf.from_path(robot_xml))

    load_objects(world, scene_objects)

    assets = asset_dict(assets_dir)

    xml_str = re.sub('-[a-f0-9]+.obj', '.obj', world.to_xml_string())

    print(xml_str)

    world = mujoco.MjModel.from_xml_string(xml=xml_str, assets=assets)
    data = mujoco.MjData(world)

    viewer = mujoco_viewer.MujocoViewer(world, data) if gui else None

    return world, data, viewer


def main():
    robot_xml = "../ur5e/ur5e.xml"
    scene_xml = "../scenes/ur5e_scene.xml"
    assets_dir = "../ur5e/assets/"

    scene_objects = "../scenes/robot_experiment_small.json"

    world, data, viewer = init(robot_xml, assets_dir, scene_xml, scene_objects)

    while viewer.is_alive:
        mujoco.mj_step(world, data)
        viewer.render()
    viewer.close()

    print("t")


if __name__ == '__main__':
    main()
