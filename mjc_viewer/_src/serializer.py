import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import mujoco
import numpy as np
from dm_control import mjcf
from dm_control.mjcf import constants
from dm_control.mjcf.traversal_utils import commit_defaults
from dm_robotics.transformations import transformations as tr
from google.protobuf import json_format

from mjc_viewer._src import config_pb2

MjcfElement = Any
ProtoCollider = config_pb2.Collider

_DEFAULT_GEOM_TYPE = "sphere"

_HTML = """
<html>
  <head>
    <title>MuJoCo visualizer</title>
    <style>
      body {
        margin: 0;
        padding: 0;
      }
      #mujoco-viewer {
        margin: 0;
        padding: 0;
        height: <!-- viewer height goes here -->;
      }
    </style>
  </head>
  <body>
    <script type="application/javascript">
      var system = <!-- system json goes here -->;
    </script>
    <div id="mujoco-viewer"></div>
    <script type="module">
      import {Viewer} from 'https://cdn.jsdelivr.net/gh/kevinzakka/mjc_viewer@main/mjc_viewer/_src/js/viewer.js';
      const domElement = document.getElementById('mujoco-viewer');
      var viewer = new Viewer(domElement, system);
    </script>
  </body>
</html>
"""


def _pos_quat_from_model(name: str, model: mujoco.MjModel) -> np.ndarray:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    geom_pos = model.geom_pos[geom_id]
    geom_quat = model.geom_quat[geom_id]
    return geom_pos, geom_quat


def _quat_to_euler_degrees(quat: np.ndarray) -> np.ndarray:
    return tr.quat_to_euler(quat) * 180.0 / np.pi


def _parse_sphere(sphere: MjcfElement, model: mujoco.MjModel) -> ProtoCollider:
    if sphere.size is None:
        commit_defaults(sphere)
    radius = sphere.size[0]
    position, _ = _pos_quat_from_model(sphere.name, model)
    return config_pb2.Collider(
        sphere=config_pb2.Collider.Sphere(radius=radius),
        position=config_pb2.Vector3(x=position[0], y=position[1], z=position[2]),
    )


def _parse_capsule(
    capsule: MjcfElement, model: mujoco.MjModel, add_radius: bool = True
) -> ProtoCollider:
    if capsule.size is None:
        commit_defaults(capsule)
    size = capsule.size
    radius = size[0]
    if capsule.fromto is not None:
        start = capsule.fromto[0:3]
        end = capsule.fromto[3:6]
        direction = end - start
        length = np.linalg.norm(direction)
    else:
        length = size[1] * 2
    if add_radius:
        length += 2 * radius
    position, quat = _pos_quat_from_model(capsule.name, model)
    rot_euler = _quat_to_euler_degrees(quat)
    return config_pb2.Collider(
        capsule=config_pb2.Collider.Capsule(radius=radius, length=length),
        rotation=config_pb2.Vector3(x=rot_euler[0], y=rot_euler[1], z=rot_euler[2]),
        position=config_pb2.Vector3(x=position[0], y=position[1], z=position[2]),
    )


def _parse_box(box: MjcfElement, model: mujoco.MjModel) -> ProtoCollider:
    if box.size is None:
        commit_defaults(box)
    size = box.size
    position, quat = _pos_quat_from_model(box.name, model)
    rot_euler = _quat_to_euler_degrees(quat)
    return config_pb2.Collider(
        box=config_pb2.Collider.Box(
            halfsize=config_pb2.Vector3(x=size[0], y=size[1], z=size[2])
        ),
        rotation=config_pb2.Vector3(x=rot_euler[0], y=rot_euler[1], z=rot_euler[2]),
        position=config_pb2.Vector3(x=position[0], y=position[1], z=position[2]),
    )


def _parse_cylinder(cylinder: MjcfElement, model: mujoco.MjModel) -> ProtoCollider:
    return _parse_capsule(cylinder, model, False)


def _parse_mesh(mesh: MjcfElement, model: mujoco.MjModel) -> ProtoCollider:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh.mesh.name)
    position = model.geom_pos[geom_id]
    quat = model.geom_quat[geom_id]
    rot_euler = _quat_to_euler_degrees(quat)
    return config_pb2.Collider(
        mesh=config_pb2.Collider.Mesh(
            name=mesh.mesh.name,
            scale=1.0,
        ),
        rotation=config_pb2.Vector3(x=rot_euler[0], y=rot_euler[1], z=rot_euler[2]),
        position=config_pb2.Vector3(x=position[0], y=position[1], z=position[2]),
    )


def _parse_plane(plane: MjcfElement, model: mujoco.MjModel) -> ProtoCollider:
    del plane, model  # Unused.
    return config_pb2.Collider(plane=config_pb2.Collider.Plane())


def parse_geom(geom: MjcfElement, model: mujoco.MjModel) -> ProtoCollider:
    if geom.type == "box":
        return _parse_box(geom, model)
    elif geom.type == "sphere":
        return _parse_sphere(geom, model)
    elif geom.type == "cylinder":
        return _parse_cylinder(geom, model)
    elif geom.type == "capsule":
        return _parse_capsule(geom, model)
    elif geom.type == "mesh":
        return _parse_mesh(geom, model)
    elif geom.type == "plane":
        return _parse_plane(geom, model)
    else:
        raise ValueError(f"Unknown geom type: {geom.type}")


class Serializer:
    """Serializes a MuJoCo model as a protobuf."""

    def __init__(
        self,
        model: mujoco.MjModel,
        assets: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._config = config_pb2.Config()

        self._mj_model = model
        with tempfile.TemporaryDirectory() as tmp_dir:
            xml_path = Path(tmp_dir) / "model.xml"
            mujoco.mj_saveLastXML(str(xml_path), model)
            self._mjcf_model = mjcf.from_path(
                str(xml_path), assets=assets, escape_separators=True
            )

        # Set simulation timestep.
        self._config.dt = self._mj_model.opt.timestep

        # Add the bodies in depth-first order.
        self._add_body(self._mjcf_model.worldbody, None)

    def _add_body(self, body: MjcfElement, parent_body: Optional[MjcfElement]) -> None:
        body_idx = len(self._config.bodies)
        _body = self._config.bodies.add()

        if not parent_body:
            _body.name = constants.WORLDBODY
            _body.frozen.position.x = (
                _body.frozen.position.y
            ) = _body.frozen.position.z = 1
            _body.frozen.rotation.x = (
                _body.frozen.rotation.y
            ) = _body.frozen.rotation.z = 1
        else:
            _body.name = body.name if body.name else f"body{body_idx}"

        geoms = body.geom if hasattr(body, "geom") else []

        if geoms:
            if _body.name == constants.WORLDBODY:
                if geoms[0].name:
                    _body.name = geoms[0].name
                else:
                    if geoms[0].type == "plane":
                        _body.name = "ground"

        for geom in geoms:
            if geom.type is None:
                # Check first if specified in defaults.
                commit_defaults(geom, "type")
                # If still None, fallback to MuJoCo default.
                if geom.type is None:
                    geom.type = _DEFAULT_GEOM_TYPE
            _body.colliders.append(parse_geom(geom, self._mj_model))

        # Recurse.
        for child_body in body.body:
            self._add_body(child_body, body)

    @property
    def config(self) -> config_pb2.Config:
        return self._config

    def render(
        self,
        positions: Sequence[np.ndarray],
        quaternions: Sequence[np.ndarray],
        height: int = 480,
    ) -> str:
        d = {
            "config": json_format.MessageToDict(self.config, True),
            "pos": [pos.tolist() for pos in positions],
            "rot": [quat.tolist() for quat in quaternions],
        }
        system = json.dumps(d)
        html = _HTML.replace("<!-- system json goes here -->", system)
        return html.replace("<!-- viewer height goes here -->", f"{height}px")
