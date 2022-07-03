import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import mujoco
import numpy as np
from dm_control import mjcf
from dm_control.mjcf import constants
from dm_control.mjcf.traversal_utils import commit_defaults
from dm_robotics.transformations import transformations as tr
from google.protobuf import json_format

from mjc_viewer._src import config_pb2, trajectory

# TODO(kevin): Fix mypy errors.
MjcfElement = Any

_DEFAULT_GEOM_TYPE = "sphere"
_DEFAULT_GEOM_RGBA = np.array([0.5, 0.5, 0.5, 1])
_IDENTITY_QUAT = np.array([1.0, 0, 0, 0])

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


def _is_worldbody(body: MjcfElement) -> bool:
    return body.tag == constants.WORLDBODY


def _maybe_to_local(pos: np.ndarray, body: MjcfElement) -> np.ndarray:
    if body.root.compiler.coordinate == "global" and body and not _is_worldbody(body):
        return pos - body.pos
    return pos


def _maybe_to_radian(angle: np.ndarray, body: MjcfElement) -> np.ndarray:
    if body.root.compiler.angle == "degrees":
        return np.radians(angle)
    return angle


def _pos_from_mjcf(elem: MjcfElement) -> np.ndarray:
    if elem.pos is None:
        return np.zeros(3)
    return _maybe_to_local(elem.pos, elem.parent)


def _quat_from_mjcf(elem: MjcfElement) -> np.ndarray:
    if _is_worldbody(elem):
        return _IDENTITY_QUAT
    if elem.euler is not None:
        return tr.euler_to_quat(_maybe_to_radian(elem.euler, elem))
    if elem.quat is not None:
        return elem.quat
    if elem.axisangle is not None:
        return tr.axisangle_to_quat(elem.axisangle)
    return _IDENTITY_QUAT


def _pos_quat_from_model(
    elem: MjcfElement, model: mujoco.MjModel
) -> Tuple[np.ndarray, np.ndarray]:
    if elem.name is not None:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, elem.name)
        geom_pos = model.geom_pos[geom_id]
        geom_quat = model.geom_quat[geom_id]
        return geom_pos, geom_quat
    pos = _pos_from_mjcf(elem)
    quat = _quat_from_mjcf(elem)
    return pos, quat


def _quat_to_euler_degrees(quat: np.ndarray) -> np.ndarray:
    return tr.quat_to_euler(quat) * 180.0 / np.pi


def _parse_sphere(sphere: MjcfElement, model: mujoco.MjModel) -> config_pb2.Collider:
    if sphere.size is None:
        commit_defaults(sphere)
    size = sphere.size
    if size is None:
        raise ValueError(f"sphere elem {sphere.name} is missing a size attribute")
    radius = size[0]
    position, _ = _pos_quat_from_model(sphere, model)
    return config_pb2.Collider(
        sphere=config_pb2.Collider.Sphere(radius=radius),
        position=config_pb2.Vector3(x=position[0], y=position[1], z=position[2]),
        material=_parse_material(sphere),
    )


def _parse_capsule(
    capsule: MjcfElement, model: mujoco.MjModel, add_radius: bool = True
) -> config_pb2.Collider:
    if capsule.size is None:
        commit_defaults(capsule)
    size = capsule.size
    if size is None:
        raise ValueError(f"capsule elem {capsule.name} is missing a size attribute")
    radius = size[0]
    if capsule.fromto is not None:
        start = capsule.fromto[0:3]
        end = capsule.fromto[3:6]
        direction = end - start
        length = float(np.linalg.norm(direction))
    else:
        length = size[1] * 2
    if add_radius:
        length += 2 * radius
    position, quat = _pos_quat_from_model(capsule, model)
    rot_euler = _quat_to_euler_degrees(quat)
    return config_pb2.Collider(
        capsule=config_pb2.Collider.Capsule(radius=radius, length=length),
        rotation=config_pb2.Vector3(x=rot_euler[0], y=rot_euler[1], z=rot_euler[2]),
        position=config_pb2.Vector3(x=position[0], y=position[1], z=position[2]),
        material=_parse_material(capsule),
    )


def _parse_box(box: MjcfElement, model: mujoco.MjModel) -> config_pb2.Collider:
    if box.size is None:
        commit_defaults(box)
    size = box.size
    if size is None:
        raise ValueError(f"box elem {box.name} is missing a size attribute")
    position, quat = _pos_quat_from_model(box, model)
    rot_euler = _quat_to_euler_degrees(quat)
    return config_pb2.Collider(
        box=config_pb2.Collider.Box(
            halfsize=config_pb2.Vector3(x=size[0], y=size[1], z=size[2])
        ),
        rotation=config_pb2.Vector3(x=rot_euler[0], y=rot_euler[1], z=rot_euler[2]),
        position=config_pb2.Vector3(x=position[0], y=position[1], z=position[2]),
        material=_parse_material(box),
    )


def _parse_mesh(mesh: MjcfElement, model: mujoco.MjModel) -> config_pb2.Collider:
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


def _parse_plane(plane: MjcfElement, model: mujoco.MjModel) -> config_pb2.Collider:
    if plane.size is None:
        commit_defaults(plane)
    size = plane.size
    if size is None:
        raise ValueError(f"plane elem {plane.name} is missing a size attribute")
    position, _ = _pos_quat_from_model(plane, model)
    return config_pb2.Collider(
        plane=config_pb2.Collider.Plane(
            size=config_pb2.Vector3(x=size[0] * 2, y=size[1] * 2, z=size[2]),
            position=config_pb2.Vector3(x=position[0], y=position[1], z=position[2]),
        ),
        material=_parse_material(plane),
    )


def _parse_ellipsoid(
    ellipsoid: MjcfElement, model: mujoco.MjModel
) -> config_pb2.Collider:
    if ellipsoid.size is None:
        commit_defaults(ellipsoid)
    size = ellipsoid.size
    if size is None:
        raise ValueError(f"ellipsoid elem {ellipsoid.name} is missing a size attribute")
    position, quat = _pos_quat_from_model(ellipsoid, model)
    rot_euler = _quat_to_euler_degrees(quat)
    return config_pb2.Collider(
        ellipsoid=config_pb2.Collider.Ellipsoid(
            radius=config_pb2.Vector3(x=size[0], y=size[1], z=size[2]),
        ),
        rotation=config_pb2.Vector3(x=rot_euler[0], y=rot_euler[1], z=rot_euler[2]),
        position=config_pb2.Vector3(x=position[0], y=position[1], z=position[2]),
        material=_parse_material(ellipsoid),
    )


def _parse_cylinder(
    cylinder: MjcfElement, model: mujoco.MjModel
) -> config_pb2.Collider:
    if cylinder.size is None:
        commit_defaults(cylinder)
    size = cylinder.size
    if size is None:
        raise ValueError(f"cylinder elem {cylinder.name} is missing a size attribute")
    position, quat = _pos_quat_from_model(cylinder, model)
    rot_euler = _quat_to_euler_degrees(quat)
    return config_pb2.Collider(
        cylinder=config_pb2.Collider.Cylinder(
            radius=size[0],
            length=size[1] * 2,
        ),
        rotation=config_pb2.Vector3(x=rot_euler[0], y=rot_euler[1], z=rot_euler[2]),
        position=config_pb2.Vector3(x=position[0], y=position[1], z=position[2]),
        material=_parse_material(cylinder),
    )


def _parse_material(elem: MjcfElement) -> config_pb2.Material:
    commit_defaults(elem, ["rgba", "material"])
    if elem.material is not None:
        color = None
        if elem.material.rgba is not None:
            color = elem.material.rgba
        if elem.material.texture is not None:
            texture = elem.material.texture
            if texture.builtin is not None:
                if texture.builtin == "checker":
                    if elem.material.texrepeat is None:
                        repeat = 1
                    else:
                        repeat = elem.material.texrepeat[0]
                    return config_pb2.Material(
                        texture=config_pb2.Texture(
                            type="checker",
                            color1=config_pb2.Vector3(
                                x=texture.rgb1[0], y=texture.rgb1[1], z=texture.rgb1[2]
                            ),
                            color2=config_pb2.Vector3(
                                x=texture.rgb2[0], y=texture.rgb2[1], z=texture.rgb2[2]
                            ),
                            repeat=repeat,
                            size=config_pb2.Vector3(
                                x=elem.size[0], y=elem.size[1], z=elem.size[2]
                            ),
                        ),
                    )
                elif texture.builtin == "flat":
                    if color is None:
                        color = texture.rgb1
                    return config_pb2.Material(
                        texture=config_pb2.Texture(
                            type="flat",
                            color1=config_pb2.Vector3(
                                x=color[0], y=color[1], z=color[2]
                            ),
                        ),
                    )
    # If no material is specified, it could be that solely an rgba attribute was
    # specified. If we don't have one, then we'll just assign the default color.
    rgba = _DEFAULT_GEOM_RGBA if elem.rgba is None else elem.rgba
    return config_pb2.Material(
        color=config_pb2.Vector3(x=rgba[0], y=rgba[1], z=rgba[2]),
        alpha=rgba[3],
    )


def parse_geom(geom: MjcfElement, model: mujoco.MjModel) -> config_pb2.Collider:
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
    elif geom.type == "ellipsoid":
        return _parse_ellipsoid(geom, model)
    else:
        raise ValueError(f"Unknown geom type: {geom.type}")


class Serializer:
    """Serializes a MuJoCo model to a protocol buffer."""

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

        # Recursively add all the bodies, starting with the worldbody.
        self._add_body(self._mjcf_model.worldbody, None)

    def _add_body(self, body: MjcfElement, parent_body: Optional[MjcfElement]) -> None:
        body_idx = len(self._config.bodies)
        _body = self._config.bodies.add()

        if not parent_body:
            _body.name = constants.WORLDBODY
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

    def render(self, trajectory: trajectory.Trajectory, height: int = 480) -> str:
        d = {
            "config": json_format.MessageToDict(self.config, True),
            "pos": [pos.tolist() for pos in trajectory.positions],
            "rot": [quat.tolist() for quat in trajectory.rotations],
        }
        system = json.dumps(d)
        html = _HTML.replace("<!-- system json goes here -->", system)
        return html.replace("<!-- viewer height goes here -->", f"{height}px")
