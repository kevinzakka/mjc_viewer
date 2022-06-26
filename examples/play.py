from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dcargs
import mujoco
import numpy as np

from mujoco_viewer import Serializer

AssetsDict = Optional[Dict[str, Any]]

_HERE = Path(__file__).resolve().parent
_MJCF_PATH = _HERE / "mjcf"


@dataclass(frozen=True)
class Args:
    out_path: str
    noisy_ctrl: bool = False
    duration: float = 1.0


def load_ant() -> Tuple[mujoco.MjModel, AssetsDict]:
    model = mujoco.MjModel.from_xml_path(str(_MJCF_PATH / "./ant.xml"))
    return model, None


def load_half_cheetah() -> Tuple[mujoco.MjModel, AssetsDict]:
    model = mujoco.MjModel.from_xml_path(str(_MJCF_PATH / "./half_cheetah.xml"))
    return model, None


def load_humanoid() -> Tuple[mujoco.MjModel, AssetsDict]:
    model = mujoco.MjModel.from_xml_path(str(_MJCF_PATH / "./humanoid.xml"))
    return model, None


def main(args: Args) -> None:
    model, assets = load_ant()
    data = mujoco.MjData(model)

    serializer = Serializer(model, assets)

    # Simulate!
    positions: List[np.ndarray] = []
    quaternions: List[np.ndarray] = []
    while data.time < args.duration:
        if args.noisy_ctrl:
            data.ctrl = np.random.uniform(*model.actuator_ctrlrange.T)
        mujoco.mj_step(model, data)
        positions.append(data.xpos.copy())
        quaternions.append(data.xquat.copy())

    # Dump as an HTML file.
    with open(Path(args.out_path) / "test.html", "w") as f:
        f.write(serializer.render(positions, quaternions))


if __name__ == "__main__":
    main(dcargs.cli(Args))
