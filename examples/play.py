from dataclasses import dataclass

import dcargs
import mujoco
import numpy as np

from mjc_viewer import Serializer, Trajectory


@dataclass(frozen=True)
class Args:
    xml_path: str
    html_path: str
    duration: float = 3.0
    add_noise: bool = False


def main(args: Args) -> None:
    # Load MuJoCo model.
    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)

    # Construct necessary viewer objects.
    serializer = Serializer(model)
    trajectory = Trajectory(data)

    # Simulate!
    trajectory.reset()
    while data.time < args.duration:
        if args.add_noise:
            data.ctrl = np.random.uniform(*model.actuator_ctrlrange.T)
        mujoco.mj_step(model, data)
        trajectory.step()

    # Dump as HTML.
    html = serializer.render(trajectory)
    with open(args.html_path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main(dcargs.cli(Args))
