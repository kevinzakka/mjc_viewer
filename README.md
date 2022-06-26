# mjc_viewer

[![PyPI Python Version][pypi-versions-badge]][pypi]
[![PyPI version][pypi-badge]][pypi]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kevinzakka/mjc_viewer/blob/master/tutorial.ipynb)

[pypi-versions-badge]: https://img.shields.io/pypi/pyversions/mjc_viewer
[pypi-badge]: https://badge.fury.io/py/mjc_viewer.svg
[pypi]: https://pypi.org/project/mjc_viewer/

`mjc_viewer` is a browser-based 3D viewer for [MuJoCo](https://mujoco.org/) that can render static trajectories from JSON.

## Installation

The recommended way to install this package is via [PyPI](https://pypi.org/project/mjc_viewer/):

```bash
pip install mjc_viewer
```

## Usage

```python
import numpy as np
import mujoco
from mjc_viewer import Serializer, Trajectory

# Load your MuJoCo model.
model = mujoco.MjModel.from_xml_path("humanoid.xml")
data = mujoco.MjData(model)

# Create a Serializer and Trajectory instance.
serializer = Serializer(model)
trajectory = Trajectory(data)

# Simulate for 3 seconds.
trajectory.reset()
while data.time < 3.0:
    data.ctrl = np.random.uniform(*model.actuator_ctrlrange.T)
    mujoco.mj_step(model, data)
    trajectory.step()

html = serializer.render(trajectory)
with open("traj.html", "w") as f:
    f.write(html)
# You can now open traj.html in a browser or render in a notebook with
# `IPython.display.HTML`.
```

## Acknowledgements

`mjc_viewer` is heavily adapted from [Brax](https://github.com/google/brax)'s javascript viewer, full credit goes to its developers.
