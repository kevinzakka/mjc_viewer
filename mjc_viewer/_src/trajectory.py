from dataclasses import dataclass
from typing import List

import mujoco
import numpy as np


def _make_readonly_float64_copy(value: np.ndarray) -> np.ndarray:
    out = np.array(value, dtype=np.float32)
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class CartesianFrame:
    xpos: np.ndarray
    xquat: np.ndarray

    @staticmethod
    def create(data: mujoco.MjData) -> "CartesianFrame":
        return CartesianFrame(
            xpos=_make_readonly_float64_copy(data.xpos),
            xquat=_make_readonly_float64_copy(data.xquat),
        )


class Trajectory:
    def __init__(self, data: mujoco.MjData) -> None:
        self._data = data
        self._frames: List[CartesianFrame] = []

    def reset(self) -> None:
        self._frames = []

    def step(self) -> None:
        self._frames.append(CartesianFrame.create(self._data))

    @property
    def positions(self) -> List[np.ndarray]:
        return [frame.xpos for frame in self._frames]

    @property
    def rotations(self) -> List[np.ndarray]:
        return [frame.xquat for frame in self._frames]
