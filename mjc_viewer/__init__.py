"""A browser-based 3D viewer for MuJoCo."""

from mjc_viewer._src.serializer import Serializer
from mjc_viewer._src.trajectory import Trajectory

__version__ = "0.0.4"

__all__ = ["Serializer", "Trajectory"]
