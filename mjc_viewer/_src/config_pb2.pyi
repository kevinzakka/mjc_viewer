"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class Vector3(google.protobuf.message.Message):
    """A point or scalar value in 3d space."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    X_FIELD_NUMBER: builtins.int
    Y_FIELD_NUMBER: builtins.int
    Z_FIELD_NUMBER: builtins.int
    x: builtins.float
    y: builtins.float
    z: builtins.float
    def __init__(self,
        *,
        x: builtins.float = ...,
        y: builtins.float = ...,
        z: builtins.float = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["x",b"x","y",b"y","z",b"z"]) -> None: ...
global___Vector3 = Vector3

class Frozen(google.protobuf.message.Message):
    """Prevents motion or rotation along specifed axes."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    POSITION_FIELD_NUMBER: builtins.int
    ROTATION_FIELD_NUMBER: builtins.int
    ALL_FIELD_NUMBER: builtins.int
    @property
    def position(self) -> global___Vector3:
        """Freeze motion along the x, y, or z axes."""
        pass
    @property
    def rotation(self) -> global___Vector3:
        """Freeze rotation around the x, y, or z axes."""
        pass
    all: builtins.bool
    """Override all the position and rotation fields, setting them to 1."""

    def __init__(self,
        *,
        position: typing.Optional[global___Vector3] = ...,
        rotation: typing.Optional[global___Vector3] = ...,
        all: builtins.bool = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["position",b"position","rotation",b"rotation"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["all",b"all","position",b"position","rotation",b"rotation"]) -> None: ...
global___Frozen = Frozen

class Body(google.protobuf.message.Message):
    """Bodies have a rigid shape, mass, and rotational inertia. Bodies may connect
    to other bodies via joints, forming a kinematic tree.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    NAME_FIELD_NUMBER: builtins.int
    COLLIDERS_FIELD_NUMBER: builtins.int
    FROZEN_FIELD_NUMBER: builtins.int
    name: typing.Text
    """Unique name for this body"""

    @property
    def colliders(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Collider]:
        """Geometric primitives that define the shape of the body"""
        pass
    @property
    def frozen(self) -> global___Frozen:
        """Prevents motion or rotation along specified axes for this body."""
        pass
    def __init__(self,
        *,
        name: typing.Text = ...,
        colliders: typing.Optional[typing.Iterable[global___Collider]] = ...,
        frozen: typing.Optional[global___Frozen] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["frozen",b"frozen"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["colliders",b"colliders","frozen",b"frozen","name",b"name"]) -> None: ...
global___Body = Body

class Collider(google.protobuf.message.Message):
    """Primitive shape that composes the collision surface of a body."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class Box(google.protobuf.message.Message):
        """A 6-sided rectangular prism"""
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        HALFSIZE_FIELD_NUMBER: builtins.int
        @property
        def halfsize(self) -> global___Vector3:
            """Half the size of the box in each dimension."""
            pass
        def __init__(self,
            *,
            halfsize: typing.Optional[global___Vector3] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["halfsize",b"halfsize"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["halfsize",b"halfsize"]) -> None: ...

    class Plane(google.protobuf.message.Message):
        """An infinite plane with normal vector (0, 0, 1)"""
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        def __init__(self,
            ) -> None: ...

    class Sphere(google.protobuf.message.Message):
        """A sphere"""
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        RADIUS_FIELD_NUMBER: builtins.int
        radius: builtins.float
        def __init__(self,
            *,
            radius: builtins.float = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["radius",b"radius"]) -> None: ...

    class Capsule(google.protobuf.message.Message):
        """A cylinder with rounded ends."""
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        RADIUS_FIELD_NUMBER: builtins.int
        LENGTH_FIELD_NUMBER: builtins.int
        END_FIELD_NUMBER: builtins.int
        radius: builtins.float
        """Radius of the sphere at each rounded end"""

        length: builtins.float
        """End-to-end length of the capsule"""

        end: builtins.int
        """Capsule end (0: both ends, 1: top end, -1: bottom end)"""

        def __init__(self,
            *,
            radius: builtins.float = ...,
            length: builtins.float = ...,
            end: builtins.int = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["end",b"end","length",b"length","radius",b"radius"]) -> None: ...

    class HeightMap(google.protobuf.message.Message):
        """A height map aligned with the x-y plane"""
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        SIZE_FIELD_NUMBER: builtins.int
        DATA_FIELD_NUMBER: builtins.int
        size: builtins.float
        """The width and length of the square height map."""

        @property
        def data(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]:
            """A flattened square matrix of the height data in row major order where
            the row index is the x-position and the column index is the y-position.
            """
            pass
        def __init__(self,
            *,
            size: builtins.float = ...,
            data: typing.Optional[typing.Iterable[builtins.float]] = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["data",b"data","size",b"size"]) -> None: ...

    class Mesh(google.protobuf.message.Message):
        """A mesh. Currently, only mesh-plane and mesh-capsule collisions are
        supported.
        """
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        NAME_FIELD_NUMBER: builtins.int
        SCALE_FIELD_NUMBER: builtins.int
        name: typing.Text
        """Name of the mesh geometry defined in the config."""

        scale: builtins.float
        """Scaling for the mesh."""

        def __init__(self,
            *,
            name: typing.Text = ...,
            scale: builtins.float = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["name",b"name","scale",b"scale"]) -> None: ...

    POSITION_FIELD_NUMBER: builtins.int
    ROTATION_FIELD_NUMBER: builtins.int
    BOX_FIELD_NUMBER: builtins.int
    PLANE_FIELD_NUMBER: builtins.int
    SPHERE_FIELD_NUMBER: builtins.int
    CAPSULE_FIELD_NUMBER: builtins.int
    HEIGHTMAP_FIELD_NUMBER: builtins.int
    MESH_FIELD_NUMBER: builtins.int
    COLOR_FIELD_NUMBER: builtins.int
    HIDDEN_FIELD_NUMBER: builtins.int
    @property
    def position(self) -> global___Vector3:
        """Position relative to parent body"""
        pass
    @property
    def rotation(self) -> global___Vector3:
        """Rotation relative to parent body"""
        pass
    @property
    def box(self) -> global___Collider.Box: ...
    @property
    def plane(self) -> global___Collider.Plane: ...
    @property
    def sphere(self) -> global___Collider.Sphere: ...
    @property
    def capsule(self) -> global___Collider.Capsule: ...
    @property
    def heightMap(self) -> global___Collider.HeightMap: ...
    @property
    def mesh(self) -> global___Collider.Mesh: ...
    color: typing.Text
    """Color of the collider in css notation (e.g. '#ff0000' or 'red')"""

    hidden: builtins.bool
    """A hidden collider is not visualized"""

    def __init__(self,
        *,
        position: typing.Optional[global___Vector3] = ...,
        rotation: typing.Optional[global___Vector3] = ...,
        box: typing.Optional[global___Collider.Box] = ...,
        plane: typing.Optional[global___Collider.Plane] = ...,
        sphere: typing.Optional[global___Collider.Sphere] = ...,
        capsule: typing.Optional[global___Collider.Capsule] = ...,
        heightMap: typing.Optional[global___Collider.HeightMap] = ...,
        mesh: typing.Optional[global___Collider.Mesh] = ...,
        color: typing.Text = ...,
        hidden: builtins.bool = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["box",b"box","capsule",b"capsule","heightMap",b"heightMap","mesh",b"mesh","plane",b"plane","position",b"position","rotation",b"rotation","sphere",b"sphere","type",b"type"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["box",b"box","capsule",b"capsule","color",b"color","heightMap",b"heightMap","hidden",b"hidden","mesh",b"mesh","plane",b"plane","position",b"position","rotation",b"rotation","sphere",b"sphere","type",b"type"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["type",b"type"]) -> typing.Optional[typing_extensions.Literal["box","plane","sphere","capsule","heightMap","mesh"]]: ...
global___Collider = Collider

class MeshGeometry(google.protobuf.message.Message):
    """Geometry of a mesh."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    NAME_FIELD_NUMBER: builtins.int
    PATH_FIELD_NUMBER: builtins.int
    VERTICES_FIELD_NUMBER: builtins.int
    FACES_FIELD_NUMBER: builtins.int
    VERTEX_NORMALS_FIELD_NUMBER: builtins.int
    FACE_NORMALS_FIELD_NUMBER: builtins.int
    name: typing.Text
    """Name of the mesh geometry. This is used in mesh colliders to refer to the
    geometry.
    """

    path: typing.Text
    """Path of the mesh file. See https://trimsh.org/ for the supported formats.
    If the path is specified, then the {vertices, faces, vertex_normals}
    fields below will be ignored and populated from the mesh defined in the
    file.
    """

    @property
    def vertices(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Vector3]:
        """Vertices of the mesh."""
        pass
    @property
    def faces(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        """Triangular faces. This will be a flattened array of triples that contain
        the indices of the vertices in the `vertices` field above. For example,

        vertices { x: -0.5 y: -0.5 z: 0 }
        vertices { x: +0.5 y: -0.5 z: 0 }
        vertices { x: +0.5 y: +0.5 z: 0 }
        vertices { x: -0.5 y: +0.5 z: 0 }
        vertices { x: 0 y: 0 z: 1.0 }
        faces: [0, 2, 1, 0, 3, 2, 0, 4, 3, 0, 1, 4, 1, 2, 4, 2, 3, 4]

        defines a pyramid with 6 faces (two for the bottom and four for the
        sides).
        """
        pass
    @property
    def vertex_normals(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Vector3]:
        """Unit normal vectors for each vertex."""
        pass
    @property
    def face_normals(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Vector3]:
        """Unit normal vectors for each face."""
        pass
    def __init__(self,
        *,
        name: typing.Text = ...,
        path: typing.Text = ...,
        vertices: typing.Optional[typing.Iterable[global___Vector3]] = ...,
        faces: typing.Optional[typing.Iterable[builtins.int]] = ...,
        vertex_normals: typing.Optional[typing.Iterable[global___Vector3]] = ...,
        face_normals: typing.Optional[typing.Iterable[global___Vector3]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["face_normals",b"face_normals","faces",b"faces","name",b"name","path",b"path","vertex_normals",b"vertex_normals","vertices",b"vertices"]) -> None: ...
global___MeshGeometry = MeshGeometry

class Config(google.protobuf.message.Message):
    """The configuration of a system."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    BODIES_FIELD_NUMBER: builtins.int
    DT_FIELD_NUMBER: builtins.int
    FROZEN_FIELD_NUMBER: builtins.int
    MESH_GEOMETRIES_FIELD_NUMBER: builtins.int
    @property
    def bodies(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Body]:
        """All of the bodies in the system"""
        pass
    dt: builtins.float
    """Amount of time to simulate each step, in seconds"""

    @property
    def frozen(self) -> global___Frozen:
        """Prevents motion or rotation along specified axes for the entire system"""
        pass
    @property
    def mesh_geometries(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___MeshGeometry]:
        """All of the mesh geometries in the system."""
        pass
    def __init__(self,
        *,
        bodies: typing.Optional[typing.Iterable[global___Body]] = ...,
        dt: builtins.float = ...,
        frozen: typing.Optional[global___Frozen] = ...,
        mesh_geometries: typing.Optional[typing.Iterable[global___MeshGeometry]] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["frozen",b"frozen"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["bodies",b"bodies","dt",b"dt","frozen",b"frozen","mesh_geometries",b"mesh_geometries"]) -> None: ...
global___Config = Config