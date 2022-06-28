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

class Texture(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    TYPE_FIELD_NUMBER: builtins.int
    COLOR1_FIELD_NUMBER: builtins.int
    COLOR2_FIELD_NUMBER: builtins.int
    REPEAT_FIELD_NUMBER: builtins.int
    SIZE_FIELD_NUMBER: builtins.int
    type: typing.Text
    @property
    def color1(self) -> global___Vector3: ...
    @property
    def color2(self) -> global___Vector3: ...
    repeat: builtins.float
    @property
    def size(self) -> global___Vector3: ...
    def __init__(self,
        *,
        type: typing.Text = ...,
        color1: typing.Optional[global___Vector3] = ...,
        color2: typing.Optional[global___Vector3] = ...,
        repeat: builtins.float = ...,
        size: typing.Optional[global___Vector3] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["color1",b"color1","color2",b"color2","size",b"size"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["color1",b"color1","color2",b"color2","repeat",b"repeat","size",b"size","type",b"type"]) -> None: ...
global___Texture = Texture

class Material(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    COLOR_FIELD_NUMBER: builtins.int
    ALPHA_FIELD_NUMBER: builtins.int
    TEXTURE_FIELD_NUMBER: builtins.int
    EMISSION_FIELD_NUMBER: builtins.int
    SPECULAR_FIELD_NUMBER: builtins.int
    SHININESS_FIELD_NUMBER: builtins.int
    REFLECTANCE_FIELD_NUMBER: builtins.int
    @property
    def color(self) -> global___Vector3: ...
    alpha: builtins.float
    @property
    def texture(self) -> global___Texture: ...
    emission: builtins.float
    specular: builtins.float
    shininess: builtins.float
    reflectance: builtins.float
    def __init__(self,
        *,
        color: typing.Optional[global___Vector3] = ...,
        alpha: builtins.float = ...,
        texture: typing.Optional[global___Texture] = ...,
        emission: builtins.float = ...,
        specular: builtins.float = ...,
        shininess: builtins.float = ...,
        reflectance: builtins.float = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["color",b"color","texture",b"texture"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["alpha",b"alpha","color",b"color","emission",b"emission","reflectance",b"reflectance","shininess",b"shininess","specular",b"specular","texture",b"texture"]) -> None: ...
global___Material = Material

class Body(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    NAME_FIELD_NUMBER: builtins.int
    COLLIDERS_FIELD_NUMBER: builtins.int
    name: typing.Text
    """Unique name for this body"""

    @property
    def colliders(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Collider]:
        """Geometric primitives that define the shape of the body"""
        pass
    def __init__(self,
        *,
        name: typing.Text = ...,
        colliders: typing.Optional[typing.Iterable[global___Collider]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["colliders",b"colliders","name",b"name"]) -> None: ...
global___Body = Body

class Collider(google.protobuf.message.Message):
    """Primitive shape that composes the collision surface of a body"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class Box(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        HALFSIZE_FIELD_NUMBER: builtins.int
        @property
        def halfsize(self) -> global___Vector3:
            """Half the size of the box in each dimension"""
            pass
        def __init__(self,
            *,
            halfsize: typing.Optional[global___Vector3] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["halfsize",b"halfsize"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["halfsize",b"halfsize"]) -> None: ...

    class Plane(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        SIZE_FIELD_NUMBER: builtins.int
        POSITION_FIELD_NUMBER: builtins.int
        @property
        def size(self) -> global___Vector3: ...
        @property
        def position(self) -> global___Vector3:
            """TODO(kevin): Do I need this field?"""
            pass
        def __init__(self,
            *,
            size: typing.Optional[global___Vector3] = ...,
            position: typing.Optional[global___Vector3] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["position",b"position","size",b"size"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["position",b"position","size",b"size"]) -> None: ...

    class Sphere(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        RADIUS_FIELD_NUMBER: builtins.int
        radius: builtins.float
        def __init__(self,
            *,
            radius: builtins.float = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["radius",b"radius"]) -> None: ...

    class Capsule(google.protobuf.message.Message):
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

    class Ellipsoid(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        RADIUS_FIELD_NUMBER: builtins.int
        @property
        def radius(self) -> global___Vector3:
            """The radius of the ellipsoid in each dimension."""
            pass
        def __init__(self,
            *,
            radius: typing.Optional[global___Vector3] = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["radius",b"radius"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["radius",b"radius"]) -> None: ...

    class Cylinder(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        RADIUS_FIELD_NUMBER: builtins.int
        LENGTH_FIELD_NUMBER: builtins.int
        radius: builtins.float
        """Radius of the cylinder"""

        length: builtins.float
        """Length of the cylinder"""

        def __init__(self,
            *,
            radius: builtins.float = ...,
            length: builtins.float = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["length",b"length","radius",b"radius"]) -> None: ...

    class Mesh(google.protobuf.message.Message):
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
    ELLIPSOID_FIELD_NUMBER: builtins.int
    CYLINDER_FIELD_NUMBER: builtins.int
    MESH_FIELD_NUMBER: builtins.int
    MATERIAL_FIELD_NUMBER: builtins.int
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
    def ellipsoid(self) -> global___Collider.Ellipsoid: ...
    @property
    def cylinder(self) -> global___Collider.Cylinder: ...
    @property
    def mesh(self) -> global___Collider.Mesh: ...
    @property
    def material(self) -> global___Material:
        """The material of the collider"""
        pass
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
        ellipsoid: typing.Optional[global___Collider.Ellipsoid] = ...,
        cylinder: typing.Optional[global___Collider.Cylinder] = ...,
        mesh: typing.Optional[global___Collider.Mesh] = ...,
        material: typing.Optional[global___Material] = ...,
        hidden: builtins.bool = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["box",b"box","capsule",b"capsule","cylinder",b"cylinder","ellipsoid",b"ellipsoid","material",b"material","mesh",b"mesh","plane",b"plane","position",b"position","rotation",b"rotation","sphere",b"sphere","type",b"type"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["box",b"box","capsule",b"capsule","cylinder",b"cylinder","ellipsoid",b"ellipsoid","hidden",b"hidden","material",b"material","mesh",b"mesh","plane",b"plane","position",b"position","rotation",b"rotation","sphere",b"sphere","type",b"type"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["type",b"type"]) -> typing.Optional[typing_extensions.Literal["box","plane","sphere","capsule","ellipsoid","cylinder","mesh"]]: ...
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
    MESH_GEOMETRIES_FIELD_NUMBER: builtins.int
    @property
    def bodies(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Body]:
        """All of the bodies in the system"""
        pass
    dt: builtins.float
    """Amount of time to simulate each step, in seconds"""

    @property
    def mesh_geometries(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___MeshGeometry]:
        """All of the mesh geometries in the system."""
        pass
    def __init__(self,
        *,
        bodies: typing.Optional[typing.Iterable[global___Body]] = ...,
        dt: builtins.float = ...,
        mesh_geometries: typing.Optional[typing.Iterable[global___MeshGeometry]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["bodies",b"bodies","dt",b"dt","mesh_geometries",b"mesh_geometries"]) -> None: ...
global___Config = Config
