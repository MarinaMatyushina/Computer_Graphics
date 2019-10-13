from math import pi, cos, sin, atan, sqrt, radians
from PyQt5.QtGui import QVector3D

def getTriangleNormal(p1, p2, p3):
    v1 = QVector3D(*p3) - QVector3D(*p1)
    v2 = QVector3D(*p2) - QVector3D(*p1)
    normal = QVector3D.crossProduct(v1, v2)
    normal.normalize()
    return normal.x(), normal.y(), normal.z()

def genCuboid(x, y, z):
    return [
        -x, -y, -z,  0.0, 0.0, -1.0,  0.0, 0.0, # coords, normals, tex coords
         x, -y, -z,  0.0, 0.0, -1.0,  1.0, 0.0,
         x,  y, -z,  0.0, 0.0, -1.0,  1.0, 1.0,
         x,  y, -z,  0.0, 0.0, -1.0,  1.0, 1.0,
        -x,  y, -z,  0.0, 0.0, -1.0,  0.0, 1.0,
        -x, -y, -z,  0.0, 0.0, -1.0,  0.0, 0.0,

        -x, -y,  z,  0.0, 0.0, 1.0,  0.0, 0.0,
         x, -y,  z,  0.0, 0.0, 1.0,  1.0, 0.0,
         x,  y,  z,  0.0, 0.0, 1.0,  1.0, 1.0,
         x,  y,  z,  0.0, 0.0, 1.0,  1.0, 1.0,
        -x,  y,  z,  0.0, 0.0, 1.0,  0.0, 1.0,
        -x, -y,  z,  0.0, 0.0, 1.0,  0.0, 0.0,

        -x,  y,  z,  -1.0, 0.0, 0.0,  1.0, 0.0,
        -x,  y, -z,  -1.0, 0.0, 0.0,  1.0, 1.0,
        -x, -y, -z,  -1.0, 0.0, 0.0,  0.0, 1.0,
        -x, -y, -z,  -1.0, 0.0, 0.0,  0.0, 1.0,
        -x, -y,  z,  -1.0, 0.0, 0.0,  0.0, 0.0,
        -x,  y,  z,  -1.0, 0.0, 0.0,  1.0, 0.0,

         x,  y,  z,  1.0, 0.0, 0.0,  1.0, 0.0,
         x,  y, -z,  1.0, 0.0, 0.0,  1.0, 1.0,
         x, -y, -z,  1.0, 0.0, 0.0,  0.0, 1.0,
         x, -y, -z,  1.0, 0.0, 0.0,  0.0, 1.0,
         x, -y,  z,  1.0, 0.0, 0.0,  0.0, 0.0,
         x,  y,  z,  1.0, 0.0, 0.0,  1.0, 0.0,

        -x, -y, -z,  0.0, -1.0, 0.0,  0.0, 1.0,
         x, -y, -z,  0.0, -1.0, 0.0,  1.0, 1.0,
         x, -y,  z,  0.0, -1.0, 0.0,  1.0, 0.0,
         x, -y,  z,  0.0, -1.0, 0.0,  1.0, 0.0,
        -x, -y,  z,  0.0, -1.0, 0.0,  0.0, 0.0,
        -x, -y, -z,  0.0, -1.0, 0.0,  0.0, 1.0,

        -x,  y, -z,  0.0, 1.0, 0.0,  0.0, 1.0,
         x,  y, -z,  0.0, 1.0, 0.0,  1.0, 1.0,
         x,  y,  z,  0.0, 1.0, 0.0,  1.0, 0.0,
         x,  y,  z,  0.0, 1.0, 0.0,  1.0, 0.0,
        -x,  y,  z,  0.0, 1.0, 0.0,  0.0, 0.0,
        -x,  y, -z,  0.0, 1.0, 0.0,  0.0, 1.0 ]


def genPyramid(height):
    p1, p2, p3, p4 = (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (-1.0, -1.0, -1.0), (1.0, -1.0, -1.0)
    p5 = (0.0, height, 0.0)
    front_normal = getTriangleNormal((p5), (p2), (p1))
    right_normal = getTriangleNormal((p5), (p4), (p2))
    back_normal = getTriangleNormal((p5), (p3), (p4))
    left_normal = getTriangleNormal((p5), (p1), (p3))

    return [

        *p5,    *front_normal,  0.0, 0.0, # coords, normals, tex coords
        *p1,    *front_normal,  1.0, 0.0,
        *p2,    *front_normal,  1.0, 1.0,

        *p5,    *right_normal,  0.0, 0.0,
        *p4,    *right_normal,  1.0, 0.0,
        *p2,    *right_normal,  1.0, 1.0,

        *p5,    *back_normal,  1.0, 0.0,
        *p4,    *back_normal,  1.0, 1.0,
        *p3,    *back_normal,  0.0, 1.0,

        *p5,    *left_normal,  1.0, 0.0,
        *p1,    *left_normal,  1.0, 1.0,
        *p3,    *left_normal,  0.0, 1.0,

        *p1,    0.0, -1.0, 0.0,  1.0, 0.0,
        *p2,    0.0, -1.0, 0.0,  1.0, 1.0,
        *p3,    0.0, -1.0, 0.0,  0.0, 1.0,
        *p3,    0.0, -1.0, 0.0,  0.0, 1.0,
        *p2,    0.0, -1.0, 0.0,  1.0, 1.0,
        *p4,    0.0, -1.0, 0.0,  1.0, 0.0]


def genTetrahedron(height):
    p1, p2, p3 = (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (1.0, -1.0, -1.0)
    p4 = (0.0, height, 0.0)
    front_normal = getTriangleNormal((p4), (p2), (p1))
    right_normal = getTriangleNormal((p4), (p3), (p2))
    left_normal = getTriangleNormal((p4), (p1), (p3))
    bottom_normal = getTriangleNormal((p1), (p2), (p3))

    return [
        *p4,    *front_normal,  0.0, 0.0, # coords, normals, tex coords
        *p1,    *front_normal,  1.0, 0.0,
        *p2,    *front_normal,  1.0, 1.0,

        *p4,    *right_normal,  0.0, 0.0,
        *p2,    *right_normal,  1.0, 0.0,
        *p3,    *right_normal,  1.0, 1.0,

        *p4,    *left_normal,  1.0, 0.0,
        *p1,    *left_normal,  1.0, 1.0,
        *p3,    *left_normal,  0.0, 1.0,

        *p1,    *bottom_normal,  1.0, 0.0,
        *p2,    *bottom_normal,  1.0, 1.0,
        *p3,    *bottom_normal,  0.0, 1.0]


class Sphere:
    def __init__(self, slices, stacks):
        self.Tesselate(slices, stacks)
        self.buildInterleavedVertices()

    def Tesselate(self, slices, stacks):
        self.clearArrays()
        for ci in range(slices):
            u = ci / (slices - 1)
            for ri in range(stacks):
                v = ri / (stacks - 1)
                vertXYZ = self.getPositionOnSphere(u, v)
                self.vertices.extend(vertXYZ)
                self.normals.extend(vertXYZ)
                self.tex_coords.extend([1.0 - u, v])
        self.calculateTriangleStripindicies(slices, stacks)

    def getPositionOnSphere(self, u, v):
        radius = 1.0
        latitude = pi * (1.0 - v)
        longitude = 2.0 * pi * u
        latitudeRadius = radius * sin(latitude)

        return (
            cos(longitude) * latitudeRadius,
            cos(latitude) * radius,
            sin(longitude) * latitudeRadius
        )

    def calculateTriangleStripindicies(self, column_count, row_count):
        for ci in range(column_count - 1):
            if ci % 2 == 0:
                for ri in range(row_count):
                    index = ci * row_count + ri
                    self.indicies.append(index + row_count)
                    self.indicies.append(index)
            else:
                for ri in range(row_count - 1, -1, -1):
                    index = ci * row_count + ri
                    self.indicies.append(index)
                    self.indicies.append(index + row_count)

    def buildInterleavedVertices(self):
        self.interleaved_vertices = []
        count = len(self.vertices)
        for i, j in zip(range(0, count, 3), range(0, count, 2)):
            self.interleaved_vertices.extend(self.vertices[i : i + 3])
            self.interleaved_vertices.extend(self.normals[i : i + 3])
            self.interleaved_vertices.extend(self.tex_coords[j : j + 2])

    def clearArrays(self):
        self.vertices = []
        self.normals = []
        self.tex_coords = []
        self.indicies = []


def genCylinder(div, width, height):
    cylinder = []
    for i in range(div):
        y, z = width * sin(2 * pi * i / div), width * cos(2 * pi * i / div)
        cylinder.extend([-height, y, z]) # position
        cylinder.extend([0, y, z]) # normal
        cylinder.extend([0, i / div]) # texture

        y, z = width * sin(2 * pi * (i + 1) / div), width * cos(2 * pi * (i + 1) / div)
        cylinder.extend([-height, y, z]) # position
        cylinder.extend([0, y, z]) # normal
        cylinder.extend([0, (i + 1.0) / div]) # texture

        cylinder.extend([height, y, z]) # position
        cylinder.extend([0, y, z]) # normal
        cylinder.extend([1, (i + 1.0) / div]) # texture

        y, z = width * sin(2 * pi * i / div), width * cos(2 * pi * i / div)
        cylinder.extend([height, y, z]) # position
        cylinder.extend([0, y, z]) # normal
        cylinder.extend([1, (i / div)]) # texture

    return cylinder

def genCircleCup(div, width):
    circle = []
    for i in range(div):
        circle.extend([1, width * sin(2 * pi * i / div), width * cos(2 * pi * i / div)]) # position
        circle.extend([1.0, 0.0, 0.0]) # normal
        # texture
        if i <= (div / 4):
            circle.extend([0, 1 - i * 4 / div])
        elif (i >= div / 4) and (i < div / 2):
            circle.extend([(i - div / 4) * 4 / div, 0])
        elif (i >= div / 2) and (i < 3 * div / 4):
            circle.extend([1, (i - div / 2) * 4 / div])
        elif i >= (3 * div / 4):
            circle.extend([1, 1 - (i - 3 * div / 4) * 4 / div])
    return circle
    

cylinder = genCylinder(1000, 0.8, 1.2)
cylinder_cup = genCircleCup(1000, 0.8)

def genCone(div, height):
    cone = []
    for i in range(0, 360, div):
        p1, p2, p3 = [0.0, height, 0.0], [cos(radians(i)), 0.0, sin(radians(i))], [cos(radians(i + div)), 0.0, sin(radians(i + div))]

        normal = getTriangleNormal(p1, p2, p3)

        cone.extend(p1) # position
        cone.extend(normal) # normal
        cone.extend([0, (i + div) / 360]) # texture

        cone.extend(p2) # position
        cone.extend(normal) # normal
        cone.extend([1, i / 360]) # texture

        cone.extend(p3) # position
        cone.extend(normal) # normal
        cone.extend([1, (i + div) / 360]) # texture

    return cone


floor_vertices = genCuboid(10, 0.3, 10)
coub_vertices = genCuboid(0.5, 0.5, 0.5)
light_source_vertices = genCuboid(0.3, 0.3, 0.3)

sphere = Sphere(40, 40)

pyramid = genPyramid(1)
tetrahedron = genTetrahedron(1)

cone = genCone(1, 3)
cone_cup = genCircleCup(1000, 1)