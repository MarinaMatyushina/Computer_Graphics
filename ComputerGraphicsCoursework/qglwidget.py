from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5.QtOpenGL import *
from PyQt5.QtCore import Qt, QPointF

from PyQt5.QtGui import *

import ctypes
import numpy as np
import os
import copy
from random import random
from math import pi, cos, sin, atan, sqrt, radians

from figures import (coub_vertices, floor_vertices, light_source_vertices, 
	sphere, cylinder, cylinder_cup, pyramid, cone, cone_cup, tetrahedron)


class glWidget(QGLWidget):

	def __init__(self, parent):

		QGLWidget.__init__(self, parent)
		#self.widget_width = 640
		#self.widget_height = 480
		#self.setFixedSize(self.widget_width, self.widget_height)

		self.rotationX = 0
		self.rotationY = 0
		self.rotationZ = 0

		self.is_pressed = False
		self.firstMouse = True
		self.isFlashLightMode = False

		self.yaw = -100.0
		self.pitch = -10.0
		self.scale = 5.0

		fx = cos(radians(self.yaw)) * cos(radians(self.pitch))
		fy = sin(radians(self.pitch))
		fz = sin(radians(self.yaw)) * cos(radians(self.pitch))
		self.cameraFront = QVector3D(fx, fy, fz)
		self.cameraPos = QVector3D(5.0, 4.0, 16.0)
		self.cameraUp = QVector3D(0.0, 1.0, 0.0)

		self.LightIntensity = QVector3D(1.0, 1.0, 1.0)
		self.LightPosition = QVector4D(-1.0, 7.0, 2.0, 0.0)

		self.projection_matrix = QMatrix4x4()
		self.projection_matrix.perspective(45.0, self.width()/self.height(), 0.1, 100.0)
		self.current_view = QMatrix4x4()
		self.current_model = QMatrix4x4()

		self.subdivision = 1
		self.radius = 0.8
		self.interleavedStride = 32 # must be 32
		self.buildVerticesFlat()


	def resizeGL(self, w, h):

		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glViewport(0, 0, w, h)
		glMatrixMode(GL_MODELVIEW)


	def paintGL(self):
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # очистка буферов


		glEnable(GL_DEPTH_TEST)

		if self.isFlashLightMode:
			self.shaderProgram.setUniformValue('light.position', QVector4D(self.cameraPos, 1.0))
		else:
			self.shaderProgram.setUniformValue('light.position', self.LightPosition)

		self.shaderProgram.setUniformValue('isFlashLightMode', self.isFlashLightMode)

		self.shaderProgram.setUniformValue('light.direction', self.cameraFront)

		self.shaderProgram.setUniformValue('ViewPosition', self.cameraPos)
		self.shaderProgram.setUniformValue('LightIntensity', self.LightIntensity)

		self.updateMatrices()

		self.drawLightSource()
		self.drawFloor()
		self.drawIcoSphere()
		self.drawSphere()
		self.drawCoub((-1.0, 1.05, 0.0), 0, 0, 0, 1.5, self.texId1)
		self.drawCylinder()
		self.drawCone((2.0, 0.3, 3.5), 0, 0.8)

		glDisable(GL_DEPTH_TEST)


	def initializeGL(self):
		glClearColor(0.53, 0.81, 0.98, 1.0)

		# VBO
		self._vertexBuffer = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, self._vertexBuffer)
		# IBO
		self._indexBuffer = glGenBuffers(1)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._indexBuffer)

		self.shaderProgram = self.initShaderProgram()

		self.attribVertexPosition = self.shaderProgram.attributeLocation("vertexPosition")
		self.attribVertexNormal = self.shaderProgram.attributeLocation("vertexNormal")
		self.attribVertexTexCoord = self.shaderProgram.attributeLocation("vertexTexCoord")
		self.attribvertexColor = self.shaderProgram.attributeLocation("vertexColor")

		glEnable( GL_TEXTURE_2D )
		self.texId1 = self.bindTexture(QPixmap("textures/red.jpg"))
		self.texId2 = self.bindTexture(QPixmap("textures/yellow.jpg"))
		self.texId3 = self.bindTexture(QPixmap("textures/purple.jpg"))
		self.texId4 = self.bindTexture(QPixmap("textures/blue.jpg"))
		self.texId5 = self.bindTexture(QPixmap("textures/green.jpg"))
		self.texId6 = self.bindTexture(QPixmap("textures/gray.jpg"))


	def mousePressEvent(self, event):
		self.lastPos = event.pos()
		self.is_pressed = True
   

	def mouseReleaseEvent(self, event):
		self.is_pressed = False


	def mouseMoveEvent(self, event):
		self.viewingCameraMode(event.x(), event.y())
		self.updateGL()


	def wheelEvent(self, event):
		self.LightPosition.setY(self.LightPosition.y() + event.angleDelta().y() / 120)
		self.updateGL()


	def viewingCameraMode(self, x, y):
		sensitivity = 0.15

		dx = (x - self.lastPos.x()) * sensitivity
		dy = (self.lastPos.y() - y) * sensitivity

		self.lastPos = QPointF(x, y)

		self.yaw   += dx
		self.pitch += dy

		if(self.pitch > 89.0):
		    self.pitch = 89.0
		if(self.pitch < -89.0):
		    self.pitch = -89.0

		self.cameraFront.setX( cos(radians(self.yaw)) * cos(radians(self.pitch)) )
		self.cameraFront.setY( sin(radians(self.pitch)) )
		self.cameraFront.setZ( sin(radians(self.yaw)) * cos(radians(self.pitch)) )
		self.cameraFront.normalize()




	def keyboardCallBack(self, event):
		cameraSpeed = 0.5
		print(event.key())
		if event.key() == ord('W') or event.key() == 1062:
			self.cameraPos += cameraSpeed * self.cameraFront
		if event.key() == ord('A') or event.key() == 1060:
			cross = QVector3D.crossProduct(self.cameraFront, self.cameraUp)
			cross.normalize()
			self.cameraPos -= cross * cameraSpeed
		if event.key() == ord('S') or event.key() == 1067:
			self.cameraPos -= cameraSpeed * self.cameraFront
		if event.key() == ord('D') or event.key() == 1042:
			cross = QVector3D.crossProduct(self.cameraFront, self.cameraUp)
			cross.normalize()
			self.cameraPos += cross * cameraSpeed
		if event.key() == 16777235: # ^
			self.LightPosition.setZ(self.LightPosition.z() - 1)
		if event.key() == 16777234: # <-
			self.LightPosition.setX(self.LightPosition.x() - 1)
		if event.key() == 16777237: # v
			self.LightPosition.setZ(self.LightPosition.z() + 1)
		if event.key() == 16777236: # ->
			self.LightPosition.setX(self.LightPosition.x() + 1)
		if event.key() == ord('F') or event.key() == 1040:
			self.isFlashLightMode = not self.isFlashLightMode
			


	def initShaderProgram(self):
		shaderProgram = QOpenGLShaderProgram()
		shaderProgram.addShaderFromSourceFile(QOpenGLShader.Vertex, "shader.vert")
		shaderProgram.addShaderFromSourceFile(QOpenGLShader.Fragment, "shader.frag")
		shaderProgram.link()
		shaderProgram.bind()

		#light
		shaderProgram.setUniformValue('light.ambient', QVector3D(0.2, 0.2, 0.2))
		shaderProgram.setUniformValue('light.diffuse', QVector3D(0.5, 0.5, 0.5))
		shaderProgram.setUniformValue('light.specular', QVector3D(1.0, 1.0, 1.0))
		shaderProgram.setUniformValue('light.constant', 1.0)
		shaderProgram.setUniformValue('light.linear', 0.014)
		shaderProgram.setUniformValue('light.quadratic', 0.0007)
		shaderProgram.setUniformValue('light.cutOff', cos(radians(12.5)))
		shaderProgram.setUniformValue('light.outerCutOff', cos(radians(17.5)))
        # material
		shaderProgram.setUniformValue('material.specular', QVector3D(0.5, 0.5, 0.5))
		shaderProgram.setUniformValue('material.shininess', 32.0)
		return shaderProgram

	def updateMatrices(self):

		view, model = QMatrix4x4(), QMatrix4x4()

		view.lookAt(self.cameraPos, self.cameraPos + self.cameraFront, self.cameraUp)

		model.rotate(self.rotationX, QVector3D(1.0, 0.0, 0.0))
		model.rotate(self.rotationY, QVector3D(0.0, 1.0, 0.0))
		model.rotate(self.rotationZ, QVector3D(0.0, 0.0, 1.0))

		self.current_view = view
		self.current_model = model

		self.setUniformMatrix(model, view)


	def getTextureId(self, filename):
		texId1 = bindTexture(QPixmap(filename))
		return texId1


	def setUniformMatrix(self, model, view):
		modelview = view * model
		#self.shaderProgram.setUniformValue("ModelViewMatrix", modelview)
		self.shaderProgram.setUniformValue("MVP", self.projection_matrix * modelview)
		self.shaderProgram.setUniformValue("NormalMatrix", model.normalMatrix())
		self.shaderProgram.setUniformValue("ModelMatrix", model)


	def drawCoub(self, translateXYZ, rotate_angleX, rotate_angleY, rotate_angleZ, scale, texID):
		self.shaderProgram.setUniformValue("isPhongModel", True)

		glBufferData(GL_ARRAY_BUFFER, np.array(coub_vertices, dtype='float32'), GL_STATIC_DRAW)

		self.setDefaultAttribPointers()
		self.enableAttributeArrays()

		model = copy.deepcopy(self.current_model)
		model.translate(*translateXYZ)
		model.rotate(rotate_angleX, QVector3D(1.0, 0.0, 0.0))
		model.rotate(rotate_angleY, QVector3D(0.0, 1.0, 0.0))
		model.rotate(rotate_angleZ, QVector3D(0.0, 0.0, 1.0))
		model.scale(scale)
		self.setUniformMatrix(model, self.current_view)

		glBindTexture(GL_TEXTURE_2D, texID)

		glDrawArrays(GL_TRIANGLES, 0, 12*3)

		self.disableAttributeArrays()

		self.setUniformMatrix(self.current_model, self.current_view)


	def drawSphere(self):
		self.shaderProgram.setUniformValue("isPhongModel", True)

		glBufferData(GL_ARRAY_BUFFER, np.array(sphere.interleaved_vertices, dtype='float32'), GL_STATIC_DRAW)
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, np.array(sphere.indicies), GL_STATIC_DRAW)

		self.setDefaultAttribPointers()
		self.enableAttributeArrays()
		model = copy.deepcopy(self.current_model)
		model.translate(2, 1.1, 0)
		model.scale(0.8)
		self.setUniformMatrix(model, self.current_view)
		glBindTexture(GL_TEXTURE_2D, self.texId4)
		glDrawElements(GL_TRIANGLE_STRIP, len(sphere.indicies), GL_UNSIGNED_INT, None)
		self.disableAttributeArrays()
		self.setUniformMatrix(self.current_model, self.current_view)


	def drawIcoSphere(self):
		self.shaderProgram.setUniformValue("isPhongModel", True)

		glBufferData(GL_ARRAY_BUFFER, np.array(self.interleavedVertices, dtype='float32'), GL_STATIC_DRAW)
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, np.array(self.indices), GL_STATIC_DRAW)

		self.setDefaultAttribPointers()
		self.enableAttributeArrays()
		model = copy.deepcopy(self.current_model)
		model.translate(5.0, 1.06, 0)
		self.setUniformMatrix(model, self.current_view)
		glBindTexture(GL_TEXTURE_2D, self.texId5)
		glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
		self.disableAttributeArrays()
		self.setUniformMatrix(self.current_model, self.current_view)


	def drawCylinder(self):
		self.shaderProgram.setUniformValue("isPhongModel", True)

		self.setDefaultAttribPointers()
		self.enableAttributeArrays()
		model = copy.deepcopy(self.current_model)
		model.translate(-1.0, 1.26, 3.5)
		model.rotate(90, QVector3D(0.0, 0.0, 1.0))
		model.scale(0.8)
		self.setUniformMatrix(model, self.current_view)
		glBindTexture(GL_TEXTURE_2D, self.texId2)
		glBufferData(GL_ARRAY_BUFFER, np.array(cylinder, dtype='float32'), GL_STATIC_DRAW)
		glDrawArrays(GL_QUADS, 0, 3 * 6 * 1000)

		model.translate(0.2, 0.0, 0.0)
		self.setUniformMatrix(model, self.current_view)
		glBufferData(GL_ARRAY_BUFFER, np.array(cylinder_cup, dtype='float32'), GL_STATIC_DRAW)
		glDrawArrays(GL_TRIANGLE_FAN, 0, 3 * 1000);

		model.rotate(180, QVector3D(0.0, 0.0, 1.0))
		model.translate(0.4, 0.0, 0.0)
		self.setUniformMatrix(model, self.current_view)
		glDrawArrays(GL_TRIANGLE_FAN, 0, 3 * 1000);

		self.disableAttributeArrays()
		self.setUniformMatrix(self.current_model, self.current_view)


	def drawCone(self, translateXYZ, rotate_angleZ, scale):
		self.shaderProgram.setUniformValue("isPhongModel", True)

		self.setDefaultAttribPointers()
		self.enableAttributeArrays()
		model = copy.deepcopy(self.current_model)
		model.translate(*translateXYZ)
		model.rotate(rotate_angleZ, QVector3D(0.0, 0.0, 1.0))
		model.scale(scale)
		
		self.setUniformMatrix(model, self.current_view)
		glBindTexture(GL_TEXTURE_2D, self.texId3)

		glBufferData(GL_ARRAY_BUFFER, np.array(cone, dtype='float32'), GL_STATIC_DRAW)
		glDrawArrays(GL_TRIANGLES, 0, 3 * 360)

		model.rotate(-90, QVector3D(0.0, 0.0, 1.0))
		model.translate(-1.0, 0.0, 0)
		self.setUniformMatrix(model, self.current_view)
		glBufferData(GL_ARRAY_BUFFER, np.array(cone_cup, dtype='float32'), GL_STATIC_DRAW)
		glDrawArrays(GL_TRIANGLE_FAN, 0, 3 * 1000)

		self.disableAttributeArrays()
		self.setUniformMatrix(self.current_model, self.current_view)	


	def drawAsix(self):
		self.shaderProgram.setUniformValue("isPhongModel", False)
		axis_coords = [0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 10]
		
		glBufferData(GL_ARRAY_BUFFER, np.array(axis_coords, dtype='float32'), GL_STATIC_DRAW)

		glVertexAttribPointer(self.attribVertexPosition, 3, GL_FLOAT, GL_FALSE, 0, None)

		# set identity model matrix before drawing
		self.setUniformMatrix(QMatrix4x4(), self.current_view)

		self.shaderProgram.enableAttributeArray("vertexPosition")
		glDrawArrays(GL_LINES, 0, 6)
		self.shaderProgram.disableAttributeArray("vertexPosition")

		self.setUniformMatrix(self.current_model, self.current_view)


	def drawFloor(self):
		self.shaderProgram.setUniformValue('isPhongModel', True)
		glBufferData(GL_ARRAY_BUFFER, np.array(floor_vertices, dtype='float32'), GL_STATIC_DRAW)
		self.setDefaultAttribPointers()
		self.enableAttributeArrays()
		glBindTexture(GL_TEXTURE_2D, self.texId6)
		glDrawArrays(GL_TRIANGLES, 0, 12*3)
		self.disableAttributeArrays()


	def drawLightSource(self):
		self.shaderProgram.setUniformValue('isPhongModel', False)
		self.shaderProgram.setUniformValue('isLightSource', True)

		glBufferData(GL_ARRAY_BUFFER, np.array(light_source_vertices, dtype='float32'), GL_STATIC_DRAW)

		glVertexAttribPointer(self.attribVertexPosition, 3, GL_FLOAT, GL_FALSE, 8 * ctypes.sizeof(ctypes.c_float), None)
	
		self.shaderProgram.enableAttributeArray(self.attribVertexPosition)

		model = QMatrix4x4()
		model.translate(QVector3D(self.LightPosition))
		self.setUniformMatrix(model, self.current_view)

		glDrawArrays(GL_TRIANGLES, 0, 12*3)

		self.shaderProgram.disableAttributeArray(self.attribVertexPosition)

		self.setUniformMatrix(self.current_model, self.current_view)
		self.shaderProgram.setUniformValue('isLightSource', False)


	def enableAttributeArrays(self):
		self.shaderProgram.enableAttributeArray(self.attribVertexPosition)
		self.shaderProgram.enableAttributeArray(self.attribVertexTexCoord)
		self.shaderProgram.enableAttributeArray(self.attribVertexNormal)

	def disableAttributeArrays(self):
		self.shaderProgram.disableAttributeArray(self.attribVertexPosition)
		self.shaderProgram.disableAttributeArray(self.attribVertexTexCoord)
		self.shaderProgram.disableAttributeArray(self.attribVertexNormal)

	def setDefaultAttribPointers(self):
		stride = 8 * ctypes.sizeof(ctypes.c_float)
		glVertexAttribPointer(self.attribVertexPosition, 3, GL_FLOAT, GL_FALSE, stride, None)
		glVertexAttribPointer(self.attribVertexNormal, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * ctypes.sizeof(ctypes.c_float)))
		glVertexAttribPointer(self.attribVertexTexCoord, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * ctypes.sizeof(ctypes.c_float)))


	def computeIcosahedronVertices(self, radius):
		H_ANGLE = pi / 180 * 72 # 72 degree = 360 / 5
		V_ANGLE = atan(1/2) # elevation = 26.565 degree
		
		vertices = [0] * 12 * 3 # array of 12 vertices (x,y,z)
		i1 = i2 = 0 # indices
		z = xy = 0 # coords
		hAngle1 = -pi / 2 - H_ANGLE / 2 # start from -126 deg at 1st row
		hAngle2 = -pi / 2 # start from -90 deg at 2nd row

		# the first top vertex at (0, 0, r)
		vertices[0] = 0
		vertices[1] = 0
		vertices[2] = radius

		# compute 10 vertices at 1st and 2nd rows
		for i in range(1, 6):
			i1 = i * 3         # index for 1st row
			i2 = (i + 5) * 3   # index for 2nd row

			z  = radius * sin(V_ANGLE)            # elevaton
			xy = radius * cos(V_ANGLE)            # length on XY plane

			vertices[i1] = xy * cos(hAngle1)      # x
			vertices[i2] = xy * cos(hAngle2)
			vertices[i1 + 1] = xy * sin(hAngle1)  # y
			vertices[i2 + 1] = xy * sin(hAngle2)
			vertices[i1 + 2] = z;                  # z
			vertices[i2 + 2] = -z

			# next horizontal angles
			hAngle1 += H_ANGLE
			hAngle2 += H_ANGLE

		# the last bottom vertex at (0, 0, -r)
		i1 = 11 * 3
		vertices[i1] = 0
		vertices[i1 + 1] = 0
		vertices[i1 + 2] = -radius

		return vertices
	        
	    
	def buildVerticesFlat(self):

		S_STEP = 186 / 2048     # horizontal texture step
		T_STEP = 322 / 1024     # vertical texture step

		# compute 12 vertices of icosahedron
		tmpVertices = self.computeIcosahedronVertices(self.radius)

		# clear memory of prev arrays
		self.vertices = []
		self.normals = []
		self.texCoords = []
		self.indices = []
		self.lineIndices = []

		v0 = v1 = v2 = v3 = v4 = v11 = 0	# vertex positions
		n = [0] * 3                         # face normal
		
		# texture coords
		t0 = []
		t1 = []
		t2 = []
		t3 = []
		t4 = []
		t11 = []

		index = 0 							# for indecies
		ind = 0								# for [ind]

		# compute and add 20 tiangles of icosahedron first
		v0 = tmpVertices[0:3]  # 1st vertex
		ind = 11 * 3
		v11 = tmpVertices[ind : ind + 3] # 12th vertex
		for i in range(1, 6):
			# 4 vertices in the 2nd row
			ind = i * 3
			v1 = tmpVertices[ind : ind + 3]
			if i < 5:
				ind = (i + 1) * 3
				v2 = tmpVertices[ind : ind + 3]
				ind = (i + 6) * 3
				v4 = tmpVertices[ind : ind + 3]
			else:
				v2 = tmpVertices[3:6]
				v4 = tmpVertices[6 * 3 : 6 * 3 + 3]
			ind = (i + 5) * 3
			v3 = tmpVertices[ind : ind + 3]

			# texture coords
			t0.extend([(2 * i - 1) * S_STEP, 0])
			t1.extend([(2 * i - 2) * S_STEP, T_STEP])
			t2.extend([(2 * i - 0) * S_STEP, T_STEP])
			t3.extend([(2 * i - 1) * S_STEP, T_STEP * 2])
			t4.extend([(2 * i + 1) * S_STEP, T_STEP * 2])
			t11.extend([2 * i  * S_STEP, T_STEP * 3])
			

			# add a triangle in 1st row
			n = self.computeFaceNormal(v0, v1, v2, n)
			self.addVertices(v0, v1, v2)
			self.addNormals(n, n, n)
			self.addTexCoords(t0, t1, t2)
			self.addIndices(index, index+1, index+2)

			# add 2 triangles in 2nd row
			n = self.computeFaceNormal(v1, v3, v2, n)
			self.addVertices(v1, v3, v2)
			self.addNormals(n, n, n)
			self.addTexCoords(t1, t3, t2)
			self.addIndices(index+3, index+4, index+5)

			n = self.computeFaceNormal(v2, v3, v4, n)
			self.addVertices(v2, v3, v4)
			self.addNormals(n, n, n)
			self.addTexCoords(t2, t3, t4)
			self.addIndices(index+6, index+7, index+8)

			# add a triangle in 3rd row
			n = self.computeFaceNormal(v3, v11, v4, n)
			self.addVertices(v3, v11, v4)
			self.addNormals(n, n, n)
			self.addTexCoords(t3, t11, t4)
			self.addIndices(index+9, index+10, index+11)

			self.lineIndices.append(index)			# (i, i+1)
			self.lineIndices.append(index+1)		# (i, i+1)
			self.lineIndices.append(index+3)		# (i+3, i+4)
			self.lineIndices.append(index+4)
			self.lineIndices.append(index+3)		# (i+3, i+5)
			self.lineIndices.append(index+5)
			self.lineIndices.append(index+4)		# (i+4, i+5)
			self.lineIndices.append(index+5)
			self.lineIndices.append(index+9)		# (i+9, i+10)
			self.lineIndices.append(index+10)
			self.lineIndices.append(index+9)		# (i+9, i+11)
			self.lineIndices.append(index+11)

			# next index
			index += 12

		# subdivide icosahedron
		self.subdivideVerticesFlat()

		# generate interleaved vertex array as well
		self.buildInterleavedVertices()


	# divide a trinage into 4 sub triangles and repeat N times
	# If subdivision=0, do nothing.
	def subdivideVerticesFlat(self):

	    # new vertex positions
	    newV1 = [0] * 3
	    newV2 = [0] * 3
	    newV3 = [0] * 3

	    # new texture coords
	    newT1 = [0] * 3
	    newT2 = [0] * 3
	    newT3 = [0] * 3

	    normal = [0] * 3	# new face normal

	    # iteration
	    for i in range(1, self.subdivision + 1):
	        # copy prev arrays
	        tmpVertices = self.vertices
	        tmpIndices = self.indices
	        tmpTexCoords = self.texCoords

	        # clear prev arrays
	        self.vertices = []
	        self.normals = []
	        self.texCoords = []
	        self.indices = []
	        self.lineIndices = []

	        index = 0 # new index value
	        ind = 0 # for [ind]
	        indexCount = len(tmpIndices)
	        for j in range(0, indexCount, 3):
	            # get 3 vertice and texcoords of a triangle
	            ind = tmpIndices[j]
	            v1 = tmpVertices[ind * 3 : ind * 3 + 3]
	            t1 = tmpTexCoords[ind * 2 : ind * 2 + 2]
	            ind = tmpIndices[j + 1]
	            v2 = tmpVertices[ind * 3 : ind * 3 + 3]
	            t2 = tmpTexCoords[ind * 2 : ind * 2 + 2]
	            ind = tmpIndices[j + 2]
	            v3 = tmpVertices[ind * 3 : ind * 3 + 3]
	            t3 = tmpTexCoords[ind * 2 : ind * 2 + 2]

	            # get 3 new vertices by spliting half on each edge
	            newV1 = self.computeHalfVertex(v1, v2, self.radius, newV1)
	            newV2 = self.computeHalfVertex(v2, v3, self.radius, newV2)
	            newV3 = self.computeHalfVertex(v1, v3, self.radius, newV3)

	            newT1 = self.computeHalfTexCoord(t1, t2, newT1)
	            newT2 = self.computeHalfTexCoord(t2, t3, newT2)
	            newT3 = self.computeHalfTexCoord(t1, t3, newT3)


	            # add 4 new triangles
	            self.addVertices(v1, newV1, newV3)
	            self.addTexCoords(t1, newT1, newT3)
	            normal = self.computeFaceNormal(v1, newV1, newV3, normal)
	            self.addNormals(normal, normal, normal)
	            self.addIndices(index, index+1, index+2)

	            self.addVertices(newV1, v2, newV2)
	            self.addTexCoords(newT1, t2, newT2)
	            normal = self.computeFaceNormal(newV1, v2, newV2, normal)
	            self.addNormals(normal, normal, normal)
	            self.addIndices(index+3, index+4, index+5)

	            self.addVertices(newV1, newV2, newV3)
	            self.addTexCoords(newT1, newT2, newT3)
	            normal = self.computeFaceNormal(newV1, newV2, newV3, normal)
	            self.addNormals(normal, normal, normal)
	            self.addIndices(index+6, index+7, index+8)

	            self.addVertices(newV3, newV2, v3)
	            self.addTexCoords(newT3, newT2, t3)
	            normal = self.computeFaceNormal(newV3, newV2, v3, normal)
	            self.addNormals(normal, normal, normal)
	            self.addIndices(index+9, index+10, index+11)

	            # add new line indices per iteration
	            self.addSubLineIndices(index, index+1, index+4, index+5, index+11, index+9) #CCW

	            # next index
	            index += 12


	def addNormal(self, nx, ny, nz):
		self.normals.extend([nx, ny, nz])


	def addNormals(self, n1, n2, n3):
		self.normals.extend(n1[0:3])
		self.normals.extend(n2[0:3])
		self.normals.extend(n3[0:3])


	def addVertex(self, x, y, z):
		self.vertices.extend([x, y, z])


	def addVertices(self, v1, v2, v3):
		self.vertices.extend(v1[0:3])
		self.vertices.extend(v2[0:3])
		self.vertices.extend(v3[0:3])


	def addTexCoord(self, s, t):
		self.texCoords.extend([s, t])


	def addTexCoords(self, t1, t2, t3):
		self.texCoords.extend(t1[0:2])
		self.texCoords.extend(t2[0:2])
		self.texCoords.extend(t3[0:2])


	def addIndices(self, i1, i2, i3):
		self.indices.extend([i1, i2, i3])


	def computeScaleForLength(self, v, length):
	    # normalize the vector then re-scale to new radius
	    return length / sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


	def computeHalfVertex(self, v1, v2, length, newV):
	    newV[0] = v1[0] + v2[0]
	    newV[1] = v1[1] + v2[1]
	    newV[2] = v1[2] + v2[2]
	    scale = self.computeScaleForLength(newV, length);
	    newV[0] *= scale
	    newV[1] *= scale
	    newV[2] *= scale
	    return newV


	def computeHalfTexCoord(self, t1, t2, newT):
		newT[0] = (t1[0] + t2[0]) * 0.5
		newT[1] = (t1[1] + t2[1]) * 0.5
		return newT

	
	def buildInterleavedVertices(self):
		self.interleavedVertices = []
		count = len(self.vertices)
		for i, j in zip(range(0, count, 3), range(0, count, 2)):
			self.interleavedVertices.extend(self.vertices[i : i + 3])
			self.interleavedVertices.extend(self.normals[i : i + 3])
			self.interleavedVertices.extend(self.texCoords[j : j + 2])


	def addSubLineIndices(self, i1, i2, i3, i4, i5, i6):
		self.lineIndices.extend([i1, i2, i2, i6, i2, i3, i2, i4, 
			i6, i4, i3, i4, i4, i5])


	def computeFaceNormal(self, v1, v2, v3, n):

	    EPSILON = 0.000001

	    # default return value (0, 0, 0)
	    n[0] = n[1] = n[2] = 0

	    # find 2 edge vectors: v1-v2, v1-v3
	    ex1 = v2[0] - v1[0]
	    ey1 = v2[1] - v1[1]
	    ez1 = v2[2] - v1[2]
	    ex2 = v3[0] - v1[0]
	    ey2 = v3[1] - v1[1]
	    ez2 = v3[2] - v1[2]

	    # cross product: e1 x e2
	    nx = ny = nz = 0.0
	    nx = ey1 * ez2 - ez1 * ey2
	    ny = ez1 * ex2 - ex1 * ez2
	    nz = ex1 * ey2 - ey1 * ex2

	    # normalize only if the length is > 0
	    length = sqrt(nx * nx + ny * ny + nz * nz)
	    if (length > EPSILON):
	        # normalize
	        lengthInv = 1.0 / length
	        n[0] = nx * lengthInv
	        n[1] = ny * lengthInv
	        n[2] = nz * lengthInv

	    return n