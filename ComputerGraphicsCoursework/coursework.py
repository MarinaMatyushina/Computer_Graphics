import sys
import glm

from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QToolTip, QComboBox, QSlider, QLabel,
	QApplication, QHBoxLayout, QVBoxLayout, QDesktopWidget, QSizePolicy, QCheckBox)

from qglwidget import glWidget

from time import time

class MainWindow(QWidget):

	def __init__(self):
		super().__init__()

		self.initUI()

	def initUI(self):

		self.glwidget = glWidget(self)

		mainLayout = QVBoxLayout()
		mainLayout.addWidget(self.glwidget)

		self.setLayout(mainLayout)

		self.setWindowTitle('Computer Graphics')
		self.setMinimumSize(800, 800)
		self.center()
		self.show()


    #centering window on screen
	def center(self):
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def keyPressEvent(self, event):
		self.glwidget.keyboardCallBack(event)
		self.glwidget.updateGL()


def main():
	app = QApplication(sys.argv)
	window = MainWindow()

	sys.exit(app.exec_())


if __name__ == '__main__':
	main()