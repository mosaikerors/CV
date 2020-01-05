from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtWidgets import QPushButton, QSlider
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog, QTextEdit
import cv2
import os
from ml_method import detect
from naive_method import my_bottle


class View(QMainWindow):

    def __init__(self):
        super(View, self).__init__()

        self.setWindowTitle('CV')
        self.setFixedSize(1300, 800)

        self.generalLayout = QHBoxLayout()
        self._centralWidget = QWidget(self)
        self.setCentralWidget(self._centralWidget)
        self._centralWidget.setLayout(self.generalLayout)

        self.hasRawImage = False

        self._createLeftBar()
        self._createMidBar()
        self._createRightBar()

    def _createLeftBar(self):
        leftBar = QVBoxLayout()

        self.fileDialogButton = QPushButton("select raw image")
        self.fileDialogButton.clicked.connect(self._openFileDialog)
        leftBar.addWidget(self.fileDialogButton)

        self.rawImage = QLabel()
        leftBar.addWidget(self.rawImage)

        self.generalLayout.addLayout(leftBar, stretch=1)

    def _createMidBar(self):
        rightBar = QVBoxLayout()

        self.TRButton = QPushButton("traditional approach")
        self.TRButton.clicked.connect(self._TRDetect)
        self.TRButton.setEnabled(False)
        rightBar.addWidget(self.TRButton)

        self.TRResult = QLabel()
        rightBar.addWidget(self.TRResult)

        self.generalLayout.addLayout(rightBar, stretch=1)

    def _createRightBar(self):
        midBar = QVBoxLayout()

        self.MLButton = QPushButton("machine learning")
        self.MLButton.clicked.connect(self._MLDetect)
        self.MLButton.setEnabled(False)
        midBar.addWidget(self.MLButton)

        self.MLResult = QLabel()
        midBar.addWidget(self.MLResult)

        self.generalLayout.addLayout(midBar, stretch=1)

    def _openFileDialog(self):
        self.fileName, self.fileType = QFileDialog.getOpenFileName(self, "Open Image", ".",
                                                                   "Image Files (*.png *.jpg *.bmp)")
        pixmap = QPixmap(self.fileName).scaled(QSize(400, 400), aspectRatioMode=Qt.KeepAspectRatio)
        self.rawImage.setPixmap(pixmap)
        self.rawImage.setAlignment(Qt.AlignCenter)
        self.MLButton.setEnabled(True)
        self.TRButton.setEnabled(True)

    def _MLDetect(self):
        tmpImage = cv2.imread(self.fileName)
        pos = self.fileName.rfind("/")
        self.yoloFilename = self.fileName[:pos] + "/yolo-image" + self.fileName[pos:]
        cv2.imwrite(self.yoloFilename, tmpImage)
        detect(self.fileName[:pos] + "/yolo-image", self.fileName[:pos] + "/yolo-result")
        self._MLDetectCallback()

    def _MLDetectCallback(self):
        resultFilename = self.yoloFilename.replace("yolo-image", "yolo-result")
        self.MLResult.setPixmap(QPixmap(resultFilename).scaled(QSize(400, 400), aspectRatioMode=Qt.KeepAspectRatio))
        self.MLResult.setAlignment(Qt.AlignCenter)
        os.remove(self.yoloFilename)
        os.remove(resultFilename)

    def _TRDetect(self):
        outfile, upData, downData, sideData = my_bottle(self.fileName)

        self.TRResult.setPixmap(QPixmap(outfile).scaled(QSize(400, 400), aspectRatioMode=Qt.KeepAspectRatio))
        self.TRResult.setAlignment(Qt.AlignCenter)
