from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtWidgets import QPushButton, QSlider
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog, QTextEdit

class View(QMainWindow):

    def __init__(self):
        super(View, self).__init__()

        self.setWindowTitle('CV')
        self.setFixedSize(600, 600)

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
        
        self.generalLayout.addLayout(leftBar, stretch=3)

    def _createMidBar(self):
        midBar = QVBoxLayout()

        self.MLButton = QPushButton("machine learning")
        self.MLButton.clicked.connect(self._MLDetect)
        self.MLButton.setEnabled(False)
        midBar.addWidget(self.MLButton)

        self.TRButton = QPushButton("traditional approach")
        self.TRButton.clicked.connect(self._TRDetect)
        self.TRButton.setEnabled(False)
        midBar.addWidget(self.TRButton)

        self.generalLayout.addLayout(midBar, stretch=1)

    def _createRightBar(self):
        rightBar = QVBoxLayout()

        self.MLResult = QLabel()
        rightBar.addWidget(self.MLResult)

        self.TRUpResult = QLabel()
        rightBar.addWidget(self.TRUpResult)
        self.TRDownResult = QLabel()
        rightBar.addWidget(self.TRDownResult)
        self.TRSideResult = QLabel()
        rightBar.addWidget(self.TRSideResult)

        self.generalLayout.addLayout(rightBar, stretch=3)

    def _openFileDialog(self):
        self.fileName, self.fileType = QFileDialog.getOpenFileName(self, "Open Image", "yolo-image", "Image Files (*.png *.jpg *.bmp)")
        self.rawImage.setPixmap(QPixmap(self.fileName))
        self.MLButton.setEnabled(True)
        self.TRButton.setEnabled(True)

    def _MLDetect(self):
        self._MLDetectCallback()

    def _MLDetectCallback(self):
        resultFilename = self.fileName.replace("yolo-image", "yolo-result")
        print(resultFilename)
        self.MLResult.setPixmap(QPixmap(resultFilename))

    def _TRDetect(self):
        upStr = "upstr"
        downStr = "downStr"
        sideStr = "sideStr"
        self.TRUpResult.setText(upStr)
        self.TRDownResult.setText(downStr)
        self.TRSideResult.setText(sideStr)