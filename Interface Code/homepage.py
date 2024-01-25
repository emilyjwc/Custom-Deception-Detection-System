# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'day25.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_HomePage(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1024, 650)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.title = QtWidgets.QLabel(self.centralwidget)

        self.title.setGeometry(QtCore.QRect(402, 30, 800, 50))
        self.title.setFont(QtGui.QFont('DFKai-SB', 24))
        self.title.setObjectName("title")

        self.label_page = QtWidgets.QLabel(self.centralwidget)
        self.label_page.setGeometry(QtCore.QRect(212, 95, 600, 450))
        self.label_page.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.label_page.setObjectName("label_videoframe")
        self.label_page.setAlignment(QtCore.Qt.AlignCenter)

        self.button_start = QtWidgets.QPushButton(self.centralwidget)
        self.button_start.setGeometry(QtCore.QRect(432, 570, 160, 60))
        self.button_start.setFont(QtGui.QFont('DFKai-SB', 18))
        self.button_start.setObjectName("button_start")


        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "基於臉部表情辨識與壓力測量之海關測謊系統"))
        self.title.setText(_translate("MainWindow", "謊言偵測系統"))
        self.button_start.setText(_translate("MainWindow", "進入系統"))

        img = cv2.imread('C:/Users/Hung/Downloads/homepage.jpeg', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytesPerline = 3 * width
        qpixmap_fix_width = 800  # 16x9 = 1920x1080 = 1280x720 = 800x450
        qpixmap_fix_height = 450
        qimg = QtGui.QImage(img, width, height, bytesPerline, QtGui.QImage.Format_RGB888)
        qpixmap = QtGui.QPixmap.fromImage(qimg)
        if qpixmap.width() / 16 >= qpixmap.height() / 9:  # like 1600/16 > 90/9, height is shorter, align width
            qpixmap = qpixmap.scaledToWidth(qpixmap_fix_width)
        else:  # like 1600/16 < 9000/9, width is shorter, align height
            qpixmap = qpixmap.scaledToHeight(qpixmap_fix_height)
        self.label_page.setPixmap(qpixmap)
        self.label_page.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)  # Center



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_HomePage()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())