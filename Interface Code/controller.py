from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow
import cv2
import math
import dlib
import time
import numpy as np
from UI import Ui_MainWindow as Ui_MainWindow

class MainWindow_controller(QMainWindow):
    def __init__(self, model, stress):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.model = model
        self.stress = stress
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_webcam()
        self.video = []
        self.c = 1


    def setup_webcam(self):
        self.ProcessCam = Camera()
        if self.ProcessCam.connect:
            self.debugBar('Connection!!!')
            self.ProcessCam.rawdata.connect(self.getRaw)
        else:
            self.debugBar('Disconnection!!!')

        self.ui.button_opencam.clicked.connect(self.openCam)
        self.ui.button_pred.clicked.connect(self.go_model)

        self.ui.button_exit.clicked.connect(self.exit)
        self.ui.button_restart.clicked.connect(self.restart)

    def exit(self):
        import sys
        sys.exit()

    windowlist = []
    def restart(self):
        window = HomePage(model=self.model, stress=self.stress)
        self.windowlist.append(window)
        self.close()
        window.show()

    #go model
    def go_model(self):
        self.stopCam()
        self.ui.button_pred.setEnabled(False)
        try:
            self.Pred = Predict(video=self.video, model=self.model, stress=self.stress)
            self.Pred.prediction.connect(self.result)
            self.Pred.start()
            self.debugBar('All images detected successfully!  Wait for Prediction')
        except:
            self.debugBar('No face detected!')

    def result(self, res):
        self.debugBar('Predict successfully!')

        res_index = np.argmax(res, axis=1)
        if (res_index[0]==1):
            level = 'Low'
        else:
            if res.flatten()[res_index[0]] - res.flatten()[res_index[1]] > 0.073742166:
                level = 'High'
            else:
                level = 'Medium'

        _translate = QtCore.QCoreApplication.translate
        self.ui.label_level.setText(_translate("MainWindow","     %s"%(level)))
        self.Pred.terminate()
        self.ui.button_exit.setVisible(False)
        self.ui.button_restart.setVisible(True)

    def getRaw(self, data):
        self.showData(data)
        fps = 15
        if (self.c % fps == 0):
            if len(self.video) < 15:
                self.video.append(data)
                self.debugBar('Error:Not enough data, 15 images are needed!')
            elif 15 <= len(self.video) < 30:
                self.video.append(data)
                self.ui.button_pred.setEnabled(True)
                self.debugBar('Enough data! Press Predict')
            else:
                del self.video[0]
                self.video.append(data)
                self.ui.button_pred.setEnabled(True)
                self.debugBar('Enough data! Press Predict')
        self.c += 1


    def openCam(self):
        if self.ProcessCam.connect:
            self.ProcessCam.open()
            self.ProcessCam.start()
            self.ui.button_opencam.setVisible(False)
            self.ui.button_pred.setVisible(True)

    def stopCam(self):
        if self.ProcessCam.connect:
            self.ProcessCam.stop()

    def showData(self, img):
        self.Ny, self.Nx, _ = img.shape
        roi_rate = 1

        # 反轉顏色
        img_new = np.zeros_like(img)
        img_new[..., 0] = img[..., 2]
        img_new[..., 1] = img[..., 1]
        img_new[..., 2] = img[..., 0]
        img = img_new

        # qimg = QtGui.QImage(img[:,:,0].copy().data, self.Nx, self.Ny, QtGui.QImage.Format_Indexed8)
        qimg = QtGui.QImage(img.data, self.Nx, self.Ny, QtGui.QImage.Format_RGB888)
        self.ui.label_videoframe.setScaledContents(True)
        self.ui.label_videoframe.setPixmap(QtGui.QPixmap.fromImage(qimg))

        self.ui.label_videoframe.setMinimumSize(self.Nx * roi_rate, self.Ny * roi_rate)
        self.ui.label_videoframe.setMaximumSize(self.Nx * roi_rate, self.Ny * roi_rate)

        def closeEvent(self, event):
            if self.ProcessCam.running:
                self.ProcessCam.close()
                self.ProcessCam.terminate()
            QtWidgets.QApplication.closeAllWindows()

        def keyPressEvent(self, event):
            if event.key() == QtCore.Qt.Key_Q:
                if self.ProcessCam.running:
                    self.ProcessCam.close()
                    time.sleep(1)
                    self.ProcessCam.terminate()
                QtWidgets.QApplication.closeAllWindows()


    def debugBar(self, msg):
        self.ui.statusBar.showMessage('         '+str(msg), 50000)


class Predict(QtCore.QThread):
    prediction = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, video, model, stress, parent=None):
        super().__init__(parent)
        self.stress = stress
        self.model = model
        self.video = video


    def run(self):
        data = []
        detector = dlib.get_frontal_face_detector()
        for frame in self.video:
            if len(data) < 15:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                face = detector(img, 1)

                for i, d in enumerate(face):
                    x1 = d.left()
                    y1 = d.top()
                    x2 = d.right()
                    y2 = d.bottom()
                    img2 = img[y1:y2, x1:x2]

                    try:
                        resized = cv2.resize(img2, (128, 128), cv2.INTER_LANCZOS4)  # 標準化圖片尺寸轉128*128
                        data.append(resized)
                    except:
                        print('resize error')
                        pass
            else:
                break

        input_1 = np.array(data)
        input_1 = np.expand_dims(input_1, axis=0)

        label_dict = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness', 4: 'anger', 5: 'disgust', 6: 'fear'}
        emonamelst = []
        plst = []
        for frame in self.video:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (48, 48))
            img = np.expand_dims(img, axis=0)  # makes image shape (1,48,48)
            img = img.reshape(1, 48, 48, 1)
            result = self.stress.predict(img)
            result = list(result[0])
            img_index = result.index(max(result))
            p = max(result)
            emoname = label_dict[img_index]
            emonamelst.append(emoname)
            plst.append(p)

        add = 0
        for i in range(15):
            if emonamelst[i] == 'surprise':
                add += 100 * self.surprise(plst[i])
            elif emonamelst[i] == 'sadness':
                add += 100 * self.sad(plst[i])
            elif emonamelst[i] == 'happiness':
                add += 100 * self.happiness(plst[i])
            elif emonamelst[i] == 'disgust':
                add += 100 * self.disgust(plst[i])
            elif emonamelst[i] == 'neutral':
                add += 100 * self.neutral(plst[i])
            elif emonamelst[i] == 'fear':
                add += 100 * self.fear(plst[i])
            elif emonamelst[i] == 'anger':
                add += 100 * self.anger(plst[i])
        input_2 = np.array(add / 15)
        input_2 = np.expand_dims(input_2, axis=0)
        y = self.model.predict([input_1, input_2])
        self.prediction.emit(y)


    def anger(self, p):
        t = 0.343 * p + 1.003
        return 2.332 * math.log(t)

    def fear(self, p):
        t = 1.356 * p + 1
        return 1.763 * math.log(t)

    def neutral(self, p):
        t = 0.01229 * p + 1.036
        return 5.03 * math.log(t)

    def disgust(self, p):
        t = 0.0123 * p + 1.019
        return 7.351 * math.log(t)

    def happiness(self, p):
        t = 5.221e-5 * p + 0.9997
        return 532.2 * math.log(t)

    def sad(self, p):
        t = 0.1328 * p + 1.009
        return 2.851 * math.log(t)

    def surprise(self, p):
        t = 0.2825 * p + 1.003
        return 2.478 * math.log(t)

class Camera(QtCore.QThread):
    rawdata = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if self.cam is None or not self.cam.isOpened():
            self.connect = False
            self.running = False
        else:
            self.connect = True
            self.running = False

    def run(self):
        while self.running and self.connect:
            ret, img = self.cam.read()
            if ret:
                self.rawdata.emit(img)
            else:
                print("Warning!!!")
                self.connect = False

    def open(self):
        if self.connect:
            self.running = True

    def stop(self):
        if self.connect:
            self.running = False

    def close(self):
        if self.connect:
            self.running = False
            time.sleep(1)
            self.cam.release()


from homepage import Ui_HomePage
class HomePage(QMainWindow):
    def __init__(self, model,stress):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.model = model
        self.stress = stress
        self.ui = Ui_HomePage()
        self.ui.setupUi(self)
        self.start()

    def start(self):
        self.ui.button_start.clicked.connect(self.go_main)

    windowlist = []
    def go_main(self):
        window = MainWindow_controller(self.model,self.stress)
        self.windowlist.append(window)
        self.close()
        window.show()
