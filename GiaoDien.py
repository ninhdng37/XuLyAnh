from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QSlider
from PyQt5 import uic, QtCore
import sys
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
import faceRecognition as fr
from PIL import ImageQt

class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi('abc.ui', self)
        # self.hsGauss.setDisabled(True)
        # self.hsMedian.setDisabled(True)
        self.disabled()

        #Loc Nhieu
        self.actionGaussianBlur.triggered.connect(self.GaussianBlur)
        self.hsGauss.sliderReleased.connect(self.GaussianBlurChange)
        #QSlider.setDisabled(true)
        # QSlider.setEnabled(True)
        self.actionmedianBlur.triggered.connect(self.medianBlur)
        self.hsMedian.sliderReleased.connect(self.medianBlurChange)

        #Do bien, do canh
        self.actionCanny.triggered.connect(self.Canny)
        self.actionLaplacian.triggered.connect(self.Laplacian)
        self.actionSobelX.triggered.connect(self.SobelX)
        self.actionSobelY.triggered.connect(self.SobelY)
        self.actionSobelCombined.triggered.connect(self.SobelCombined)
        self.actionPrewitt.triggered.connect(self.Prewitt)

        #Face Detection
        self.btnFaceDetection.clicked.connect(self.detectFace)

        #Face Recognition
        self.btnFaceRecognition.clicked.connect(self.recognizeFace)

        #Save Image
        self.btnSaveImage.clicked.connect(self.saveImage)

        #Choose Image
        self.btnChonAnh.clicked.connect(self.open_img)
        self.btnReset.clicked.connect(self.reset)

        self.image = []
        self.tmp = []
        self.img = []
        self.show()

    def loadImage(self, fname):
        self.image = cv2.imread(fname)
        self.tmp = self.image
        self.displayImage()

    def disabled(self):
        self.hsGauss.setDisabled(True)
        self.hsMedian.setDisabled(True)

    def open_img(self):
        fname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'D:/XuLyAnh/BaiTapLon',
                                                    "*.jpg *.png")
        if fname:
            self.loadImage(fname)
            self.lblAnhXuLy.setPixmap(QPixmap())
        else:
            print("Invalid Image")

    def displayImage(self, window=1):
        qformat = QImage.Format_Indexed8

        if len(self.image.shape) == 3:
            if (self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        # image.shape[0] l?? s??? pixel theo chi???u Y
        # image.shape[1] l?? s??? pixel theo chi???u X
        # image.shape[2] l??u s??? channel bi???u th??? m???i pixel
        img = img.rgbSwapped()  # chuy???n ?????i hi???u qu??? m???t ???nh RGB th??nh m???t ???nh BGR.
        if window == 1:
            self.lblAnhGoc.setPixmap(QPixmap.fromImage(img).scaled(self.lblAnhGoc.width(), self.lblAnhGoc.height()))
            self.lblAnhGoc.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)  # c??n ch???nh v??? tr?? xu???t hi???n c???a h??nh tr??n lable
        if window == 2:
            self.lblAnhXuLy.setPixmap(QPixmap.fromImage(img).scaled(self.lblAnhGoc.width(), self.lblAnhGoc.height()))
            self.lblAnhXuLy.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def thongBao(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()

    def ImageWasChoosen(self):
        if len(self.tmp) != 0:
            self.disabled()
            return True
        else:
            self.thongBao('Ban chua chon anh')

    def GaussianBlur(self):
        if self.ImageWasChoosen():
            self.hsGauss.setEnabled(True)
            self.image = cv2.GaussianBlur(self.tmp, (5, 5), self.hsGauss.value())
            self.displayImage(2)

    def GaussianBlurChange(self):
        self.image = cv2.GaussianBlur(self.tmp, (5, 5), self.hsGauss.value())
        self.displayImage(2)

    def medianBlur(self):
        if self.ImageWasChoosen():
            self.hsMedian.setEnabled(True)
            a = self.hsMedian.value()
            if a % 2 == 0:
                a = a + 1
            self.image = cv2.medianBlur(self.tmp, a)
            self.displayImage(2)

    def medianBlurChange(self):
        # QSlider.sliderReleased
        a = self.hsMedian.value()
        if a % 2 == 0:
            a = a + 1
        self.image = cv2.medianBlur(self.tmp, a)
        self.displayImage(2)


    def Canny(self):
        if self.ImageWasChoosen():
            self.image = cv2.Canny(self.tmp, self.lblAnhXuLy.width(), self.lblAnhXuLy.height())
            self.displayImage(2)

    def Laplacian(self):
        if self.ImageWasChoosen():
            self.image = cv2.Laplacian(self.tmp, cv2.CV_64F, ksize=3)
            self.image = np.uint8(np.absolute(self.image))
            self.displayImage(2)

    def SobelX(self):
        if self.ImageWasChoosen():
            self.image = cv2.Sobel(self.tmp, cv2.CV_64F, 1, 0)
            self.image = np.uint8(np.absolute(self.image))
            self.displayImage(2)

    def SobelY(self):
        if self.ImageWasChoosen():
            self.image = cv2.Sobel(self.tmp, cv2.CV_64F, 0, 1)
            self.image = np.uint8(np.absolute(self.image))
            self.displayImage(2)

    def SobelCombined(self):
        if self.ImageWasChoosen():
            self.image = cv2.Sobel(self.tmp, cv2.CV_64F, 1, 0)
            self.image = np.uint8(np.absolute(self.image))

            self.img = cv2.Sobel(self.tmp, cv2.CV_64F, 0, 1)
            self.img = np.uint8(np.absolute(self.img))

            self.image = cv2.bitwise_or(self.image, self.img)
            # self.image = self.tmp
            # #
            # # # Converting the image to grayscale, as Sobel Operator requires
            # # # input image to be of mode Grayscale (L)
            # self.image = self.image.convert("L")
            # #
            # # # Calculating Edges using the passed laplican Kernel
            # self.image = self.image.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
            #                                                -1, -1, -1, -1), 1, 0))
            self.displayImage(2)

    def Prewitt(self):
        if self.ImageWasChoosen():
            kernelX = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            kernelY = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            self.image = cv2.filter2D(self.tmp, -1, kernelX)
            self.img = cv2.filter2D(self.tmp, -1, kernelY)
            self.image = self.image + self.img
            self.displayImage(2)

    def detectFace(self):
        if self.ImageWasChoosen():
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            self.image = self.tmp
            self.image = cv2.resize(self.image, (800, 600))
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (255, 0, 0), 3)
            self.displayImage(2)

    def recognizeFace(self):
        self.image = self.tmp
        faces_detected, gray_img = fr.faceDetection(self.image)
        faces, faceID = fr.labels_for_training_data('trainingImages')
        face_recognizer = fr.train_classifier(faces, faceID)
        face_recognizer.write('trainingData.yml')
        name = {0: "Hung", 1: "Ninh", 2: "Danh", 3: "Phuong",
                4: "Quynh"}  # creating dictionary containing names for each label

        for face in faces_detected:
            (x, y, w, h) = face
            roi_gray = gray_img[y:y + h, x:x + h]
            label, confidence = face_recognizer.predict(roi_gray)  # predicting the label of given image
            print("confidence:", confidence)
            print("label:", label)
            fr.draw_rect(self.image, face)
            predicted_name = name[label]
            if (confidence > 80):  # If confidence more than 37 then don't print predicted face text on screen
                continue
            fr.put_text(self.image, predicted_name, x, y)
        self.displayImage(2)

    def saveImage(self):
        if self.lblAnhXuLy.pixmap() is not None:
            image = ImageQt.fromqpixmap(self.lblAnhXuLy.pixmap())
            filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                      "PNG(*.png);;JPEG(*.jpg)")
            if filePath == "":
                return
            image.save(filePath)
        else:
            self.thongBao('Ch??a c?? ???nh x??? l?? ????? l??u')


    def reset(self):
        self.lblAnhGoc.clear()
        self.lblAnhXuLy.clear()
        self.disabled()
        self.tmp = self.image = []

app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
