import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QHBoxLayout
from PyQt5.QtGui import QPixmap, QIcon, QFont, QPalette, QBrush, QPainter, QColor
from PyQt5.QtCore import Qt, QRect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the trained model
model = load_model('veggie_spoilage_predictor.h5')
class_labels = ['Good', 'Starting to spoil', 'Rotten']

class ImageClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Food Spoilage Predictor')
        self.setGeometry(100, 100, 860, 768)

        # Set background image from current directory
        oImage = QPixmap('./app_bg.jpg')
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(oImage.scaled(self.size(), Qt.KeepAspectRatioByExpanding)))
        self.setPalette(palette)

        # Layoutsclea
        mainLayout = QVBoxLayout()

        # Title with smaller white background
        title = QLabel('Food Spoilage Detector')
        title.setFont(QFont('Arial', 24))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("QLabel { background-color: white; margin: 20px; }")
        title.setMaximumHeight(90)  # Adjust height

        # Image view
        self.imageLabel = QLabel(self)
        self.imageLabel.resize(400, 400)

        # Button
        btnLoad = QPushButton('Load Image', self)
        btnLoad.setFont(QFont('Arial', 16))
        btnLoad.clicked.connect(self.loadImage)
        btnLoad.setFixedHeight(70)  # Double the height
        btnLoad.setFixedWidth(270)  # Adjust width as needed
        btnLoad.setStyleSheet("QPushButton { margin: 5px; }")

        # Center the Load button
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(btnLoad)
        buttonLayout.setAlignment(Qt.AlignCenter)

        # Prediction label with smaller white background
        self.predictionLabel = QLabel('Prediction: None', self)
        self.predictionLabel.setFont(QFont('Arial', 16))
        self.predictionLabel.setStyleSheet("QLabel { background-color: white; margin: 20px; }")
        self.predictionLabel.setMaximumHeight(80)

        # Add widgets to the layouts
        mainLayout.addWidget(title)
        mainLayout.addWidget(self.imageLabel)
        mainLayout.addLayout(buttonLayout)  # Add button layout instead of button directly
        mainLayout.addWidget(self.predictionLabel)

        self.setLayout(mainLayout)

    def loadImage(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '.', "Image files (*.jpg *.jpeg *.png)")
        if fname:
            self.displayImage(fname)
            self.predictImage(fname)

    def displayImage(self, path):
        pixmap = QPixmap(path)
        self.imageLabel.setPixmap(pixmap)
        self.imageLabel.setScaledContents(True)

    def predictImage(self, path):
        img = load_img(path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction[0])]

        self.predictionLabel.setText(f'Prediction: {predicted_class}')

# Run the application
app = QApplication(sys.argv)
ex = ImageClassifierApp()
ex.show()
sys.exit(app.exec_())
