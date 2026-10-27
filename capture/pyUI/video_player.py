from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt
from datetime import datetime
from cv2 import getTickCount

class VideoPlayer(QWidget):
    def __init__(self, file):
        super().__init__()
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoWidget = QVideoWidget(self)
        self.videoWidget.setAspectRatioMode(Qt.IgnoreAspectRatio)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.videoWidget)
        self.layout.setContentsMargins(0, 0, 0, 0)  # Set layout margins to zero
        self.file = file
        self.setLayout(self.layout)

        # Connect to the mediaStatusChanged signal
        self.mediaPlayer.mediaStatusChanged.connect(self.onMediaStatusChanged)

    def playVideo(self):
        self.startTime = datetime.now()
        self.startTick = getTickCount()
        media = QMediaContent(QUrl.fromLocalFile(self.file))
        self.mediaPlayer.setMedia(media)
        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.mediaPlayer.play()

    def onMediaStatusChanged(self, status):
        # Check if the media finished playing
        if status == QMediaPlayer.EndOfMedia:
            self.stopTime = datetime.now()
            self.stopTick = getTickCount()
            self.moveToNextPage()

    def get_start_stop(self):
        return self.startTime, self.stopTime, self.startTick, self.stopTick
