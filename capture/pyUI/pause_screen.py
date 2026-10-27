from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow

class PauseScreen(QMainWindow):
    def __init__(self):
        # Initialize the parent QMainWindow class
        super(PauseScreen, self).__init__()

        # Load the UI elements into the StartScreen object
        uic.loadUi('./UI/pause_screen.ui', self)

        # Connect 'clicked' signal to the slot
        self.resumeButton.clicked.connect(self.onClickSubmit)

    def onClickSubmit(self):
        self.moveToNextPage()
