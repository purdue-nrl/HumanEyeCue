from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow

class EndDemoScreen(QMainWindow):
    def __init__(self):
        # Initialize the parent QMainWindow class
        super(EndDemoScreen, self).__init__()

        # Load the UI elements into the StartScreen object
        uic.loadUi('./UI/end_demo_screen.ui', self)

        # Connect 'clicked' signal to the slot
        self.nextButton.clicked.connect(self.onClickEnd)

    def onClickEnd(self):
        self.moveToNextPage()
