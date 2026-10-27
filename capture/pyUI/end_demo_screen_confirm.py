from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import Qt

class EndDemoScreenConfirm(QMainWindow):
    def __init__(self):
        # Initialize the parent QMainWindow class
        super(EndDemoScreenConfirm, self).__init__()

        # Load the UI elements into the StartScreen object
        uic.loadUi('./UI/end_demo_screen_confirm.ui', self)

        # Connect 'clicked' signal to the slot
        self.nextButton.clicked.connect(self.onClickEnd)
        self.confirmInput.setText("Please type yes to confirm")
        self.confirmInput.setFocusPolicy(Qt.NoFocus)
        self.confirmInput.mousePressEvent = lambda x: self.confirmInput.setFocusPolicy(Qt.StrongFocus)

    def onClickEnd(self):
        text = self.confirmInput.text()
        if text.lower() == "yes":
            self.moveToNextPage()
