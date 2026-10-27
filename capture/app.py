import sys
from PyQt5 import uic
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from pyUI import StartScreen, VideoPlayer
from pyDataset import DataHandler
from pySQL import SQLHandler
import json
from datetime import datetime
from cv2 import getTickCount
from pyUI import StartScreen, EndScreen, PauseScreen, EndDemoScreenConfirm
import pyautogui
import time
from PyQt5.QtGui import QIcon, QPixmap

class App(QMainWindow):
    def __init__(self, pages, session_handler, config):
        super(App, self).__init__()
        uic.loadUi('./UI/united_app.ui', self)
        self.currentPage = 0
        self.pages = pages
        self.session_handler = session_handler
        self.pages = pages

        for page_idx, page_view in enumerate(pages):
            page_view.moveToNextPage = self.onMoveToNextPage
            page_view.session_handler = self.session_handler
            self.stackedWidget.insertWidget(page_idx, page_view)

        if config['full_screen']:
            self.showFullScreen()
        else:
            self.show()

    def addPagesToStack(self, pages):
        pause_screen = self.pages.pop(-1)
        end_screen = self.pages.pop(-1)
        self.stackedWidget.removeWidget(pause_screen)
        self.stackedWidget.removeWidget(end_screen)

        page_offset = len(self.pages)
        self.pages += pages

        for page_idx, page_view in enumerate(pages):
            page_view.moveToNextPage = self.onMoveToNextPage
            page_view.session_handler = self.session_handler
            self.stackedWidget.insertWidget(page_offset + page_idx, page_view)

        self.stackedWidget.insertWidget(len(self.pages), end_screen)
        self.stackedWidget.insertWidget(len(self.pages) + 1, pause_screen)
        self.pages += [end_screen, pause_screen]

    def onMoveToNextPage(self, args=None):
        # Simulate text type used for sync with gaze point analysis
        # This is backup incase the drift in tick time is too long
        for c in "next_page":
            pyautogui.press(c)
            time.sleep(0.02)

        if args:
            if 'pages' in args:
                # Called from the start screen load, which loads the questions for the current user
                pages = args['pages']
                self.addPagesToStack(pages)

            # Called from answer screen
            if 'pause' in args:
                pausePageIndex = len(self.pages) - 1
                self.stackedWidget.setCurrentIndex(pausePageIndex)
                return

        self.currentPage += 1
        if self.currentPage < len(self.pages):
            self.stackedWidget.setCurrentIndex(self.currentPage)
            self.pages[self.currentPage].startTime = datetime.now()
            self.pages[self.currentPage].startTick = getTickCount()
            if isinstance(self.pages[self.currentPage], VideoPlayer):
                self.pages[self.currentPage].playVideo()
        else:
            pass
            # self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Load config from json file
    with open("./config.json", 'r') as f:
        config = json.loads(f.read())

    svg_file = "./icons/icon.svg"
    pixmap = QPixmap(svg_file)
    icon = QIcon(pixmap)
    app.setWindowIcon(icon)

    data_handler = DataHandler(root=config["question_root_path"])
    session_handler = SQLHandler(config=config)
    pages = [
        StartScreen(
            root_path=config["root_path"],
            data_handler=data_handler,
            session_handler=session_handler,
            warn_at_demo_end=config["warn_at_demo_end"]),
        EndScreen(),
        PauseScreen()]

    window = App(pages, session_handler=session_handler, config=config)
    app.exec_()
