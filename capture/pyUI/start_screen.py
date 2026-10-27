import os
import random
import subprocess
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow
from pyUI import UserStartScreen, VideoPlayer, Question, Answer, EndDemoScreen, EndDemoScreenConfirm

class StartScreen(QMainWindow):
    def __init__(self, root_path, data_handler, session_handler, warn_at_demo_end):
        # Initialize the parent QMainWindow class
        super(StartScreen, self).__init__()

        self.data_handler = data_handler
        self.session_handler = session_handler

        # Load the UI elements into the StartScreen object
        uic.loadUi('./UI/start_screen.ui', self)

        # Connect 'clicked' signal to the slot
        self.startButton.clicked.connect(self.onClickSubmit)
        self.root_path = root_path
        self.warn_at_demo_end = warn_at_demo_end

    def open_file_with_default_app(self, file_path):
        # Check if the file exists
        if not os.path.isfile(file_path):
            print("File does not exist.")
            return

        # Open the file using the default application
        if os.name == 'nt':  # for Windows
            os.startfile(file_path)
        elif os.name == 'posix':  # for macOS and Linux
            subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', file_path])

    def create_project_for_user_and_open_gp(self, hr_uid):
        prj_path = os.path.join(self.root_path, f'User_{hr_uid}')
        if not os.path.exists(prj_path):
            os.makedirs(prj_path)
            os.makedirs(os.path.join(prj_path, "result"))
            os.makedirs(os.path.join(prj_path, "src"))
            os.makedirs(os.path.join(prj_path, "user"))

            # Open and read the YAML file
            with open(os.path.join(self.root_path, "template", "NewProject.prj"), 'r') as file:
                data = file.readlines()

            data[8] = data[8].replace("NewMedia0", f'user_{hr_uid}')

            write_file_path = os.path.join(prj_path, f'User_{hr_uid}.prj')
            # Save the updated content to a new file
            with open(write_file_path, 'w') as file:
                file.writelines(data)

        prj_file = os.path.join(prj_path, f'User_{hr_uid}.prj')
        self.open_file_with_default_app(prj_file)

    def onClickSubmit(self):
        email_address = self.emailLineEdit.text()
        is_valid_user_id, user_id = self.session_handler.is_email_registered(email_address)

        if is_valid_user_id:
            self.user_id = user_id
            unanswered = self.session_handler.get_video_questions_unanswered(user_id)
            hr_uid = self.session_handler.get_hruid_for_userid(user_id)
            self.create_project_for_user_and_open_gp(hr_uid)
            self.session_handler.update_session_info(user_id)
            random.shuffle(unanswered)
            data_order = self.session_handler.schedule(unanswered)

            # Add a start screen for the participant to click continue before the proceed
            # This is the screen the participant will see
            pages = [UserStartScreen(email_address)]

            # All videos for demo questions have only a single question, hence q_idx == 0 for all video_idx
            demo_questions = [
                (-3, 0),
                (-2, 0),
                (-1, 0)
            ]

            # Add demo questions to warm up the participant
            for demo_question_id, (video_idx, q_idx) in enumerate(demo_questions):
                pages += self.get_pages_for_question(
                    user_id,
                    video_idx,
                    q_idx,
                    demo=True,
                    question_id=demo_question_id + 1)

            if self.warn_at_demo_end:
                # Screen to warn the users the demo has ended
                pages += [EndDemoScreen(), EndDemoScreenConfirm()]

            # Add questions for the the actual data collection
            for video_idx, q_idx in data_order:
                pages += self.get_pages_for_question(
                    user_id,
                    video_idx,
                    q_idx,
                    demo=False)

            # Transitions to the next page in the application, callback set by the main app
            self.moveToNextPage({
                    'pages': pages,
                }
            )

    def get_pages_for_question(self, user_id, video_idx, q_idx, demo, question_id=0):
        q_details = self.data_handler.get_question_options(video_idx, q_idx)
        pre = Question(
            q_details["question"],
            q_details["choices"],
            q_details["question_type"],
            user_id,
            demo,
            question_id)

        video = VideoPlayer(q_details["video_path"])
        post = Answer(
            video_idx,
            q_idx,
            q_details["question"],
            q_details["choices"],
            q_details["question_type"],
            user_id,
            pre.get_start_stop,
            video.get_start_stop,
            demo=demo,
            question_id=question_id)

        return [pre, video, post]
