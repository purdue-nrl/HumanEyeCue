from PyQt5.QtWidgets import QCheckBox, QLineEdit, QRadioButton
from pyUI import BaseChoiceUI

class Answer(BaseChoiceUI):
    def __init__(self,
            video_no,
            question_no,
            question,
            options,
            question_type,
            user_id,
            get_pre_start_stop,
            get_video_start_stop,
            demo,
            question_id):

        # Load demo .ui, the only difference is it says DEMO on top of the screen
        # otherwise the UI is exactly the same
        if demo:
            ui_file = './UI/post_video_question_demo.ui'
        else:
            ui_file = './UI/post_video_question.ui'

        super(Answer, self).__init__(ui_file, question, options, question_type, user_id, demo, question_id)
        self.pauseSubmitButton.clicked.connect(self.onPauseClicked)
        self.get_pre_start_stop = get_pre_start_stop
        self.get_video_start_stop = get_video_start_stop
        self.video_no = video_no
        self.question_no = question_no

    def submit_info(self):
        # If demo question do not submit data to SQL database
        if self.demo:
            return True

        pre_start_time, pre_stop_time, pre_start_tick, pre_stop_tick = self.get_pre_start_stop()
        video_start_time, video_stop_time, video_start_tick, video_stop_tick = self.get_video_start_stop()
        post_start_time, post_stop_time, post_start_tick, post_stop_tick = self.get_start_stop()
        answer = []

        for box_id, box in enumerate(self.boxes):
            if isinstance(self.boxes[0], QCheckBox) and box.isChecked():
                answer.append(box_id)

            if isinstance(self.boxes[0], QRadioButton) and box.isChecked():
                answer.append(box_id)

            if isinstance(self.boxes[0], QLineEdit):
                try:
                    text = box.text()
                    answer.append(float(text))
                except:
                    return False

        self.session_handler.send_info(
            self.user_id,
            self.video_no,
            self.question_no,
            pre_start_time,
            pre_stop_time,
            video_start_time,
            video_stop_time,
            post_start_time,
            post_stop_time,
            pre_start_tick,
            pre_stop_tick,
            video_start_tick,
            video_stop_tick,
            post_start_tick,
            post_stop_tick,
            answer,
            self.qtype)

        return True

    def onPauseClicked(self):
        super().onNextClicked()
        if self.submit_info():
            self.moveToNextPage(
                {'pause': True}
            )

    def onNextClicked(self):
        super().onNextClicked()

        if self.submit_info():
            self.moveToNextPage()
