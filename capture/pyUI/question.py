from PyQt5 import uic
from pyUI import BaseChoiceUI

class Question(BaseChoiceUI):
    def __init__(self, question, options, question_type, user_id, demo, question_id):
        # Load demo .ui, the only difference is it says DEMO on top of the screen
        # otherwise the UI is exactly the same
        if demo:
            ui_file = './UI/pre_video_question_demo.ui'
        else:
            ui_file = './UI/pre_video_question.ui'

        super(Question, self).__init__(
            ui_file,
            question,
            options,
            question_type,
            user_id,
            demo,
            question_id)

        # For question disable all the checkboxes/input elements
        for box in self.boxes:
            box.setEnabled(False)

    def onNextClicked(self):
        super().onNextClicked()
        self.moveToNextPage()
