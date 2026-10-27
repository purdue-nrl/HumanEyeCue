from PyQt5 import uic, QtCore
from PyQt5.QtWidgets import QMainWindow
from datetime import datetime
from PyQt5.QtGui import QFont
from cv2 import getTickCount
from PyQt5.QtWidgets import QMainWindow, QCheckBox, QLabel, QLineEdit, QRadioButton, QSpacerItem, QSizePolicy

class BaseChoiceUI(QMainWindow):
    def __init__(
            self,
            ui_file,
            question,
            options,
            question_type,
            user_id,
            demo,
            question_id):
        super(BaseChoiceUI, self).__init__()
        uic.loadUi(ui_file, self)
        questionString = self.question.text()
        questionParts = self.split_string_with_dash(question, start_string_len=9)
        if demo:
            questionParts = [f"{question_id}"] + questionParts

        newQuestion = questionString.format(*questionParts)
        self.question.setText(newQuestion)
        self.user_id = user_id
        self.boxes = []
        self.options = options
        self.demo = demo

        self.qtype = question_type
        qtype_object_map = {
            'descriptive:count': QLineEdit,
            'descriptive:exist': QRadioButton,
            'descriptive:query_material': QRadioButton,
            'descriptive:query_shape': QRadioButton,
            'descriptive:query_color': QRadioButton,
            'explanatory': QCheckBox,
            'predictive': QCheckBox,
            'counterfactual': QCheckBox
        }

        # Create a QFont object for the options
        font = QFont("Arial", 22)

        if self.qtype == 'descriptive:count':
            options = ['Enter your (0-20) answer below']

        for row, option in enumerate(options):
            checkbox = qtype_object_map[self.qtype]()  # Create a checkbox with the option text
            checkbox.setMaximumWidth(40)
            checkbox.setFont(font)

            self.boxes.append(checkbox)

            questionParts = self.split_string_with_dash(option, start_string_len=9)

            # Split the option string into parts to manage line breaks
            optionParts = self.split_string_with_dash(option, start_string_len=9)
            # Join parts to display them on separate lines
            optionText = "\n".join(optionParts).strip()

            label = QLabel(optionText)
            label.setFont(font)  # Set the font for the label
            label.setAlignment(QtCore.Qt.AlignLeft)
            label.mousePressEvent = lambda event, row=row: self.onClickLabel(row)
            label.row = row

            verticalSpacer = QSpacerItem(40, 40, QSizePolicy.Fixed, QSizePolicy.Minimum)

            self.gridLayout_2.addWidget(checkbox, 2 * row, 0)  # Add checkbox to the grid layout
            self.gridLayout_2.addWidget(label, 2 * row, 1)  # Add checkbox to the grid layout
            self.gridLayout_2.addItem(verticalSpacer, 2 * row + 1, 1)  # Add checkbox to the grid layout


        self.gridLayout_2.addWidget(self.nextButton, 2 * len(options), 3)
        self.nextButton.clicked.connect(self.onNextClicked)

    def onClickLabel(self, idx):
        widget = self.boxes[idx]
        if isinstance(widget, (QRadioButton, QCheckBox)):
            widget.setChecked(not widget.isChecked())

    def get_start_stop(self):
        return self.startTime, self.stopTime, self.startTick, self.stopTick

    def split_string_with_dash(self, input_string, max_chars=80, max_lines=3, start_string_len=6):
        parts = []
        current_part = ""
        max_chars_start = max_chars - start_string_len

        for char in input_string:
            if len(current_part) >= max_chars_start and char.isspace():
                parts.append(current_part.strip() + '-')
                max_chars_start = max_chars
                current_part = '-'
            else:
                current_part += char

        if current_part:
            parts.append(current_part.rstrip('-'))

        while len(parts) < max_lines:
            parts.append("")

        return parts

    def onNextClicked(self):
        self.stopTime = datetime.now()
        self.stopTick = getTickCount()
