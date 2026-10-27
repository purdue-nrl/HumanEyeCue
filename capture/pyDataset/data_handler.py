# The json module is imported to allow for easy JSON manipulation. This module provides functions for parsing JSON strings and converting data between JSON format and Python objects.
import json

'''
Define a class named DataHandler to encapsulate the operations related to handling the data.
Example usage of the DataHandler class:
Assuming the JSON file "template_data.json" is in the same directory and formatted as described,
you can create an instance of DataHandler and use it to access question options like so:

handler = DataHandler()
options = handler.get_question_options(1)
print(options)
'''
class DataHandler:
    # The __init__ method is the constructor in Python. It's called when an instance of the class is created.
    def __init__(self, root = "../VQA-DATA/data/all/") -> None:
        # The root directory for all videos and questions.
        self.data_root = root
        # Open the JSON file containing the template data in read mode ('r'). The file path is "./template_data.json".
        with open(self.data_root + "qa_all_selected_with_demo.json", 'r') as f:
            # Use json.load() to parse the JSON file. This function reads the file and converts the JSON formatted string into a Python dictionary.
            # The parsed JSON data is then stored in the instance variable self.input_data_json.
            self.input_data_json = json.load(f)

    def get_shuffled_video_and_questions(self, user_id):
        # Legacy example schedule; the capture app uses SQLHandler.schedule().
        # get the questions and videos that need to be presented to the current user_id
        return [
            (1, 0),
            (1, 2),
            (2, 3)
        ]

    # Define a method called get_question_options that retrieves the question options for a given video and question index.
    def get_question_options(self, video_idx: int, question_idx:int):
        # The video_idx parameter specifies the video index as an integer.
        # The dictionary below contains:
        # 'scene_index' -> the id of the video w.r.t. CLEVRER dataset.
        # 'video_filename' -> the name of the video file. Can be used to retrieve the corresponding video.
        # 'questions' -> the list of questions associated with the chosen video.
        # 'split' -> the split of the corresponding video ("train", "valid", or "test")
        chosen_video_metadata = self.input_data_json[str(video_idx)]
        # The question_idx parameter specifies the question index as an integer for the corresponding video.
        chosen_question_metadata = chosen_video_metadata['questions'][question_idx]
        # The question prompt itself:
        question_prompt = chosen_question_metadata['question']

        # The original question types and their nature:
            # descriptive:count            -> open ended, requires numeric input
            # descriptive:exist            -> open ended, but there are total of 2 options (yes, no)
            # descriptive:query_material   -> open ended, but there are total of 2 options (metal, rubber)
            # descriptive:query_shape      -> open ended, but there are total of 3 options (cube, sphere, cylinder)
            # descriptive:query_color      -> open ended, but there are total of 8 options (gray, red, blue, green, brown, cyan, purple, yellow)
            # explanatory                   -> multiple choice questions with possibility of multiple correct options
            # predictive                    -> 2-choice questions (i.e. there are only 2 choices for every question)
            # counterfactual                -> multiple (but up to 4) choice questions with possibility of multiple correct options
        question_type = chosen_question_metadata['question_type']
        # Based on the question type, the choices are:
            # descriptive:count            -> empty list
            # descriptive:exist            -> the list with 2 options: ["yes", "no"]
            # descriptive:query_material   -> the list with 2 options: ["metal", "rubber"]
            # descriptive:query_shape      -> the list with 3 options ["cube", "sphere", "cylinder"]
            # descriptive:query_color      -> the list with 8 options ["gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow"]
            # explanatory                  -> the list with dictionaries for each option, such that the actual text of an option is chosen_question_metadata['choices'][option_idx]['choice']
            # predictive                   -> the list with dictionaries for each option, such that the actual text of an option is chosen_question_metadata['choices'][option_idx]['choice']
            # counterfactual               -> the list with dictionaries for each option, such that the actual text of an option is chosen_question_metadata['choices'][option_idx]['choice']
        # Hence, to possibly simplify:
        if 'descriptive' in question_type:
            question_choices = chosen_question_metadata['choices']
            if 'answer' in chosen_question_metadata:
                answer = chosen_question_metadata['answer']
            else:
                answer = None
        else:
            question_choices = [chosen_question_metadata['choices'][option_idx]['choice'] for option_idx in range(len(chosen_question_metadata['choices']))]
            try:
                answer = [chosen_question_metadata['choices'][option_idx]['answer'] for option_idx in range(len(chosen_question_metadata['choices']))]
            except:
                answer = None

        output = {
            "video_path": self.data_root + "video_all_selected/" + chosen_video_metadata['video_filename'],
            "question": question_prompt,
            "choices": question_choices,
            "question_type": question_type,
            "answer": answer
        }
        return output
