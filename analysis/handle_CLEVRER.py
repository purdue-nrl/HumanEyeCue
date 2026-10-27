"""
This is a custom dataset class to load a subset of selected files of the original CLEVRER dataset.
"""

import json
import numpy as np
import pycocotools._mask as _mask
import cv2 as cv

class CLEVRER(object):
    def __init__(self, root = "./data/clevrer/") -> None:
        # The root folder is expected to have the structure of:
        # -- root:
            # -- qa_and_parsed_programs_all_selected.json
            # -- video_all_selected/
                # -- video_00047.mp4
                # -- video_00171.mp4
                # ...
            # -- processed_proposals_all_selected/
                # -- sim_00047.json
                # -- sim_00171.json
                # ...
        self.root = root
        self._videos_path = self.root + "video_all_selected/"
        self._main_metadata_path = self.root + "qa_and_parsed_programs_all_selected.json"
        self._content_metadata_path = self.root + "processed_proposals_all_selected/"

        # (1) the main metadata, containing:
            # selected video info, including:
                # their original scene index w.r.t. original CLEVRER dataset (scene_index:int),
                # and the video filename (video_filename:str),
            # selected subset of questions for the corresponding video files (questions:List[dict]),
            # parsed (neuro-symbolic) programs for the selected subset of questions,
            # original split of the selected file, which one of the following: "train", "valid", and "test".
        self.main_metadata = self._load_json(self._main_metadata_path)

        # (2) the processed proposals folder, which contains the content metadata of each corresponding video.
        # Hence, each processed proposal contains:
            # ground truth list of objects occuring in the video,
            # ground truth list of "in" and "out" events, i.e. the objects that enter and leave the field of view during the video,
            # ground truth list of "collision" events,
            # per-frame metadata (for in total 25 frames/sec x 5 sec = 125 frames), containing:
                # the list of objects present in each frame,
                # and their corresponding masks .
        # NOTE: in per-frame metadata, the list of objects is in no particular order, meaning that
        # the order of objects in per-frame metadata and in the ground truth list of objects IS NOT THE SAME.
        # Instead, if a particular object is needed, they need to be "searched" based on their attributes, i.e.  'color', 'material', 'shape'.
        self.content_metadata = {}
        for file_idx, file in self.main_metadata.items():
            scene_index = file['scene_index']
            processed_proposal_filename = "sim_{:05d}.json".format(scene_index)
            processed_proposal = self._load_json(self._content_metadata_path + processed_proposal_filename)
            assert processed_proposal["video_index"] == scene_index, "Attempting to load processed proposal that has video_index={} different from scene_index={}!".format(processed_proposal["video_index"], scene_index)
            self.content_metadata[file_idx] = processed_proposal
        # NOTE: file_idx and video_idx are used interchangeably in the context of selected files!

        self.decoded_mask_dims = (320, 480) # the np array shape of decoded masks


    def _load_json(self, path):
        with open(path, "r") as file_on_path:
            data = json.load(file_on_path)
        return data

    def get_all_objects_in_video(self, video_idx: int):
        video_objects = {}
        map_attributes_to_objects = {} # (color, material, shape) -> object_id
        for video_object in self.content_metadata[str(video_idx)]['ground_truth']['objects']:
            # object_id -> attributes
            video_object_id         = video_object['id']
            video_objects[video_object_id] = video_object
            # attributes -> object_id
            video_object_color      = video_object['color']
            video_object_material   = video_object['material']
            video_object_shape      = video_object['shape']
            map_attributes_to_objects[(video_object_color, video_object_material, video_object_shape)] = video_object_id
        return video_objects, map_attributes_to_objects

    def _get_object_ids_based_on_attributes(self, attributes, map_attributes_to_objects):
        # Finds object_ids EXACTLY matching given attributes, given as (color, material, shape)
        object_ids = []
        for object_attributes in attributes:
            try:
                object_id = map_attributes_to_objects[object_attributes]
            except:
                object_id = None
            object_ids.append(object_id)
        return object_ids

    def _decode_mask(self, mask_info):
        if type(mask_info) == list:
            return _mask.decode(mask_info)
        else:
            return _mask.decode([mask_info])[:,:,0]

    def _grow_mask(self, mask, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask_grown = cv.dilate(mask, kernel, iterations=1)
        return mask_grown

    def get_all_per_frame_masks(self, video_idx: int, grow_mask_by_n_pixels = None):
        video_objects, map_attributes_to_objects = self.get_all_objects_in_video(video_idx)
        num_video_objects = len(video_objects)
        all_per_frame_masks = []
        for frame in self.content_metadata[str(video_idx)]['frames']:
            masks_sorted_by_object_id = np.zeros((num_video_objects, self.decoded_mask_dims[0], self.decoded_mask_dims[1]), dtype=np.uint8)
            for video_object in frame['objects']:
                video_object_color      = video_object['color']
                video_object_material   = video_object['material']
                video_object_shape      = video_object['shape']
                object_attributes       = (video_object_color, video_object_material, video_object_shape)
                video_object_id         = self._get_object_ids_based_on_attributes([object_attributes], map_attributes_to_objects)[0]
                if not video_object_id is None:
                    mask_info = video_object['mask']
                    mask = self._decode_mask(mask_info)
                    if grow_mask_by_n_pixels:
                        assert grow_mask_by_n_pixels <= 40, "Growing mask is limited to maximum by 20 pixels!"
                        mask = self._grow_mask(mask, kernel_size=grow_mask_by_n_pixels)
                    masks_sorted_by_object_id[video_object_id, :, :] = mask
            all_per_frame_masks.append(masks_sorted_by_object_id)
        all_per_frame_masks = np.stack(all_per_frame_masks, axis=0)
        return all_per_frame_masks # binary mask in shape (number_of_frames, total_number_of_objects_appearing_through_entire_video, video_height, video_width)

    def get_video_question_pair(self, video_idx: int, question_idx: int):
    # def get_question_options(self, video_idx: int, question_idx:int):
        """ This method retrives a requested pair of video and question.
        Input:
            video_idx (the same as file_idx): int - specifies the index of one of the selected videos.
            question_idx: int - specifies the index of one of the questions for the corresponding requested video.

        Returns:
            The dictionary file, which contains:
                video_path: str - specifies the direct path to the video file.
                question: str - the prompt of the requested question.
                choices: List[str] - the text of each answer option of the requested question.
                question_type: str - one of the 8 question types (see below).
                answer - the correct answer for the questions which have it.
        """
        """ The original question types, their nature, and their choices:
            descriptive:count               -> open ended, requires numeric input
                                            -> choices: empty list
            descriptive:exist               -> open ended, but there are total of 2 options (yes, no)
                                            -> choices: the list with 2 options: ["yes", "no"]
            descriptive:query_material      -> open ended, but there are total of 2 options (metal, rubber)
                                            -> choices: the list with 2 options: ["metal", "rubber"]
            descriptive:query_shape         -> open ended, but there are total of 3 options (cube, sphere, cylinder)
                                            -> choices: the list with 3 options ["cube", "sphere", "cylinder"]
            descriptive:query_color         -> open ended, but there are total of 8 options (gray, red, blue, green, brown, cyan, purple, yellow)
                                            -> choices: the list with 8 options ["gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow"]
            explanatory                     -> multiple choice questions with possibility of multiple correct options
                                            -> choices: the list with dictionaries for each option, such that the actual text of an option is chosen_question_metadata['choices'][option_idx]['choice']
            predictive                      -> 2-choice questions (i.e. there are only 2 choices for every question)
                                            -> choices: the list with dictionaries for each option, such that the actual text of an option is chosen_question_metadata['choices'][option_idx]['choice']
            counterfactual                  -> multiple (but up to 4) choice questions with possibility of multiple correct options
                                            -> choices: the list with dictionaries for each option, such that the actual text of an option is chosen_question_metadata['choices'][option_idx]['choice']
        """

        # The video_idx parameter specifies the video index as an integer. NOTE: video_idx and file_idx is used interchangeably!
        chosen_video_metadata = self.main_metadata[str(video_idx)]
        # The question_idx parameter specifies the question index as an integer for the corresponding video.
        chosen_question_metadata = chosen_video_metadata['questions'][question_idx]
        # The question prompt itself:
        question_prompt = chosen_question_metadata['question']
        # The question type:
        question_type = chosen_question_metadata['question_type']
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
            "video_path": self._videos_path + chosen_video_metadata['video_filename'],
            "question": question_prompt,
            "choices": question_choices,
            "question_type": question_type,
            "answer": answer
        }
        return output

    def get_video_path(self, video_idx: int):
        """ This method retrives a path to a specific video in the root data folder.
        Input:
            video_idx (the same as file_idx): int - specifies the index of one of the selected videos.

        Returns:
            video_path: str - specifies the direct path to the video file.
        """
        # The video_idx parameter specifies the video index as an integer. NOTE: video_idx and file_idx is used interchangeably!
        chosen_video_metadata = self.main_metadata[str(video_idx)]
        video_path = self._videos_path + chosen_video_metadata['video_filename']

        return video_path


if __name__ == '__main__':
    data_handler = CLEVRER()
    all_per_frame_masks = data_handler.get_all_per_frame_masks(0)
