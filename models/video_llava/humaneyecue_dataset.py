import torch
from torch.utils.data import Dataset
import json
# SQL support is optional; the public workflow uses local CSV files.
import logging
import av
import csv
import ast
import numpy as np
import pickle
import os
import random
from humaneyecue_helper_functions import calculate_pixels_for_angle, create_human_saliency_maps_from_gaze

class HumanEyeCue(Dataset):
    def __init__(self, humaneyecue_config_file, split, video_sampling_method, video_num_frames, attention_type):
        super().__init__()

        # 0. Parse input arguments
        ## 0.a. Process configuration file
        print(f"Configuring HumanEyeCue dataset...\n")
        self.configs_file = humaneyecue_config_file
        with open(self.configs_file, 'r') as f:
            self.configs = json.load(f)
        if "sql_configs" in self.configs and not self.configs["sql_configs"] is None:
            logger = logging.getLogger('HumanEyeCue SQL Logger')
            logger.setLevel(logging.INFO)
            logger_handler = logging.StreamHandler()
            logger_handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
            logger.addHandler(logger_handler)
            from handle_SQLHandler import SQLHandler
            self.sql_handler = SQLHandler(config=self.configs["sql_configs"], logger=logger)

        ## 0.b. Process other variables
        assert self.configs["questions_get_from"] in ['sql', 'csv', 'parquet'], f"Received unexpected questions_get_from={self.configs['questions_get_from']}! Options are: ('sql', 'csv', 'parquet').\n"
        assert self.configs["video_frames_get_from"] in ['raw_videos', 'stored_file'], f"Received unexpected video_frames_get_from={self.configs['video_frames_get_from']}! Options are: ('raw_videos', 'stored_file').\n"
        assert self.configs["attention_get_from"] in ['sql', 'csv', 'parquet', 'stored_file'], f"Received unexpected attention_get_from={self.configs['attention_get_from']}! Options are: ('sql', 'csv', 'parquet', 'stored_file').\n"
        split_txt2int = {"train": (0,), "valid": (1,), "test": (2,), "trainval": (0,1), "all": (0,1,2)}
        split_int2txt = {(0,): "train", (1,): "valid", (2,): "test", (0,1): "trainval", (0,1,2): "all"}
        if split in split_txt2int.keys():
            self.split = split_txt2int[split]
            self.split_name = split
        elif split in [(0,), (1,), (2,), (0,1), (0,1,2)]:
            self.split = split
            self.split_name = split_int2txt[split]
        else:
            raise ValueError(f"Received unexpected split={split}! Options are: ('train', 'valid', 'test', 'trainval', 'all').\n")
        assert video_sampling_method in ['uniform', 'random'], f"Receive unexpected video_sampling_method={video_sampling_method}! Options: ('uniform', 'random').\n"
        assert attention_type in [None, 'spatial', 'temporal', 'spatiotemporal'], f"Received unexpected attention_type={attention_type}! Options are: (None, 'spatial', 'temporal', 'spatiotemporal').\n"
        print(f"Completed configuring HumanEyeCue dataset.\n")

        ## 0.c. initialize storage variables
        print(f"Prepping HumanEyeCue's {self.split} split data...\n")
        self.video_data = []
        self.question_data = []
        self.attention_data = []

        # 1. Parse questions
        self.parse_questions()

        # 2-3. Parse videos and attention (if needed)
        self.video_sampling_method = video_sampling_method
        self.video_num_frames = video_num_frames
        self.attention_type = attention_type
        if self.attention_type is None:
            self.parse_videos()
            self.parse_attention()      # <--- just in case we need attention data
        elif self.attention_type == 'spatial':
            self.parse_videos()         # <--- first, we sample video frames (and their indices)....
            self.parse_attention()      # <--- then, we get spatial attention for those sampled video frames.
        elif self.attention_type == 'temporal' or self.attention_type == 'spatiotemporal':
            self.video_sampling_method = 'human_attention'
            self.attention_sigma = calculate_pixels_for_angle(
                viewing_distance_cm = self.configs['collection_settings']['viewing_distance_cm'],
                visual_angle_deg = self.configs['collection_settings']['visual_angle_degrees'],
                screen_resolution = tuple(self.configs['collection_settings']['screen_resolution']),
                screen_diagonal_inches = self.configs['collection_settings']['screen_diagonal_inches'])
            self.attention_sigma *= self.configs['collection_settings']['video_resolution'][1] / self.configs['collection_settings']['screen_resolution'][1]
            self.attention_refractory_period_frames = int(self.configs.get("attention_refractory", 3))
            self.parse_attention()      # <--- first, we get which video frames are significant/important based on attention...
            self.parse_videos()         # <--- then, we sample those video frames.
        else:
            raise ValueError(f"Received unexpected attention_type={self.attention_type}! Options are: (None, 'spatial', 'temporal', 'spatiotemporal').\n")
        print(f"Completed prepping HumanEyeCue's {self.split} split data.\n")

    def __len__(self):
        assert len(self.video_data) == len(self.question_data), "Mismatch in the length of video_data and question_data!\n"
        return len(self.question_data)

    def __getitem__(self, index):
        qdata, vdata, adata = self.question_data[index], self.video_data[index], self.attention_data[index]
        assert qdata["v_no"] == vdata["v_no"] == adata["v_no"] and qdata["q_no"] == vdata["q_no"] == adata["q_no"], "Mismatch detected in requested question_data and video_data!"
        return qdata, vdata, adata


    def parse_questions(self):
        # Check configs to determine how to get questions
        questions_source = self.configs["questions_get_from"]
        if questions_source == 'sql':
            if not "sql_configs" in self.configs or self.configs["sql_configs"] is None:
                raise ValueError(f"Questions are set to be parsed from questions_get_from=sql, but no sql_confis are provided in {self.configs_file}!\n")

            print(f"Parsing questions using provided SQL...\n")
            query_questions = f"SELECT * FROM {self.sql_handler.database}.QUESTIONS q WHERE q.num_responses > 0"
            questions_data, conn = self.sql_handler.execute_sql_query(query=query_questions, commit=False)
            conn.close()
            for row in questions_data: # NOTE: This script assumes that the SQL data is row-based, not column-based.
                q_id, video_no, question_no, question, choices, question_type, answer, num_responses, video, split = row
                if split in self.split:
                    # Storing both v_no and q_no, in case we would need different set of frames for different questions belonging to the same video
                    video_entry = {"v_no": video_no, "q_no": question_no, "v_fn": video}
                    self.video_data.append(video_entry)

                    attention_entry = {"v_no": video_no, "q_no": question_no}
                    self.attention_data.append(attention_entry)

                    question_entry = {"v_no": video_no,
                                      "q_no": question_no,
                                      "q_type": question_type,
                                      "q_text": question,
                                      "q_choices": choices,
                                      "q_gt_answer": answer,
                                      "q_num_human_responses": num_responses}
                    self.question_data.append(question_entry)
            print(f"Parsed a total of {len(self.question_data)} questions!\n")
            return
        elif questions_source == 'csv':
            if not "questions_csv_file" in self.configs or self.configs["questions_csv_file"] is None:
                raise ValueError(f"Questions are set to be parsed from questions_get_from=csv, but no questions_csv_file is provided in {self.configs_file}!\n")

            print(f"Parsing questions using provided CSV...\n")
            with open(self.configs["questions_csv_file"], 'r') as csv_file:
                data = list(csv.reader(csv_file, delimiter=','))
            headers = data[0]
            for row in data[1:]:
                q_id, video_no, question_no, question, choices, question_type, answer, num_responses, video, split = row
                if int(split) in self.split:
                    # Storing both v_no and q_no, in case we would need different set of frames for different questions belonging to the same video
                    video_entry = {"v_no": int(video_no), "q_no": int(question_no), "v_fn": video}
                    self.video_data.append(video_entry)

                    attention_entry = {"v_no": int(video_no), "q_no": int(question_no)}
                    self.attention_data.append(attention_entry)

                    question_entry = {"v_no": int(video_no),
                                      "q_no": int(question_no),
                                      "q_type": question_type,
                                      "q_text": question,
                                      "q_choices": ast.literal_eval(choices),
                                      "q_gt_answer": ast.literal_eval(answer) if answer != 'null' else None,
                                      "q_num_human_responses": int(num_responses)}
                    self.question_data.append(question_entry)
            print(f"Parsed a total of {len(self.question_data)} questions!\n")
            return
        elif questions_source == 'parquet':
            raise NotImplementedError(f"Not implemented! Feel free to DIY or to ping the authors to finish this!")
        else:
            raise ValueError(f"Unknown option is provided for questions_get_from={questions_source}!\n")


    def parse_videos(self):
        """
        Parse the required number of video frames based on the chosen sampling type.
        """
        video_frames_source = self.configs["video_frames_get_from"]
        if video_frames_source == 'raw_videos':
            if not "videos_root_dir" in self.configs or self.configs["videos_root_dir"] is None:
                raise ValueError(f"Video frames are set to be parsed from video_frames_get_from=raw_videos, but no videos_root_dir is provided in {self.configs_file}!\n")

            print(f"Parsing video frames using provided raw videos...\n")
            video_root_dir = self.configs["videos_root_dir"]
            cnt_videos_parsed = 0

            # sample video frames based on the specified sampling method
            if self.attention_type is None or self.attention_type == 'spatial':
                for entry_idx, v_data in enumerate(self.video_data):
                    frames, indices = self.sample_and_read_video_frames(video_path = video_root_dir + v_data["v_fn"],
                                                                        sampling_method = self.video_sampling_method,
                                                                        num_frames = self.video_num_frames)
                    video_entry = {"indices": indices, "frames": frames}
                    v_data.update(video_entry)
                    self.video_data[entry_idx] = v_data
                    cnt_videos_parsed += 1
            # sample video frames based on the attention-based importance, i.e. indices should already be provided by parse_attention()
            else:
                for entry_idx, v_data in enumerate(self.video_data):
                    frames = self.read_video_frames(video_path = video_root_dir + v_data["v_fn"],
                                                    indices = v_data["indices"])
                    video_entry = {"frames": frames}
                    v_data.update(video_entry)
                    self.video_data[entry_idx] = v_data
                    cnt_videos_parsed += 1

            print(f"Parsed video frames for a total of {cnt_videos_parsed} videos!\n")
            if cnt_videos_parsed != len(self.video_data):
                print(f"WARNING: Only {cnt_videos_parsed} out of {len(self.video_data)} videos have valid frames parsed!\n")
            return
        elif video_frames_source == 'stored_file':
            if not "video_frames_file" in self.configs or self.configs["video_frames_file"] is None:
                raise ValueError(f"Video frames are set to be loaded from video_frames_get_from=stored_file, but no video_frames_file is provided in {self.configs_file}!\n")

            print(f"Parsing video frames from stored file...\n")
            variable_settings = str((self.split, self.attention_type, self.video_sampling_method, self.video_num_frames))
            if variable_settings in self.configs["video_frames_file"]:
                with open(self.configs["video_frames_file"][variable_settings], "rb") as video_frames_file:
                    self.video_data = pickle.load(video_frames_file)
                cnt_mismatch = 0
                for qdata, vdata in zip(self.question_data, self.video_data):
                    if qdata['v_no'] != vdata['v_no'] or qdata['q_no'] != vdata['q_no']:
                        cnt_mismatch += 1
                if self.attention_type == 'temporal' or self.attention_type == 'spatiotemporal':
                    for adata, vdata in zip(self.attention_data, self.video_data):
                        if not np.array_equal(adata.get('indices'), vdata.get('indices')):
                            cnt_mismatch += 1
            else:
                raise ValueError(f"Video frames are requested to be loaded for {variable_settings} settings, but no such entry is found in video_frames_file field in {self.configs_file}!\n")
            if cnt_mismatch == 0:
                print(f"Parsed video frames from stored file! No mismatch is found between question_data and video_data!\n")
            else:
                print(f"WARNING: Parsed video frames from stored file, BUT a total of {cnt_mismatch} were NOT matched between question_data and loaded video_data!")
            return
        else:
            raise NotImplementedError(f"Unknown option is provided for video_frames_get_from={video_frames_source}!\n")


    def parse_attention(self):
        """
        Parse the attention data for the sampled video frames.
        """
        # Check configs to determine how to get attention data
        attention_source = self.configs["attention_get_from"]
        if attention_source == 'sql':
            # Query the SQL database to get attention data
            raise NotImplementedError(f"Not implemented! Feel free to DIY or to ping the authors to finish this!")

        elif attention_source == 'csv':
            # Load attention data from a CSV file
            if not "attention_csv_path" in self.configs or self.configs["attention_csv_path"] is None:
                raise ValueError(f"Attention data is set to be parsed from attention_get_from=csv, but no attention_csv_path is provided in {self.configs_file}!\n")

            if not self.configs["attention_csv_path"]["dir"] is None:
                # A directory containing multiple CSV files
                attention_csv_path = self.configs["attention_csv_path"]["dir"]
                if not os.path.isdir(attention_csv_path):
                    raise ValueError(f"Attention CSV directory {attention_csv_path} does not exist!\n")
                print(f"Parsing attention data from CSV files in {attention_csv_path}...\n")
                cnt_attention_parsed = 0

                # sample attention data based on the already sampled video frame indices
                if self.attention_type is None or self.attention_type == 'spatial':
                    for entry_idx, a_data in enumerate(self.attention_data):
                        v_data = self.video_data[entry_idx]
                        assert v_data['v_no'] == a_data['v_no'] and v_data['q_no'] == a_data['q_no'], f"Video and question numbers do not match!"
                        vq_video_csv_file = f"{attention_csv_path}/video_no_{v_data['v_no']}/video_no_{v_data['v_no']}_question_no_{v_data['q_no']}_video.csv"
                        if not os.path.exists(vq_video_csv_file):
                            raise ValueError(f"Attention CSV file {vq_video_csv_file} does not exist!\n")
                        a_raw_data = self.parse_attention_single_vq_pair_csv(csv_file = vq_video_csv_file)

                        frame_offset = int(self.configs["attention_tracker_fps"]/self.configs["attention_videos_fps"])  # 150 fps / 25 fps = 6
                        accumulate = self.configs.get("attention_accumulate", False)
                        gaze_indices = []
                        gaze_per_frame = []
                        for vidx in v_data["indices"]:
                            frame_gaze = []
                            for hr_uid, gaze_list in a_raw_data.items():
                                if accumulate:
                                    start = vidx * frame_offset
                                    end = start + frame_offset
                                    frame_gaze.extend(gaze_list[start:end] if end <= len(gaze_list) else gaze_list[start:])
                                else:
                                    idx = vidx * frame_offset
                                    if idx < len(gaze_list):
                                        frame_gaze.append(gaze_list[idx])
                            gaze_indices.append(vidx)
                            gaze_per_frame.append(frame_gaze)
                        attention_entry = {"indices": np.array(gaze_indices), "gaze": gaze_per_frame}
                        a_data.update(attention_entry)
                        self.attention_data[entry_idx] = a_data
                        cnt_attention_parsed += 1

                # sample attention data based on the attention-based importance, i.e. indices should come from the attention data itself
                else:
                    for entry_idx, a_data in enumerate(self.attention_data):
                        vq_video_csv_file = f"{attention_csv_path}/video_no_{a_data['v_no']}/video_no_{a_data['v_no']}_question_no_{a_data['q_no']}_video.csv"
                        if not os.path.exists(vq_video_csv_file):
                            raise ValueError(f"Attention CSV file {vq_video_csv_file} does not exist!\n")
                        a_raw_data = self.parse_attention_single_vq_pair_csv(csv_file = vq_video_csv_file)
                        total_video_frames = self.configs.get("video_frames_total", 125)

                        frame_offset = int(self.configs["attention_tracker_fps"]/self.configs["attention_videos_fps"])  # 150 fps / 25 fps = 6
                        accumulate = self.configs.get("attention_accumulate", False)
                        ## 1. reformat gaze data from per-user to per-frame
                        all_gaze_indices = []
                        all_gaze_per_frame = []
                        for vidx in range(total_video_frames):
                            frame_gaze = []
                            for hr_uid, gaze_list in a_raw_data.items():
                                if accumulate:
                                    start = vidx * frame_offset
                                    end = start + frame_offset
                                    frame_gaze.extend(gaze_list[start:end] if end <= len(gaze_list) else gaze_list[start:])
                                else:
                                    idx = vidx * frame_offset
                                    if idx < len(gaze_list):
                                        frame_gaze.append(gaze_list[idx])
                            all_gaze_indices.append(vidx)
                            all_gaze_per_frame.append(frame_gaze)

                        ## 2. create human saliency maps from gaze data for all frames
                        saliency_maps = create_human_saliency_maps_from_gaze(
                            human_gaze=np.expand_dims(np.asarray(all_gaze_per_frame), axis=0),  # shape: (1, num_frames, num_gaze_points, 2)
                            saliency_map_size=self.configs['collection_settings']['video_resolution'],
                            sigma_px=self.attention_sigma)

                        ## 3. compute per-frame peak values of each saliency map
                        saliency_maps   = saliency_maps[0]   # shape: (num_frames, height, width)   # within dataset file, it is always one sample processing at a time, as collate happens outside within the dataloader
                        flattened_sms   = saliency_maps.reshape(total_video_frames, -1)
                        peaks           = np.nanmax(flattened_sms, axis=1) # shape: (num_frames)
                        peaks           = np.where(np.isfinite(peaks), peaks, -np.inf)  # if an entire frame were NaN, replace its peak with -inf so it won't be selected later

                        ## 4. select important frames based on the peak value of each saliency map
                        selected_gaze_indices = self.select_salient_frames(
                            peaks=peaks,
                            num_frames_to_select=self.video_num_frames,
                            refractory_period=self.attention_refractory_period_frames)

                        ## 5. optionally, plot a (temporal) graph of the peak value of saliency map vs. frame index
                        # plt.figure()
                        # plt.plot(np.arange(total_video_frames), peaks)
                        # plt.xlabel("Frame index")
                        # plt.ylabel("Peak saliency value")
                        # plt.title("Peak saliency per frame")
                        # plt.tight_layout()
                        # for idx in selected_gaze_indices:
                        #     plt.axvline(x=idx, color='r', linestyle='--', alpha=0.7, linewidth=1)
                        # plt.show()  # uncomment if running interactively

                        ## 6. collect (slice) gaze data for the selected important frames
                        selected_gaze_per_frame = []
                        for fidx in selected_gaze_indices:
                            # all_gaze_per_frame[fidx] is already list[num_recordings] of tuple(2,)
                            # Ensure tuples (in case inner items are lists/np arrays)
                            per_frame_gaze = [tuple(xy) for xy in all_gaze_per_frame[fidx]]
                            selected_gaze_per_frame.append(per_frame_gaze)

                        attention_entry = {"indices": np.array(selected_gaze_indices), "gaze": selected_gaze_per_frame}
                        a_data.update(attention_entry)
                        self.attention_data[entry_idx] = a_data

                        v_data = self.video_data[entry_idx]
                        video_entry = {"indices": np.array(selected_gaze_indices)}
                        v_data.update(video_entry)
                        self.video_data[entry_idx] = v_data
                        cnt_attention_parsed += 1

                print(f"Parsed attention data for a total of {cnt_attention_parsed} videos!\n")
                if cnt_attention_parsed != len(self.video_data):
                    print(f"Warning: Only {cnt_attention_parsed} out of {len(self.video_data)} videos have valid attention data parsed!\n")
                return

            elif not self.configs["attention_csv_path"]["file"] is None:
                # A single CSV file
                attention_csv_path = self.configs["attention_csv_path"]["file"]
                raise NotImplementedError(f"Not implemented! Feel free to DIY or to ping the authors to finish this!")
            else:
                raise ValueError(f"Attention data is set to be parsed from attention_get_from=csv, but no valid options is provided in attention_csv_path.keys() = {self.configs['attention_csv_path'].keys()}!\n")

        elif attention_source == 'parquet':
            # Load attention data from a Parquet file
            raise NotImplementedError(f"Not implemented! Feel free to DIY or to ping the authors to finish this!")
        elif attention_source == 'stored_file':
            # Load attention data from a stored file
            if not "attention_file" in self.configs or self.configs["attention_file"] is None:
                raise ValueError(f"Attention data is set to be loaded from attention_get_from=stored_file, but no attention_file is provided in {self.configs_file}!\n")

            print(f"Parsing attention data from stored file...\n")
            variable_settings = str((self.split, self.attention_type, self.video_sampling_method, self.video_num_frames))
            if variable_settings in self.configs["attention_file"]:
                with open(self.configs["attention_file"][variable_settings], "rb") as attention_file:
                    self.attention_data = pickle.load(attention_file)
                cnt_mismatch = 0
                for vdata, adata in zip(self.video_data, self.attention_data):
                    if vdata['v_no'] != adata['v_no'] or vdata['q_no'] != adata['q_no']:
                        cnt_mismatch += 1
                if self.attention_type is None or self.attention_type == 'spatial':
                    for adata, vdata in zip(self.attention_data, self.video_data):
                        if not np.array_equal(adata.get('indices'), vdata.get('indices')):
                            cnt_mismatch += 1
            else:
                raise ValueError(f"Attention data is requested to be loaded for {variable_settings} settings, but no such entry is found in attention_file field in {self.configs_file}!\n")
            if cnt_mismatch == 0:
                print(f"Parsed attention data from stored file! No mismatch is found between video_data and attention_data!\n")
            else:
                print(f"WARNING: Parsed attention data from stored file, BUT a total of {cnt_mismatch} were NOT matched between video_data and loaded attention_data!")
            return
        else:
            raise ValueError(f"Unknown option is provided for attention_get_from={attention_source}!\n")


    def select_salient_frames(self, peaks, num_frames_to_select, refractory_period=0):
        """
        Select frames with highest saliency peaks under a refractory constraint.
        If not enough frames can be selected, fill the remaining slots randomly.

        Args:
            peaks (np.ndarray): Array of shape [num_frames] containing per-frame peak values.
            num_frames_to_select (int): Number of frames to select (K).
            refractory_period (int): Number of frames to exclude on both sides of each selected frame.

        Returns:
            list[int]: Indices of selected frames (chronologically sorted).
        """
        num_frames = len(peaks)
        available = np.ones(num_frames, dtype=bool)
        selected_idxs = []
        peaks_work = peaks.copy()

        # --- Greedy selection phase ---
        #    - Greedily take the highest remaining peak
        #    - After picking i, ban [i - R, i + R] where R=refractory_period
        while len(selected_idxs) < min(num_frames_to_select, num_frames) and available.any():
            masked_scores = np.where(available, peaks_work, -np.inf)
            i = int(np.argmax(masked_scores))

            # Stop if no valid frame left
            if not np.isfinite(masked_scores[i]):
                break

            selected_idxs.append(i)

            # Exclude refractory neighborhood
            L = max(0, i - refractory_period)
            U_excl = min(num_frames, i + refractory_period + 1)
            available[L:U_excl] = False

        # --- Fallback: random fill if fewer than required frames selected ---
        remaining_needed = num_frames_to_select - len(selected_idxs)
        if remaining_needed > 0:
            remaining_candidates = [i for i in range(num_frames) if i not in selected_idxs]
            if remaining_candidates:
                extra_picks = random.sample(
                    remaining_candidates,
                    min(remaining_needed, len(remaining_candidates))
                )
                selected_idxs.extend(extra_picks)

        # --- Final sort for chronological order ---
        return sorted(selected_idxs)


    def save_attention_data(self, save_path='./'):
        filename = f"attention_data_{self.split_name}_{self.attention_type}_{self.video_sampling_method}_{self.video_num_frames}.pkl"
        full_path = os.path.join(save_path, filename)
        with open(full_path, 'wb') as f:
            pickle.dump(self.attention_data, f)
        print(f"Saved attention data to {full_path}!\n")

    def save_video_data(self, save_path='./'):
        filename = f"video_data_{self.split_name}_{self.attention_type}_{self.video_sampling_method}_{self.video_num_frames}.pkl"
        full_path = os.path.join(save_path, filename)
        with open(full_path, 'wb') as f:
            pickle.dump(self.video_data, f)
        print(f"Saved video data to {full_path}!\n")


    def parse_attention_single_vq_pair_csv(self, csv_file):
        result = {}
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                hr_uid = row['hr_uid']
                try:
                    x = float(row['FPOGX'])
                    y = float(row['FPOGY'])
                except (ValueError, KeyError):
                    continue  # skip rows with invalid or missing data
                if hr_uid not in result:
                    result[hr_uid] = []
                result[hr_uid].append((x, y))
        return result


    def sample_and_read_video_frames(self, video_path, sampling_method, num_frames):
        """
        Read a video and select video frames.
        Based on: https://colab.research.google.com/drive/1tmjp68IcCnLYASeW8Hg5pHyMdeQCIwP-
        """
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        if sampling_method == 'uniform':
            indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        elif sampling_method == 'random':
            np.random.seed(self.configs["seed"])
            indices = np.sort(np.random.choice(total_frames, size=num_frames, replace=False))
        else:
            raise ValueError(f"Receive unexpected video_sampling_method={sampling_method}! Options: ('uniform', 'random').\n")
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        frames = np.stack([x.to_ndarray(format="rgb24") for x in frames])
        return frames, indices


    def read_video_frames(self, video_path, indices):
        container = av.open(video_path)
        container.seek(0)
        frames = []
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        frames = np.stack([x.to_ndarray(format="rgb24") for x in frames])
        return frames
