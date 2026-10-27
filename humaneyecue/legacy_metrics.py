
import numpy as np
import cv2
import operator



# def gaussian(squared_distance, sigma):
#     return (1.0 / (2 * np.pi * sigma ** 2)) * np.exp(-squared_distance / (2 * sigma ** 2))

# def create_saliency_map(fixations, recording_screen_size, saliency_map_size, sigma=35):
#     """
#     Generates a saliency map based on fixation points using a Gaussian distribution.

#     fixations (list of tuples): List of (x, y) tuples for fixation points.
#     recording_source_size (tuple): The (height, width) of the screen at which fixations were captured.
#     saliency_map_size (tuple): The (height, width) of the output (downsampled) saliency map. Default is to match the size of the source video.
#     sigma (int): Standard deviation of the Gaussian kernel.

#     Returns:
#     numpy.ndarray: A 3D array representing the sequence of 2D saliency maps.
#     """
#     saliency_maps = []

#     # Create a Gaussian kernel
#     size = 6 * sigma + 1
#     gauss_kernel = np.fromfunction(lambda x, y: gaussian((x - size//2) ** 2 + (y - size//2) ** 2, sigma), (size, size))

#     # Add each fixation to the map
#     for idx, (x_raw, y_raw) in enumerate(fixations):
#         x = min(max(x_raw*recording_screen_size[1], 0), recording_screen_size[1] - 1)
#         y = min(max(y_raw*recording_screen_size[0], 0), recording_screen_size[0] - 1)
#         x, y = int(x), int(y)  # Ensure coordinates are integers
#         # Define the bounds for kernel placement
#         x_start, x_end = x - size // 2, x + size // 2 + 1
#         y_start, y_end = y - size // 2, y + size // 2 + 1

#         # Adjust if Gaussian goes out of image bounds
#         x_start_k, y_start_k = 0, 0
#         x_end_k, y_end_k = size, size
#         if x_start < 0:
#             x_start_k = -x_start
#             x_start = 0
#         if y_start < 0:
#             y_start_k = -y_start
#             y_start = 0
#         if x_end > recording_screen_size[1]:
#             x_end_k -= x_end - recording_screen_size[1]
#             x_end = recording_screen_size[1]
#         if y_end > recording_screen_size[0]:
#             y_end_k -= y_end - recording_screen_size[0]
#             y_end = recording_screen_size[0]

#         # Create an empty image for the saliency map
#         saliency_map = np.zeros(recording_screen_size, dtype=np.float32)

#         # Add the Gaussian kernel to the saliency map
#         saliency_map[y_start:y_end, x_start:x_end] += gauss_kernel[y_start_k:y_end_k, x_start_k:x_end_k]

#         # Normalize to 0-255 for visualization
#         saliency_map_src = cv2.normalize(saliency_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
#         saliency_map_out = cv2.resize(saliency_map_src, (saliency_map_size[1], saliency_map_size[0]), interpolation=cv2.INTER_AREA)
#         saliency_maps.append(saliency_map_out)
#     return np.stack(saliency_maps, axis=0)

# # Example usage:
# fixations = [(150, 150), (400, 300), (250, 250)]  # Example fixation points
# output_size = (640, 480)  # Size of the output saliency map

# saliency_maps = create_saliency_map(fixations, output_size)
# # Display each map one by one
# for i, map in enumerate(saliency_maps):
#     cv2.imshow(f'Saliency Map {i+1}', map)
#     if cv2.waitKey(0) & 0xFF == ord('q'):  # Press 'q' to quit or any key to continue
#         break
# cv2.destroyAllWindows()


# def sum_all_per_frame_saliency_maps(per_frame_saliency_maps: dict, output_size: tuple, keep_last_data_frames: bool = True):
#     frame_cnt_per_uid = [data.shape[0] for hr_uid, data in per_frame_saliency_maps.items()]

#     if keep_last_data_frames:
#         sum_per_frame_SMs = np.zeros((max(frame_cnt_per_uid), output_size[0], output_size[1]), dtype=np.float32)
#         uid_cnt_per_frame = np.zeros((max(frame_cnt_per_uid), 1, 1), dtype=np.uint8)
#         for hr_uid, data in per_frame_saliency_maps.items():
#             sum_per_frame_SMs[0:data.shape[0], :, :] += data
#             uid_cnt_per_frame[0:data.shape[0], :, :] += 1
#     else:
#         sum_per_frame_SMs = np.zeros((min(frame_cnt_per_uid), output_size[0], output_size[1]), dtype=np.float32)
#         uid_cnt_per_frame = np.zeros((min(frame_cnt_per_uid), 1, 1), dtype=np.uint8)
#         for hr_uid, data in per_frame_saliency_maps.items():
#             sum_per_frame_SMs[:, :, :] += data[0:sum_per_frame_SMs.shape[0], :, :]
#             uid_cnt_per_frame[:, :, :] += 1
#     return sum_per_frame_SMs, uid_cnt_per_frame


# def subtract_one_per_frame_saliency_map(sum_per_frame_SMs, uid_cnt_per_frame, per_frame_saliency_maps: dict, subtract_hr_uid: int, keep_last_data_frames: bool = True):
#     data = per_frame_saliency_maps[subtract_hr_uid]

#     if keep_last_data_frames:
#         sum_per_frame_SMs[0:data.shape[0], :, :] -= data
#         uid_cnt_per_frame[0:data.shape[0], :, :] -= 1
#     else:
#         sum_per_frame_SMs[:, :, :] += data[0:sum_per_frame_SMs.shape[0], :, :]
#         uid_cnt_per_frame[:, :, :] += 1
#     return sum_per_frame_SMs, uid_cnt_per_frame


# def average_per_frame_saliency_maps(sum_per_frame_SMs, uid_cnt_per_frame):
#     assert (uid_cnt_per_frame == 0).sum() == 0, """One of the frames in sum_per_frame_SMs has zero user recordings, which would result in division by zero!
#     Please, check if there is at least one user recording for all frames in sum_per_frame_SMs"""

#     average_per_frame_SMs = sum_per_frame_SMs/uid_cnt_per_frame
#     return average_per_frame_SMs


# def create_normalized_saliency_maps(user_fixations, recording_screen_size, saliency_map_size, sigma=35):
#     """
#     Generates per-frame saliency maps based on fixation points of a set of users using a Gaussian distribution.

#     user_fixations (numpy.ndarray): A 3D numpy array in the shape of (number_of_users, number_of_frames, tuple of (x, y) coordinates) for fixation points.
#     recording_screen_size (tuple): The (height, width) of the screen at which fixations were captured.
#     saliency_map_size (tuple): The (height, width) of the output (downsampled) saliency maps. Default is to match the size of the video.
#     sigma (int): Standard deviation of the Gaussian kernel.

#     Returns:
#     numpy.ndarray: A 3D array representing the per-frame (sequence) of 2D saliency maps.
#     """
#     num_users, num_frames, _ = user_fixations.shape
#     saliency_maps = []

#     for frame_idx in range(num_frames):
#         frame_map = np.zeros(recording_screen_size, dtype=np.float32)
#         valid_users_count = 0 # Counter for users with valid fixations

#         for user_idx in range(num_users):
#             x_raw, y_raw = user_fixations[user_idx, frame_idx]
#             if not np.isnan(x_raw) and not np.isnan(y_raw) and 0.0 <= x_raw < 1.0 and 0.0 <= y_raw < 1.0:
#                 x = int(x_raw*recording_screen_size[1]) # Ensure coordinates are scaled and casted into integers
#                 y = int(y_raw*recording_screen_size[0])
#                 frame_map[y, x] += 1
#                 valid_users_count += 1 # Increment valid users count

#         if valid_users_count > 0: # Check if there are any valid fixations
#             frame_map /= valid_users_count
#             blurred_map = cv2.GaussianBlur(frame_map, (0, 0), sigmaX=sigma)
#         else:
#             blurred_map = frame_map  # Keep zero map if no valid data present

#         resized_map = cv2.resize(blurred_map, (saliency_map_size[1], saliency_map_size[0]))
#         normalized_map = cv2.normalize(resized_map, None, 0, 255, cv2.NORM_MINMAX)
#         saliency_maps.append(normalized_map.astype(np.uint8))
#     return saliency_maps


def calculate_ppd(viewing_distance_cm = 55, visual_angle_deg = 2, screen_resolution = (1080, 1920),  screen_width_cm = None, screen_diagonal_inches = None):
    if screen_width_cm and screen_diagonal_inches is None:
        screen_width_cm = screen_width_cm
    elif screen_diagonal_inches and screen_width_cm is None:
        if screen_resolution == (1080, 1920):
            aspect_ratio = np.sqrt(16**2 + 9**2)
        else:
            raise Exception("Please, add aspect ratio for screen resolution = {} to the function code or provide screen width in cm.".format(screen_resolution))
        screen_diagonal_cm = screen_diagonal_inches * 2.54
        screen_width_cm = screen_diagonal_cm * (16/aspect_ratio)
    else:
        raise ValueError("Please, specify either screen diagonal in inches or screen width in centemiers!")

    # Calculate pixels per degree (PPD)
    ppd = 2 * viewing_distance_cm * np.tan(np.radians(visual_angle_deg/2)) * (screen_resolution[1]/screen_width_cm)
    return ppd


def create_normalized_saliency_maps(user_fixations, saliency_map_size=(320, 480), sigma=17.345):
    """
    Generates per-frame saliency maps based on fixation points of a set of users using a Gaussian distribution.

    user_fixations (numpy.ndarray): A 3D numpy array in the shape of (number_of_users, number_of_frames, tuple of (x, y) coordinates) for fixation points.
    saliency_map_size (tuple): The (height, width) of the output (downsampled) saliency maps. Default is to match the size of the video.
    sigma (int): Standard deviation of the Gaussian kernel.

    Returns:
    numpy.ndarray: A 3D array representing the per-frame (sequence) of 2D saliency maps.
    """
    num_users, num_frames, _ = user_fixations.shape
    saliency_maps = []

    for frame_idx in range(num_frames):
        frame_map = np.zeros(saliency_map_size, dtype=np.float32)
        valid_users_count = 0 # Counter for users with valid fixations

        for user_idx in range(num_users):
            x_raw, y_raw = user_fixations[user_idx, frame_idx]
            if not np.isnan(x_raw) and not np.isnan(y_raw) and 0.0 <= x_raw < 1.0 and 0.0 <= y_raw < 1.0:
                x = int(x_raw*saliency_map_size[1]) # Ensure coordinates are scaled and casted into integers
                y = int(y_raw*saliency_map_size[0])
                frame_map[y, x] += 1
                valid_users_count += 1 # Increment valid users count

        if valid_users_count > 0: # Check if there are any valid fixations
            frame_map /= valid_users_count
            blurred_map = cv2.GaussianBlur(frame_map, (0, 0), sigmaX=sigma)
            total_sum = np.sum(blurred_map)
            normalized_map = blurred_map / total_sum
        else:
            blurred_map = frame_map  # Keep zero map if no valid data present
            normalized_map = blurred_map

        # normalized_map = cv2.normalize(blurred_map, None, 0, 255, cv2.NORM_MINMAX)
        # saliency_maps.append(normalized_map.astype(np.uint8))

        # saliency_maps.append(blurred_map)

        saliency_maps.append(normalized_map)
    return saliency_maps


def compute_fixation_saliency_percentiles_only(user_fixations, saliency_maps):
    """
    Computes the percentile score of the saliency value at each fixation point.

    Args:
    user_fixations (numpy array): Array of shape (num_fixations, 2) for (x, y) coordinates.
    saliency_maps (list of numpy arrays): List of 2D saliency maps, where the length of the list is num_frames.

    Returns:
    list: List of percentile values for each fixation, including zeros for NaN coordinates.
    """
    percentiles = []
    num_frames = len(saliency_maps)

    for fixation_idx in range(user_fixations.shape[0]):
        x_raw, y_raw = user_fixations[fixation_idx]
        # Check if either coordinate is NaN or out of bounds, and if it's within the range of available saliency maps
        if not (np.isnan(x_raw) or np.isnan(y_raw) or fixation_idx >= num_frames or not (0.0 <= x_raw < 1.0 and 0.0 <= y_raw < 1.0)):
            saliency_map_height, saliency_map_width = saliency_maps[fixation_idx].shape
            x = int(x_raw*saliency_map_width)
            y = int(y_raw*saliency_map_height)
            saliency_value = saliency_maps[fixation_idx][y, x]
            less_than = np.sum(saliency_maps[fixation_idx] < saliency_value)
            total_pixels = saliency_maps[fixation_idx].size
            percentile = less_than*100.0 / total_pixels
            percentiles.append(percentile)
    return percentiles


def compute_fixation_saliency_percentiles_and_nss(user_fixations, saliency_maps):
    """
    Computes the percentile score of the saliency value at each fixation point and the normalized saliency scores (NSS).

    Args:
    user_fixations (numpy array): Array of shape (num_fixations, 2) for (x, y) coordinates.
    saliency_maps (list of numpy arrays): List of 2D saliency maps, where the length of the list is num_frames.

    Returns:
    list: List of percentile values for each fixation.
    list: List of NSS scores for each valid fixation.
    """
    percentiles = []
    normalized_saliency_scores = []
    num_frames = len(saliency_maps)

    for fixation_idx in range(user_fixations.shape[0]):
        x_raw, y_raw = user_fixations[fixation_idx]
        # Check if either coordinate is NaN or out of bounds, and if it's within the range of available saliency maps
        if not (np.isnan(x_raw) or np.isnan(y_raw) or fixation_idx >= num_frames or not (0.0 <= x_raw < 1.0 and 0.0 <= y_raw < 1.0)):
            saliency_map_height, saliency_map_width = saliency_maps[fixation_idx].shape
            x = int(x_raw*saliency_map_width)
            y = int(y_raw*saliency_map_height)
            # Compute percentile
            saliency_value = saliency_maps[fixation_idx][y, x]
            less_than = np.sum(saliency_maps[fixation_idx] < saliency_value)
            total_pixels = saliency_maps[fixation_idx].size
            percentile = less_than*100.0 / total_pixels
            percentiles.append(percentile)


            # Normalize saliency map to have zero mean and unit standard deviation
            mean = np.mean(saliency_maps[fixation_idx])
            std = np.std(saliency_maps[fixation_idx])
            if std == 0:
                normalized_saliency_scores.append(0.0)
            else:
                saliency_normalized = (saliency_maps[fixation_idx] - mean) / std
                # Compute NSS
                nss = saliency_normalized[y, x]
                normalized_saliency_scores.append(nss)
    return percentiles, normalized_saliency_scores



def compute_fixation_saliency_nss_only(user_fixations, saliency_maps):
    """
    Computes the normalized saliency scores (NSS) at each fixation point.

    Args:
    user_fixations (numpy array): Array of shape (num_fixations, 2) for (x, y) coordinates.
    saliency_maps (list of numpy arrays): List of 2D saliency maps, where the length of the list is num_frames.

    Returns:
    list: List of NSS scores for each valid fixation.
    """
    normalized_saliency_scores = []
    num_frames = len(saliency_maps)

    for fixation_idx in range(user_fixations.shape[0]):
        x_raw, y_raw = user_fixations[fixation_idx]
        # Check if either coordinate is NaN or out of bounds, and if it's within the range of available saliency maps
        if not (np.isnan(x_raw) or np.isnan(y_raw) or fixation_idx >= num_frames or not (0.0 <= x_raw < 1.0 and 0.0 <= y_raw < 1.0)):
            saliency_map_height, saliency_map_width = saliency_maps[fixation_idx].shape
            x = int(x_raw*saliency_map_width)
            y = int(y_raw*saliency_map_height)
            # Normalize saliency map to have zero mean and unit standard deviation
            mean = np.mean(saliency_maps[fixation_idx])
            std = np.std(saliency_maps[fixation_idx])
            if std == 0:
                normalized_saliency_scores.append(0.0)
            else:
                saliency_normalized = (saliency_maps[fixation_idx] - mean) / std
                # Compute NSS
                nss = saliency_normalized[y, x]
                normalized_saliency_scores.append(nss)
    return normalized_saliency_scores




################################################################
###### Functions for attention visualization over videos #######
################################################################
# OpenCV colormap: from color word to BGR color code
# CLEVRER colors ["gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow"]
color_palette = {"gray": (128, 128, 128), "red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 255, 0),
                 "brown": (0, 75, 150), "cyan": (255, 255, 0), "purple": (240, 32, 160), "yellow": (0, 255, 255)}

### Plotting helper function
def plot_gaze_on_frame(frame_input, user_id, user_gaze_center, user_gaze_color):
    # Function for drawing gaze data onto frame
    frame_modified = frame_input.copy()

    # the outer border
    user_gaze_radius_out = 8
    user_gaze_color_out = (255, 255, 255)
    user_gaze_thickness_out = -1
    frame_modified = cv2.circle(frame_modified, user_gaze_center, user_gaze_radius_out, user_gaze_color_out, user_gaze_thickness_out)
    # the inner circle
    user_gaze_radius_in = 6
    user_gaze_thickness_in = -1
    frame_modified = cv2.circle(frame_modified, user_gaze_center, user_gaze_radius_in, user_gaze_color, user_gaze_thickness_in)
    # the user id
    user_id_text = str(user_id)
    user_id_text_face = cv2.FONT_HERSHEY_PLAIN
    user_id_text_scale = 1
    user_id_text_thickness = 1
    user_id_text_size, _ = cv2.getTextSize(user_id_text, user_id_text_face, user_id_text_scale, user_id_text_thickness)
    user_id_text_origin = (user_gaze_center[0] - user_id_text_size[0] // 2, user_gaze_center[1] + user_id_text_size[1] // 2)
    user_id_text_color = (0, 0, 0)
    frame_modified = cv2.putText(frame_modified, user_id_text, user_id_text_origin, user_id_text_face, user_id_text_scale, user_id_text_thickness)
    return frame_modified

### Gaze analysis helper function
def determine_gaze_on_objects(all_objects, current_frame_all_object_masks, eye_data_y_coord, eye_data_x_coord):
    user_gaze_color = None
    # Determines if the gaze is falling within any of the visible objects within the video
    objects_at_gaze = current_frame_all_object_masks[:, eye_data_y_coord, eye_data_x_coord]

    if sum(objects_at_gaze) == 0:
        object_at_gaze_id = None
        user_gaze_color = (255, 255, 255)
    elif sum(objects_at_gaze) == 1:
        object_at_gaze_id = np.argmax(objects_at_gaze)
        object_at_gaze_color = all_objects[object_at_gaze_id]['color']
        user_gaze_color = color_palette[object_at_gaze_color]
    elif sum(objects_at_gaze) > 1:
        for object_at_gaze_id, object_at_gaze_val in enumerate(objects_at_gaze):
            if object_at_gaze_val == 1:
                object_at_gaze_color = all_objects[object_at_gaze_id]['color']
                if user_gaze_color is None:
                    user_gaze_color  = color_palette[object_at_gaze_color]
                else:
                    user_gaze_color = tuple(map(operator.add, user_gaze_color, color_palette[object_at_gaze_color]))
        user_gaze_color /= sum(objects_at_gaze)
    return objects_at_gaze, user_gaze_color
