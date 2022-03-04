import pandas as pd
import numpy as np
import random
import cv2
from helmet_assignment.score import NFLAssignmentScorer

def add_cols(df):
    """Parse video_frame to game_play and video

    Args:
        df (pd.DataFrame):

    Returns:
        pd.DataFrame:
    """
    df["game_play"] = df["video_frame"].str.split("_").str[:2].str.join("_")
    if "video" not in df.columns:
        df["video"] = df["video_frame"].str.split("_").str[:3].str.join("_") + ".mp4"
    return df


def create_triangle(center, sidelength):
    """Creates a triangle with equal sides

    Args:
        center (tuple): Center of the future triangle
        sidelength (int): length of side

    Returns:
        np.array: thre dots with [x,y] coordinates
    """
    a_x = int(center[0] - sidelength / 2)
    b_x = int(center[0] + sidelength / 2)
    c_x = int(center[0])

    h = int(round(np.sqrt(sidelength ** 2 - (sidelength / 2) ** 2)))
    a_y = int(center[1] - h / 2)
    b_y = int(center[1] - h / 2)
    c_y = int(center[1] + h / 2)

    return np.array([[a_x, a_y], [b_x, b_y], [c_x, c_y]])


def find_nearest(array, value):
    """finds the closest est_frame in array to @value

    Args:
        array (np.array): [description]
        value (int): [description]

    Returns:
        int: closest est_frame
    """
    value = int(value)
    array = np.asarray(array).astype(int)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def modern_posit(image_pts, world_pts, focal_length, center):
    """Finds the optimal matrix of rotation and transformation, based on 2D and 3D coordinates

    Args:
        image_pts ([type]): [description]
        world_pts ([type]): [description]
        focal_length ([type]): [description]
        center ([type]): [description]

    Returns:
        rot (np.array), trans (np.array), count(int): [description]
    """
    nb_points = np.shape(image_pts)[0]

    # centered & scaled pixel coordinates
    centered_image = np.divide(np.subtract(image_pts, center), focal_length)
    ui = centered_image[:, 0]
    vi = centered_image[:, 1]

    # homogeneous world coordinates
    homogeneous_world_pts = np.append(world_pts, np.ones((nb_points, 1)), 1)

    # pseudo inverse
    object_mat = np.linalg.pinv(homogeneous_world_pts)
    # print(object_mat)

    converged = 0
    count = 0
    t_x = 0.0
    t_y = 0.0
    t_z = 0.0
    r1 = 0.0
    r2 = 0.0
    r3 = 0.0
    while converged == 0:
        # POS part of the algorithm
        # rotation vectors
        r1_t = np.matmul(object_mat, ui)
        r2_t = np.matmul(object_mat, vi)
        # 1/t_z1 is norm of r1_t
        t_z1 = 1 / np.linalg.norm(r1_t[0:3])
        # 1/tz_2 is norm of r2_t
        t_z2 = 1 / np.linalg.norm(r2_t[0:3])

        # geometric average
        t_z = np.sqrt(t_z1 * t_z2)

        r1_n = np.multiply(r1_t, t_z)
        r2_n = np.multiply(r2_t, t_z)
        r1 = r1_n[0:3]
        r2 = r2_n[0:3]
        r3 = np.cross(r1, r2)
        r3_t = np.append(r3, t_z)
        t_x = r1_n[3]
        t_y = r2_n[3]

        # Now update the z/T z or epsilon
        # then ui, vi
        epsilon_i = np.matmul(homogeneous_world_pts, np.divide(r3_t, t_z))
        old_ui = ui
        old_vi = vi
        ui = np.multiply(epsilon_i, centered_image[:, 0])
        vi = np.multiply(epsilon_i, centered_image[:, 1])

        # check for convergence
        delta_ui = ui - old_ui
        delta_vi = vi - old_vi
        delta = np.square(focal_length) * (
            np.square(np.linalg.norm(delta_ui)) + np.square(np.linalg.norm(delta_vi))
        )

        converged = 1 if count > 0 and delta < 1 else 0
        count = count + 1
        if count > 1000:
            break

    trans = np.array([t_x, t_y, t_z], np.float64)
    rot = np.array([r1, r2, r3], np.float64)

    return rot, trans, count

def get_3d_angles(R):
    """
    Illustration of the rotation matrix / sometimes called 'orientation' matrix
    R = [
           R11 , R12 , R13,
           R21 , R22 , R23,
           R31 , R32 , R33
        ]

    REMARKS:
    1. this implementation is meant to make the mathematics easy to be deciphered
    from the script, not so much on 'optimized' code.
    You can then optimize it to your own style.

    2. I have utilized naval rigid body terminology here whereby;
    2.1 roll -> rotation about x-axis
    2.2 pitch -> rotation about the y-axis
    2.3 yaw -> rotation about the z-axis (this is pointing 'upwards')
    """
    from math import asin, pi, atan2, cos

    R11 = R[0][0]
    R12 = R[0][1]
    R13 = R[0][2]

    R21 = R[1][0]
    R22 = R[1][1]
    R23 = R[1][2]

    R31 = R[2][0]
    R32 = R[2][1]
    R33 = R[2][2]

    if R31 != 1 and R31 != -1:
        pitch_1 = -1 * asin(R31)
        pitch_2 = pi - pitch_1
        roll_1 = atan2(R32 / cos(pitch_1), R33 / cos(pitch_1))
        roll_2 = atan2(R32 / cos(pitch_2), R33 / cos(pitch_2))
        yaw_1 = atan2(R21 / cos(pitch_1), R11 / cos(pitch_1))
        yaw_2 = atan2(R21 / cos(pitch_2), R11 / cos(pitch_2))

        # IMPORTANT NOTE here, there is more than one solution but we choose the first for this case for simplicity !
        # You can insert your own domain logic here on how to handle both solutions appropriately (see the reference publication link for more info).
        pitch = pitch_1
        roll = roll_1
        yaw = yaw_1
    else:
        yaw = 0  # anything (we default this to zero)
        if R31 == -1:
            pitch = pi / 2
            roll = yaw + atan2(R12, R13)
        else:
            pitch = -pi / 2
            roll = -1 * yaw + atan2(-1 * R12, -1 * R13)

    # convert from radians to degrees
    roll = roll * 180 / pi
    pitch = pitch * 180 / pi
    yaw = yaw * 180 / pi

    rxyz_deg = [roll, pitch, yaw]
    return rxyz_deg



def find_min_track_distance(example_tracks):
    example_tracks = example_tracks[["player", "x", "y", "team"]]
    example_tracks_h = example_tracks.loc[example_tracks["team"] == "Home"].copy()
    example_tracks_v = example_tracks.loc[example_tracks["team"] == "Away"].copy()
    this_v = example_tracks_v[["x", "y"]].values
    example_tracks_h["xy"] = example_tracks_h[["x", "y"]].apply(
        lambda z: np.linalg.norm(this_v - np.array([z.x, z.y]), axis=1).min(), axis=1
    )
    min_dist = example_tracks_h["xy"].min()
    return min_dist



def compute_color_for_id(label):
    """Simple function that adds fixed color depending on the id

    Args:
        label (int): just digit

    Returns:
        tuple: color
    """

    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def plot_one_box(x, im, color=None, label=None, line_thickness=3):
    """Plots one bounding box on image 'im' using OpenCV

    Args:
        x (list): 4 numbers: x,y for left up and down right box dots
        im (image): image for printing boxes
        color (tuple, optional): RGB color. Defaults to None.
        label (str, optional): label for box. Defaults to None.
        line_thickness (int, optional): thickness of line for lines and text. Defaults to 3.

    Returns:
        [type]: [description]
    """
    # Plots one bounding box on image 'im' using OpenCV
    assert (
        im.data.contiguous
    ), "Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image."
    tl = (
        line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:  # если label, то задаем его параметры
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            im,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    return im


def add_deepsort_label_col(out):
    """Find the top occuring label for each deepsort_cluster
    Find the # of times that label appears for the deepsort_cluster.

    Args:
        out (pd.DataFrame): result of deepsort_helmets

    Returns:
        pd.DataFrame: with additional columns ['label_deepsort','label_count_deepsort']
    """
    # Find the top occuring label for each deepsort_cluster
    sortlabel_map = (
        out.groupby("deepsort_cluster")["label"]
        .value_counts()
        .sort_values(ascending=False)
        .to_frame()
        .rename(columns={"label": "label_count"})
        .reset_index()
        .groupby(["deepsort_cluster"])
        .first()["label"]
        .to_dict()
    )
    # Find the # of times that label appears for the deepsort_cluster.
    sortlabelcount_map = (
        out.groupby("deepsort_cluster")["label"]
        .value_counts()
        .sort_values(ascending=False)
        .to_frame()
        .rename(columns={"label": "label_count"})
        .reset_index()
        .groupby(["deepsort_cluster"])
        .first()["label_count"]
        .to_dict()
    )

    out["label_deepsort"] = out["deepsort_cluster"].map(sortlabel_map)
    out["label_count_deepsort"] = out["deepsort_cluster"].map(sortlabelcount_map)

    return out


def add_deepsort_label_col_weight(out):
    """Weighted. Find the top occuring label for each deepsort_cluster
    Weighted. Find the # of times that label appears for the deepsort_cluster.

    Args:
        out (pd.DataFrame): result of deepsort_helmets

    Returns:
        pd.DataFrame: with additional columns ['label_deepsort','label_count_deepsort']
    """
    # Find the top occuring label for each deepsort_cluster
    sortlabel_map = (
        out.groupby(["deepsort_cluster", "label"])["dist_weight"]
        .sum()
        .sort_values(ascending=False)
        .to_frame()
        .rename(columns={"label": "label_count"})
        .reset_index()
        .groupby(["deepsort_cluster"])
        .first()["label"]
        .to_dict()
    )
    # Find the # of times that label appears for the deepsort_cluster.
    sortlabelcount_map = (
        out.groupby(["deepsort_cluster", "label"])["dist_weight"]
        .sum()
        .sort_values(ascending=False)
        .to_frame()
        .rename(columns={"label": "label_count"})
        .reset_index()
        .groupby(["deepsort_cluster"])
        .first()["dist_weight"]
        .to_dict()
    )

    out["label_deepsort"] = out["deepsort_cluster"].map(sortlabel_map)
    out["label_count_deepsort"] = out["deepsort_cluster"].map(sortlabelcount_map)

    return out


def score_vs_deepsort(myvideo, out, labels):
    """Score the base predictions compared to the deepsort postprocessed predictions.

    Args:
        myvideo (str): name of video gameplay_view_frame
        out (pd.DataFrame): [description]
        labels (pd.DataFrame): DataFrame with labels
    """
    myvideo_mp4 = myvideo + ".mp4"
    labels_video = labels.query("video == @myvideo_mp4")
    scorer = NFLAssignmentScorer(labels_video)
    out_deduped = out.groupby(["video_frame", "label"]).first().reset_index()
    base_video_score = scorer.score(out_deduped)

    out_preds = out.drop("label", axis=1).rename(columns={"label_deepsort": "label"})
    print(out_preds.shape)
    out_preds = out_preds.groupby(["video_frame", "label"]).first().reset_index()
    print(out_preds.shape)
    deepsort_video_score = scorer.score(out_preds)
    print(f"{base_video_score:0.5f} before --> {deepsort_video_score:0.5f} deepsort")



