#!/usr/bin/env python3
"""
Trajectory Classification Using a Refined Rule Engine and a Decision Tree Classifier,
with Hyperparameter Tuning on Train/Val/Test Splits and Visualization of Ground Truth and Predicted Trajectories.

This script:
  1. Loads and resamples trajectory data.
  2. Extracts features (including enhanced kinematic features like jerk and deceleration) from each trajectory.
  3. Uses either the ground truth labels or the refined rule engine to produce binary labels.
  4. Splits data into train/validation/test, performs hyperparameter tuning on train, evaluates on validation.
  5. Trains best model and evaluates on test set.
  6. Overlays the ground truth and predicted labels on a background image for the test set.
  7. Includes functionality to visualize jerk along a sample trajectory.

Adjust paths, thresholds, and parameters as needed.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from tslearn.preprocessing import TimeSeriesResampler

# -----------------------------
# PARAMETERS & PATHS
# -----------------------------
SAMPLE = "12"
FOLDER_PATH = SAMPLE
IMAGE_PATH = f"sample{SAMPLE}.jpg"
GROUND_TRUTH_CSV_PATH = "trajectory_images/combined_labels.csv"
N_POINTS = 50

OUTPUT_DIR = f"trajectory_DT_sample{SAMPLE}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# THRESHOLDS (Refined rules)
# -----------------------------
THRESHOLD_PARAMS: Dict[str, Any] = {
    "moving_threshold": 10.0,
    "straightness_threshold": 1.30,
    "hv_angle_tolerance_deg": 30.0,
    "turn_angle_tolerance_deg": 45.0,
    "sudden_change_threshold_deg": 65.0,
    "min_segment_length": 5.0,
    # Threshold from image for pseudo-jerk (0.4 rad), converted to degrees for consistency if needed
    # This is not directly used in rules yet, but available. DT will learn from continuous features.
    "pseudo_jerk_angle_change_rad_threshold": 0.4,
}
THRESHOLD_PARAMS["sudden_change_threshold_rad"] = np.deg2rad(
    THRESHOLD_PARAMS["sudden_change_threshold_deg"]
)
THRESHOLD_PARAMS["hv_angle_tolerance_rad"] = np.deg2rad(
    THRESHOLD_PARAMS["hv_angle_tolerance_deg"]
)
THRESHOLD_PARAMS["turn_angle_tolerance_rad"] = np.deg2rad(
    THRESHOLD_PARAMS["turn_angle_tolerance_deg"]
)

# -----------------------------
# CENTRAL ZONE POLYGON (intersection checks)
# -----------------------------
central_zone_polygon = np.array(
    [[1650, 842], [1650, 1331], [2271, 1331], [2271, 842]], dtype=np.int32
)

# -----------------------------
# COLOR MAPS FOR VISUALIZATION (Binary: 0=Normal, 1=Abnormal)
# -----------------------------
BINARY_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (0, 255, 0),  # Green for Normal
    1: (0, 0, 255),  # Red for Abnormal
}


# -----------------------------
# DATA LOADING & PREPROCESSING
# -----------------------------
def extract_trajectory(csv_path: str) -> Optional[np.ndarray]:
    try:
        df = pd.read_csv(csv_path, usecols=["frameNo", "left", "top", "w", "h"])
        df["center_x"] = df["left"] + df["w"] / 2
        df["center_y"] = df["top"] + df["h"] / 2
        if len(df) < 2:
            return None
        return df[["frameNo", "center_x", "center_y"]].values
    except Exception as e:
        # print(f"Error extracting trajectory from {csv_path}: {e}")
        return None


def resample_trajectory(traj: np.ndarray, n_points: int) -> Optional[np.ndarray]:
    if traj is None or traj.shape[0] < 2:
        return None
    xy = traj[:, 1:3]
    try:
        # Ensure xy is float64 and C-contiguous
        xy_contig = np.ascontiguousarray(xy, dtype=np.float64)
        resampler = TimeSeriesResampler(sz=n_points)
        # Reshape to (n_series, n_timesteps, n_features)
        resampled_xy = resampler.fit_transform(xy_contig.reshape(1, -1, 2))[0]
        # Interpolate frame numbers
        frames = np.linspace(traj[0, 0], traj[-1, 0], n_points)
        return np.column_stack((frames, resampled_xy))
    except Exception as e:
        # print(f"Error resampling trajectory: {e}")
        return None


def load_trajectories(
    folder_path: str, n_points: int
) -> Tuple[List[np.ndarray], List[str]]:
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return [], []
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])
    trajs, names = [], []
    for f_name in files:
        file_path = os.path.join(folder_path, f_name)
        t = extract_trajectory(file_path)
        if t is not None:
            rt = resample_trajectory(t, n_points)
            if rt is not None:
                trajs.append(rt)
                names.append(f_name)
    print(
        f"Loaded and resampled {len(trajs)} trajectories from {len(files)} files in {folder_path}."
    )
    return trajs, names


def load_ground_truth_from_csv(gt_csv: str, fnames: List[str]) -> Optional[np.ndarray]:
    if not os.path.exists(gt_csv):
        print(f"Ground truth CSV not found: {gt_csv}")
        return None
    try:
        df = pd.read_csv(gt_csv)
        if "filename" not in df.columns or "label" not in df.columns:
            print("Ground truth CSV missing 'filename' or 'label' column.")
            return None

        # Create a mapping from basename of filename to label
        # Handles cases where filename in CSV might have a path
        mapping = {
            os.path.basename(row.filename): row.label for _, row in df.iterrows()
        }

        # Match fnames (which are basenames) with labels from mapping
        labels = [mapping.get(fname, -1) for fname in fnames]  # -1 if not found

        num_matched = sum(l != -1 for l in labels)
        if num_matched == 0 and len(fnames) > 0:
            print(
                f"Warning: No ground truth labels matched for {len(fnames)} trajectories. Check filenames in CSV."
            )
        else:
            print(
                f"Matched ground truth for {num_matched} out of {len(fnames)} trajectories."
            )
        return np.array(labels)
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return None


# -----------------------------
# ANGLES / RULE-ENGINE UTILITIES
# -----------------------------
def angle_between_points(p1: np.ndarray, p2: np.ndarray) -> float:
    delta = p2 - p1
    return np.arctan2(delta[1], delta[0]) if np.linalg.norm(delta) > 1e-9 else 0.0


def smallest_angle_diff(angle1: float, angle2: float) -> float:
    diff = angle1 - angle2
    return (diff + np.pi) % (2 * np.pi) - np.pi


def get_trajectory_angles(
    traj: np.ndarray, min_segment_len: float
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    points = traj[:, 1:3]
    if len(points) < 2:
        return None, None, None

    overall_angle = angle_between_points(points[0], points[-1])

    start_angle = None
    for i in range(len(points) - 1):
        if np.linalg.norm(points[i + 1] - points[i]) >= min_segment_len:
            start_angle = angle_between_points(points[i], points[i + 1])
            break
    if start_angle is None:  # Fallback if no segment long enough
        start_angle = angle_between_points(points[0], points[1])

    end_angle = None
    for i in range(len(points) - 2, -1, -1):
        if np.linalg.norm(points[i + 1] - points[i]) >= min_segment_len:
            end_angle = angle_between_points(points[i], points[i + 1])
            break
    if end_angle is None:  # Fallback
        end_angle = angle_between_points(points[-2], points[-1])

    return start_angle, end_angle, overall_angle


def get_direction(angle: Optional[float], tolerance_rad: float) -> Optional[str]:
    if angle is None:
        return None
    if abs(smallest_angle_diff(angle, 0)) < tolerance_rad:
        return "W_to_E"
    if abs(smallest_angle_diff(angle, np.pi)) < tolerance_rad:
        return "E_to_W"
    if abs(smallest_angle_diff(angle, np.pi / 2)) < tolerance_rad:
        return "S_to_N"
    if abs(smallest_angle_diff(angle, -np.pi / 2)) < tolerance_rad:
        return "N_to_S"
    return None


def detect_sudden_change(
    traj: np.ndarray, angle_threshold_rad: float
) -> Tuple[bool, float]:
    points = traj[:, 1:3]
    if len(points) < 3:
        return False, 0.0

    segment_vectors = np.diff(points, axis=0)
    segment_norms = np.linalg.norm(segment_vectors, axis=1)

    # Filter out very short segments that might cause noisy angle calculations
    valid_vectors = segment_vectors[segment_norms > 1.0]  # Min length 1 pixel

    if len(valid_vectors) < 2:
        return False, 0.0

    segment_angles = np.arctan2(valid_vectors[:, 1], valid_vectors[:, 0])
    angle_changes = smallest_angle_diff(segment_angles[1:], segment_angles[:-1])

    max_angle_change = np.max(np.abs(angle_changes)) if angle_changes.size > 0 else 0.0

    return max_angle_change > angle_threshold_rad, max_angle_change


def intersects_zone(traj: np.ndarray, zone_polygon: np.ndarray) -> bool:
    if (
        zone_polygon is None
        or len(zone_polygon) < 3
        or traj is None
        or traj.shape[0] == 0
    ):
        return False
    points = traj[:, 1:3].astype(np.float32)  # cv2.pointPolygonTest needs float32
    for pt in points:
        if cv2.pointPolygonTest(zone_polygon, tuple(pt), False) >= 0:
            return True
    return False


# -----------------------------
# REFINED RULE ENGINE
# -----------------------------
def assign_custom_cluster_refined(
    traj: np.ndarray, zone: np.ndarray, thresholds: Dict[str, Any]
) -> int:
    if traj is None or len(traj) < 2:
        return 19  # Unclassifiable / Error

    points = traj[:, 1:3]
    direct_dist = np.linalg.norm(points[-1] - points[0])

    if direct_dist < thresholds["moving_threshold"]:
        return 18  # Stationary or very short movement

    start_angle, end_angle, overall_angle = get_trajectory_angles(
        traj, thresholds["min_segment_length"]
    )
    if (
        start_angle is None or overall_angle is None
    ):  # Should not happen if len(traj) >=2
        return 19

    # Straightness
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = np.sum(segment_lengths)
    straightness_ratio = total_length / (
        direct_dist + 1e-6
    )  # Add epsilon to avoid division by zero

    direction = get_direction(overall_angle, thresholds["hv_angle_tolerance_rad"])

    if straightness_ratio < thresholds["straightness_threshold"]:  # Considered straight
        if direction == "W_to_E":
            return 1
        if direction == "E_to_W":
            return 2
        if direction == "N_to_S":
            return 3
        if direction == "S_to_N":
            return 4
        # Could be diagonal straight, fall through or assign specific cluster

    # Turn detection (Simplified for brevity, original had more complex turn logic)
    # Example: if start_dir and end_dir indicate a turn (e.g. W_to_E then S_to_N for a right turn)
    # This part would typically use start_angle, end_angle, and turn_angle_tolerance_rad
    # For now, we rely on sudden change for more complex abnormal movements

    is_sudden_change, _ = detect_sudden_change(
        traj, thresholds["sudden_change_threshold_rad"]
    )
    if is_sudden_change:
        return 17  # Sudden change / Erratic

    # Default for trajectories not fitting other categories (e.g. curves, complex)
    return 19  # Other / Complex / Potentially abnormal depending on mapping


def map_clusters_to_binary_refined(cluster_labels: List[int]) -> np.ndarray:
    # Define which cluster IDs are considered "abnormal" (1)
    # This mapping is crucial and application-dependent.
    # For example, stationary (18), sudden_change (17), unclassifiable (19)
    # and potentially complex turns if they have specific IDs.
    abnormal_clusters = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}  # Example set
    return np.array(
        [1 if label in abnormal_clusters else 0 for label in cluster_labels]
    )


# -----------------------------
# FEATURE EXTRACTION (ENHANCED)
# -----------------------------
def get_feature_names() -> List[str]:
    """Return names of all features for reference."""
    return [
        # Original features
        "direct_distance",
        "total_length",
        "straightness_ratio",
        "overall_angle",
        "start_angle",
        "end_angle",
        "turn_difference_angle",
        "has_sudden_change",
        "intersects_zone",
        # New kinematic features
        "max_speed",
        "mean_speed",
        "speed_variance",
        "max_acceleration",
        "mean_acceleration",
        "acceleration_variance",
        "max_jerk",
        "mean_jerk",  # Translational Jerk
        "max_deceleration",
        "mean_deceleration",
        "max_angular_velocity",
        "max_angular_acceleration",  # Angular Kinematics
    ]


def extract_enhanced_features(
    traj: np.ndarray, thresholds: Dict[str, Any], zone_polygon: np.ndarray
) -> List[float]:
    """Extracts a comprehensive set of features from a trajectory."""
    if traj is None or len(traj) < 2:  # Need at least 2 points for basic features
        return [0.0] * len(get_feature_names())

    pts = traj[:, 1:3]
    timestamps = traj[:, 0]  # frameNo assumed to be time

    # Basic geometric features
    direct_distance = np.linalg.norm(pts[-1] - pts[0])
    segment_vectors = np.diff(pts, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    total_length = np.sum(segment_lengths)
    straightness_ratio = total_length / (direct_distance + 1e-9)

    start_angle, end_angle, overall_angle = get_trajectory_angles(
        traj, thresholds["min_segment_length"]
    )
    overall_angle = overall_angle if overall_angle is not None else 0.0
    start_angle = start_angle if start_angle is not None else 0.0
    end_angle = end_angle if end_angle is not None else 0.0
    turn_difference_angle = (
        abs(smallest_angle_diff(start_angle, end_angle))
        if start_angle is not None and end_angle is not None
        else 0.0
    )

    has_sudden_change, _ = detect_sudden_change(
        traj, thresholds["sudden_change_threshold_rad"]
    )
    intersects = 1.0 if intersects_zone(traj, zone_polygon) else 0.0

    # Kinematic features (velocity, acceleration, jerk, deceleration)
    # Initialize kinematic features to 0, update if calculable
    max_speed, mean_speed, speed_var = 0.0, 0.0, 0.0
    max_accel, mean_accel, accel_var = 0.0, 0.0, 0.0
    max_jerk, mean_jerk = 0.0, 0.0
    max_decel, mean_decel = 0.0, 0.0
    max_ang_vel, max_ang_accel = 0.0, 0.0

    if len(pts) >= 2:
        dt = np.diff(timestamps)
        dt = np.where(dt <= 0, 1e-6, dt)  # Avoid division by zero or non-positive dt

        velocities = segment_vectors / dt[:, np.newaxis]
        speeds = np.linalg.norm(velocities, axis=1)
        if len(speeds) > 0:
            max_speed = np.max(speeds)
            mean_speed = np.mean(speeds)
            speed_var = np.var(speeds)

        if len(pts) >= 3:  # Need at least 3 points for acceleration
            accelerations = (
                np.diff(velocities, axis=0) / dt[:-1][:, np.newaxis]
            )  # dt for velocity segments
            accel_magnitudes = np.linalg.norm(accelerations, axis=1)
            if len(accel_magnitudes) > 0:
                max_accel = np.max(accel_magnitudes)
                mean_accel = np.mean(accel_magnitudes)
                accel_var = np.var(accel_magnitudes)

            # Deceleration (longitudinal component)
            longitudinal_accels = []
            if len(accelerations) > 0 and len(velocities) > len(
                accelerations
            ):  # Ensure velocities[i] is valid
                for i in range(len(accelerations)):
                    v_dir = velocities[i] / (np.linalg.norm(velocities[i]) + 1e-9)
                    longitudinal_accels.append(np.dot(accelerations[i], v_dir))

            if longitudinal_accels:
                deceleration_values = [
                    -la for la in longitudinal_accels if la < 0
                ]  # Store as positive values
                if deceleration_values:
                    max_decel = np.max(deceleration_values)
                    mean_decel = np.mean(deceleration_values)

            if len(pts) >= 4:  # Need at least 4 points for jerk
                # dt for acceleration segments. Timestamps for accel are t1, t2, ... tn-2.
                # dt_accel = dt[:-1] -> timestamps differences for velocity segments
                # We need dt for acceleration segments themselves
                # Accels are at approx times (t0+t1)/2, (t1+t2)/2 ...
                # Or, more simply, dt for jerk is based on original timestamps for accel calculation
                dt_for_jerk_calc = dt[
                    :-2
                ]  # Corresponds to the dt between start of accel segments

                jerks = np.diff(accelerations, axis=0) / dt_for_jerk_calc[:, np.newaxis]
                jerk_magnitudes = np.linalg.norm(jerks, axis=1)
                if len(jerk_magnitudes) > 0:
                    max_jerk = np.max(jerk_magnitudes)
                    mean_jerk = np.mean(jerk_magnitudes)

        # Angular kinematics
        if len(pts) >= 2:
            segment_angles = np.arctan2(segment_vectors[:, 1], segment_vectors[:, 0])
            if len(segment_angles) >= 2:  # Need at least 2 angles for angular velocity
                angle_diffs = smallest_angle_diff(
                    segment_angles[1:], segment_angles[:-1]
                )
                # dt for angle_diffs is dt[:-1] (corresponding to segments producing these angles)
                angular_velocities = angle_diffs / dt[:-1]
                if len(angular_velocities) > 0:
                    max_ang_vel = np.max(np.abs(angular_velocities))

                if (
                    len(angular_velocities) >= 2
                ):  # Need at least 2 ang_vel for ang_accel
                    # dt for angular_velocities. These are changes between segments.
                    # dt for angular_accels is dt[:-2]
                    angular_accelerations = np.diff(angular_velocities) / dt[:-2]
                    if len(angular_accelerations) > 0:
                        max_ang_accel = np.max(np.abs(angular_accelerations))

    return [
        direct_distance,
        total_length,
        straightness_ratio,
        overall_angle,
        start_angle,
        end_angle,
        turn_difference_angle,
        float(has_sudden_change),
        intersects,
        max_speed,
        mean_speed,
        speed_var,
        max_accel,
        mean_accel,
        accel_var,
        max_jerk,
        mean_jerk,
        max_decel,
        mean_decel,
        max_ang_vel,
        max_ang_accel,
    ]


def build_enhanced_feature_matrix(
    trajs: List[np.ndarray], thresholds: Dict[str, Any], zone_polygon: np.ndarray
) -> np.ndarray:
    if not trajs:
        return np.empty((0, len(get_feature_names())))

    feature_list = [
        extract_enhanced_features(t, thresholds, zone_polygon) for t in trajs
    ]

    # Ensure all feature vectors have the same length (robustness for edge cases in extract_enhanced_features)
    expected_len = len(get_feature_names())
    processed_feature_list = []
    for i, f_vec in enumerate(feature_list):
        if len(f_vec) == expected_len:
            processed_feature_list.append(f_vec)
        else:
            # This case should ideally not happen if extract_enhanced_features is robust
            print(
                f"Warning: Feature vector for trajectory {i} has length {len(f_vec)}, expected {expected_len}. Using zeros."
            )
            processed_feature_list.append([0.0] * expected_len)

    return np.array(processed_feature_list)


# -----------------------------
# HYPERPARAMETER TUNING + TRAIN on Train/Val
# -----------------------------
def tune_and_train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    param_grid: Dict[str, List[Any]] = {
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "criterion": ["gini", "entropy"],
        "class_weight": [None, "balanced"],  # Added None
    }

    base_estimator = DecisionTreeClassifier(random_state=random_state)

    # Handle cases with very few samples or only one class in y_train for GridSearchCV
    if len(np.unique(y_train)) < 2 or len(y_train) < inner_cv.get_n_splits():
        print(
            "Warning: Not enough samples or classes for GridSearchCV. Training with default parameters."
        )
        best_estimator = DecisionTreeClassifier(
            random_state=random_state, class_weight="balanced"
        )
        best_estimator.fit(X_train, y_train)
    else:
        grid_search = GridSearchCV(
            base_estimator,
            param_grid,
            scoring="f1_weighted",  # f1_weighted for binary/multiclass
            cv=inner_cv,
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)
        best_estimator = grid_search.best_estimator_
        print("Best hyperparameters found by GridSearchCV:", grid_search.best_params_)

    # Validation performance
    y_val_pred = best_estimator.predict(X_val)
    print("\n-- Validation Set Performance --")
    print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_val, y_val_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))

    return best_estimator


# -----------------------------
# VISUALIZATION on IMAGE
# -----------------------------
def draw_trajectories(
    image: np.ndarray,
    trajectories: List[np.ndarray],
    labels: np.ndarray,
    color_map: Dict[int, Tuple[int, int, int]],
    thickness: int = 2,
) -> np.ndarray:
    img_copy = image.copy()
    for i, traj in enumerate(trajectories):
        if traj is None or len(traj) < 2:
            continue
        points = traj[:, 1:3].astype(np.int32).reshape((-1, 1, 2))
        label = int(labels[i])
        color = color_map.get(
            label, (128, 128, 128)
        )  # Default to gray if label not in map
        cv2.polylines(
            img_copy, [points], isClosed=False, color=color, thickness=thickness
        )
    return img_copy


def show_image(image_array: np.ndarray, title: str = ""):
    plt.figure(figsize=(12, 9))
    plt.imshow(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    # Save to output directory
    sanitized_title = title.lower().replace(" ", "_").replace(":", "").replace("/", "_")
    save_path = os.path.join(OUTPUT_DIR, f"{sanitized_title}.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization: {save_path}")
    plt.show()


def visualize_on_image(
    base_image_path: str,
    trajectories: List[np.ndarray],
    labels: np.ndarray,
    title: str,
    color_map: Dict[int, Tuple[int, int, int]],
    zone_polygon: Optional[np.ndarray] = None,
):
    if os.path.exists(base_image_path):
        base_image = cv2.imread(base_image_path)
    else:
        print(
            f"Warning: Base image not found at {base_image_path}. Using a white background."
        )
        # Determine max dimensions from trajectories if possible, otherwise default
        max_x, max_y = 1920, 1080  # Default
        if trajectories:
            all_pts_list = [
                t[:, 1:3] for t in trajectories if t is not None and len(t) > 0
            ]
            if all_pts_list:  # Check if list is not empty
                all_pts = np.vstack(all_pts_list)
                if len(all_pts) > 0:
                    max_x = int(np.max(all_pts[:, 0])) + 50
                    max_y = int(np.max(all_pts[:, 1])) + 50
        base_image = (
            np.ones((max_y, max_x, 3), dtype=np.uint8) * 255
        )  # White background

    overlay_image = draw_trajectories(base_image, trajectories, labels, color_map)

    if zone_polygon is not None and len(zone_polygon) > 0:
        cv2.polylines(
            overlay_image,
            [zone_polygon.reshape((-1, 1, 2))],
            isClosed=True,
            color=(255, 0, 255),
            thickness=3,
        )  # Magenta for zone

    show_image(overlay_image, title)


# -----------------------------
# PARAMETER VISUALIZATION ALONG TRAJECTORY
# -----------------------------
def visualize_trajectory_by_parameter(
    base_image_path: str,
    trajectory: np.ndarray,
    parameter_values: np.ndarray,
    param_name: str,
    segment_offset: int = 0,  # Index of the first point of the segment that parameter_values[0] corresponds to
    colormap: str = "viridis",
    alpha: float = 0.8,
    line_thickness: int = 3,
):
    """
    Visualizes a trajectory on an image, with segments colored by a parameter value.

    Args:
        base_image_path: Path to the background image.
        trajectory: A single trajectory (N, 3) array [frame, x, y].
        parameter_values: 1D array of values for segments. Length M.
        param_name: Name of the parameter for title and colorbar label.
        segment_offset: The starting segment index that parameter_values[0] refers to.
                        e.g., if jerk is calculated from point k to k+3 and associated with segment (k+1, k+2),
                        and parameter_values[0] is the first jerk, then it colors segment (pts[1],pts[2]).
                        So segment_offset would be 1 if parameter_values[0] applies to segment defined by (pts[1],pts[2]).
        colormap: Matplotlib colormap name.
        alpha: Transparency of the trajectory lines.
        line_thickness: Thickness of the trajectory lines.
    """
    if trajectory is None or len(trajectory) < 2:
        print(f"Cannot visualize {param_name}: Trajectory is too short or None.")
        return
    if parameter_values is None or len(parameter_values) == 0:
        print(f"Cannot visualize {param_name}: No parameter values provided.")
        return

    if os.path.exists(base_image_path):
        img = cv2.imread(base_image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print(
            f"Warning: Base image {base_image_path} not found. Using white background."
        )
        max_x = int(np.max(trajectory[:, 1])) + 50
        max_y = int(np.max(trajectory[:, 2])) + 50
        img_rgb = np.ones((max_y, max_x, 3), dtype=np.uint8) * 255

    plt.figure(figsize=(12, 9))
    plt.imshow(img_rgb)  # This establishes the current Axes

    pts = trajectory[:, 1:3]
    cmap = plt.get_cmap(colormap)

    # Normalize parameter values for coloring
    min_val, max_val = np.min(parameter_values), np.max(parameter_values)
    if min_val == max_val:  # Avoid division by zero if all values are the same
        normalized_params = np.zeros_like(parameter_values, dtype=float)
    else:
        normalized_params = (parameter_values - min_val) / (max_val - min_val)

    # Plot segments colored by parameter_values
    # parameter_values[i] corresponds to segment (pts[i+segment_offset], pts[i+segment_offset+1])
    num_segments_to_color = len(parameter_values)

    for i in range(num_segments_to_color):
        idx_start = i + segment_offset
        idx_end = idx_start + 1

        if idx_end < len(pts):  # Ensure we don't go out of bounds for points
            p1 = pts[idx_start]
            p2 = pts[idx_end]
            color = cmap(normalized_params[i])
            plt.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=color,
                linewidth=line_thickness,
                alpha=alpha,
            )
        else:
            # This implies parameter_values is longer than drawable segments with offset
            break

    # Add a colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=min_val, vmax=max_val)
    )
    sm.set_array([])  # You need this for the colorbar to work with ScalarMappable
    # ******** THIS IS THE CORRECTED LINE ********
    cbar = plt.colorbar(sm, ax=plt.gca(), alpha=alpha)  # Pass current axes
    # ********************************************
    cbar.set_label(
        f"{param_name} (Value Range: {min_val:.2e} to {max_val:.2e})", fontsize=12
    )

    plt.title(f"Trajectory Visualization by {param_name}", fontsize=16)
    plt.axis("off")
    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR, f"traj_viz_{param_name.lower().replace(' ', '_')}.png"
    )
    plt.savefig(save_path, dpi=150)
    print(f"Saved {param_name} visualization: {save_path}")
    plt.show()


def analyze_and_visualize_kinematic_parameter(
    trajectory: np.ndarray, base_image_path: str, param_type: str = "jerk"
):
    """Calculates and visualizes a kinematic parameter (jerk, accel, speed) along a trajectory."""
    if trajectory is None or len(trajectory) < 2:
        print(f"Trajectory too short for {param_type} analysis.")
        return

    pts = trajectory[:, 1:3]
    timestamps = trajectory[:, 0]

    param_values = None
    segment_offset = 0  # Default for speed (param per segment 0 to N-2)

    if len(pts) < 2:
        return
    dt = np.diff(timestamps)
    dt = np.where(dt <= 0, 1e-6, dt)
    segment_vectors = np.diff(pts, axis=0)
    velocities = segment_vectors / dt[:, np.newaxis]
    speeds = np.linalg.norm(velocities, axis=1)

    if param_type == "speed":
        if len(speeds) > 0:
            param_values = speeds
            segment_offset = 0  # speeds[i] for segment (pts[i], pts[i+1])
        else:
            print("Not enough data for speed calculation.")
            return

    elif param_type == "acceleration":
        if len(pts) >= 3:
            accelerations = np.diff(velocities, axis=0) / dt[:-1][:, np.newaxis]
            accel_magnitudes = np.linalg.norm(accelerations, axis=1)
            if len(accel_magnitudes) > 0:
                param_values = accel_magnitudes
                # accel_magnitudes[i] is from vel[i] and vel[i+1].
                # vel[i] is for seg (pts[i], pts[i+1]). vel[i+1] for (pts[i+1], pts[i+2]).
                # So accel[i] can be associated with point pts[i+1] or segment (pts[i+1], pts[i+2]).
                segment_offset = 1
            else:
                print("Not enough data for acceleration calculation.")
                return
        else:
            print("Trajectory too short for acceleration (needs >= 3 points).")
            return

    elif param_type == "jerk":
        if len(pts) >= 4:
            accelerations = np.diff(velocities, axis=0) / dt[:-1][:, np.newaxis]
            if len(accelerations) >= 2:  # Need at least 2 accel vectors for jerk
                jerks = (
                    np.diff(accelerations, axis=0) / dt[:-2][:, np.newaxis]
                )  # dt for accel segments
                jerk_magnitudes = np.linalg.norm(jerks, axis=1)
                if len(jerk_magnitudes) > 0:
                    param_values = jerk_magnitudes
                    # jerks[i] is from accel[i] and accel[i+1].
                    # accel[i] associated with seg (pts[i+1], pts[i+2]).
                    # accel[i+1] associated with seg (pts[i+2], pts[i+3]).
                    # So jerk[i] can be associated with point pts[i+2] or segment (pts[i+2], pts[i+3]).
                    segment_offset = 2
                else:
                    print("Not enough data for jerk calculation.")
                    return
            else:
                print("Not enough accelerations for jerk (needs >= 2 accelerations).")
                return
        else:
            print("Trajectory too short for jerk (needs >= 4 points).")
            return
    else:
        print(f"Unknown parameter type: {param_type}")
        return

    if param_values is not None and len(param_values) > 0:
        visualize_trajectory_by_parameter(
            base_image_path,
            trajectory,
            param_values,
            param_name=param_type.capitalize(),
            segment_offset=segment_offset,
        )
    else:
        print(f"No {param_type} values to visualize.")


# -----------------------------
# MAIN
# -----------------------------
def main():
    print(f"Starting trajectory analysis for sample: {SAMPLE}")
    print(f"Output directory: {OUTPUT_DIR}")

    # 1. Load Trajectories
    trajectories, trajectory_fnames = load_trajectories(FOLDER_PATH, N_POINTS)
    if not trajectories:
        print("No trajectories loaded. Exiting.")
        return

    # Illustrative: Analyze and visualize jerk for the first loaded trajectory (if it exists)
    if trajectories and IMAGE_PATH:
        print("\n--- Illustrative Kinematic Analysis of First Trajectory ---")
        # analyze_and_visualize_kinematic_parameter(trajectories[0], IMAGE_PATH, param_type="speed")
        # analyze_and_visualize_kinematic_parameter(trajectories[0], IMAGE_PATH, param_type="acceleration")
        analyze_and_visualize_kinematic_parameter(
            trajectories[0], IMAGE_PATH, param_type="jerk"
        )
        print("--- End of Illustrative Analysis ---\n")

    # 2. Load/Generate Labels
    ground_truth_labels = load_ground_truth_from_csv(
        GROUND_TRUTH_CSV_PATH, trajectory_fnames
    )

    y_labels: np.ndarray
    if ground_truth_labels is not None and len(ground_truth_labels) == len(
        trajectories
    ):
        # Filter out trajectories for which GT label is -1 (not found)
        valid_indices = [
            i for i, label in enumerate(ground_truth_labels) if label != -1
        ]
        if len(valid_indices) < len(trajectories):
            print(
                f"Filtered out {len(trajectories) - len(valid_indices)} trajectories due to missing GT labels."
            )

        trajectories = [trajectories[i] for i in valid_indices]
        trajectory_fnames = [trajectory_fnames[i] for i in valid_indices]
        y_labels = ground_truth_labels[valid_indices]

        if not trajectories:  # All trajectories were filtered out
            print("No trajectories remaining after filtering for GT labels. Exiting.")
            return
        print(f"Using {len(y_labels)} ground truth labels.")
    else:
        print(
            "Ground truth labels not available or mismatched. Using rule-based labels."
        )
        rule_based_cluster_labels = [
            assign_custom_cluster_refined(t, central_zone_polygon, THRESHOLD_PARAMS)
            for t in trajectories
        ]
        y_labels = map_clusters_to_binary_refined(rule_based_cluster_labels)
        print(f"Generated {len(y_labels)} rule-based binary labels.")

    # 3. Extract Features
    print(f"Extracting enhanced features for {len(trajectories)} trajectories...")
    X_features = build_enhanced_feature_matrix(
        trajectories, THRESHOLD_PARAMS, central_zone_polygon
    )
    print(f"Feature matrix shape: {X_features.shape}")
    # print("Feature names:", get_feature_names()) # Optional: print feature names

    # 4. Data Splitting (Train/Validation/Test)
    if len(X_features) == 0 or len(y_labels) == 0:
        print("No data available for training/testing. Exiting.")
        return
    if len(X_features) != len(y_labels):
        print(
            f"Mismatch between features ({len(X_features)}) and labels ({len(y_labels)}). Exiting."
        )
        return

    # Ensure there are enough samples for splitting and stratification
    min_samples_for_stratify = (
        np.min(np.unique(y_labels, return_counts=True)[1])
        if len(np.unique(y_labels)) > 1
        else len(y_labels)
    )
    if (
        min_samples_for_stratify < 2 and len(np.unique(y_labels)) > 1
    ):  # Need at least 2 samples of minority class for stratification
        print(
            f"Warning: Minority class has only {min_samples_for_stratify} sample. Stratification might fail or be ineffective."
        )
        # Consider alternative splitting if stratification is problematic
        stratify_option = None
    elif len(np.unique(y_labels)) == 1:
        print(
            f"Warning: Only one class present in y_labels. Stratification is not applicable."
        )
        stratify_option = None
    else:
        stratify_option = y_labels

    try:
        indices = np.arange(len(y_labels))
        # Split into initial train+val and test
        # Ensure test_size is not too large if dataset is small
        test_set_size = min(
            0.2, (5 / len(indices)) if len(indices) > 5 else 0.2
        )  # Ensure at least 5 test samples if possible, adjust if less than 5 elements
        if len(indices) <= 5:  # handle very small datasets
            test_set_size = 0
        elif (
            len(indices) * (1 - test_set_size) < 10
        ):  # if training set becomes too small
            test_set_size = (
                0.1 if len(indices) > 10 else 0
            )  # adjust if total data is extremely small

        if test_set_size > 0 and len(indices) > 1:
            train_val_indices, test_indices = train_test_split(
                indices,
                test_size=test_set_size,
                random_state=42,
                stratify=(
                    stratify_option
                    if stratify_option is not None and test_set_size * len(indices) >= 2
                    else None
                ),  # Stratify only if test set large enough for it
            )
        else:  # Not enough for test set or only 1 sample total
            train_val_indices = indices
            test_indices = np.array([], dtype=int)

        # Split train+val into train and validation
        y_train_val = y_labels[train_val_indices]
        stratify_train_val_option = y_train_val if len(train_val_indices) > 0 else None
        if len(train_val_indices) > 0 and len(np.unique(y_train_val)) > 1:
            min_samples_train_val = np.min(
                np.unique(y_train_val, return_counts=True)[1]
            )
            if min_samples_train_val < 2:
                stratify_train_val_option = None
        elif len(train_val_indices) > 0 and len(np.unique(y_train_val)) == 1:
            stratify_train_val_option = None

        val_set_size = min(
            0.25, (5 / len(train_val_indices)) if len(train_val_indices) > 5 else 0.25
        )
        if len(train_val_indices) <= 5:  # handle very small train_val datasets
            val_set_size = 0
        elif len(train_val_indices) * (1 - val_set_size) < 10:
            val_set_size = 0.1 if len(train_val_indices) > 10 else 0

        if (
            val_set_size > 0 and len(train_val_indices) > 1
        ):  # Need at least 2 samples to split
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=val_set_size,
                random_state=42,
                stratify=(
                    stratify_train_val_option
                    if stratify_train_val_option is not None
                    and val_set_size * len(train_val_indices) >= 2
                    else None
                ),
            )
        else:  # Not enough to split, use all for training, none for validation (or handle as error)
            # print("Warning: Not enough data in train_val_indices to create a validation set. Using all for training.")
            train_indices = train_val_indices
            val_indices = np.array([], dtype=int)  # Empty validation set

        X_train, y_train = X_features[train_indices], y_labels[train_indices]
        X_val, y_val = (
            (X_features[val_indices], y_labels[val_indices])
            if len(val_indices) > 0
            else (np.empty((0, X_features.shape[1])), np.empty(0))
        )
        X_test, y_test = (
            (X_features[test_indices], y_labels[test_indices])
            if len(test_indices) > 0
            else (np.empty((0, X_features.shape[1])), np.empty(0))
        )

        trajs_train = [trajectories[i] for i in train_indices]
        trajs_val = (
            [trajectories[i] for i in val_indices] if len(val_indices) > 0 else []
        )
        trajs_test = (
            [trajectories[i] for i in test_indices] if len(test_indices) > 0 else []
        )

        print(
            f"Data split sizes: Train={len(y_train)}, Validation={len(y_val)}, Test={len(y_test)}"
        )
        if len(y_train) == 0:
            print(
                "Training set is empty. Cannot proceed with model training. Check data and splitting."
            )
            return

    except ValueError as e:
        print(
            f"Error during data splitting (possibly due to small dataset or class imbalance): {e}"
        )
        print("Skipping model training and evaluation.")
        # Fallback: use all data for training if splitting fails catastrophically
        X_train, y_train = X_features, y_labels
        X_val, y_val = np.empty((0, X_features.shape[1])), np.empty(0)
        X_test, y_test = np.empty((0, X_features.shape[1])), np.empty(0)
        trajs_train = trajectories
        trajs_val, trajs_test = [], []
        print(
            f"Fallback: Using all {len(y_train)} samples for training due to splitting error."
        )

    # 5. Tune and Train Decision Tree
    print("\n--- Model Training and Tuning ---")
    if (
        len(X_train) > 0 and len(X_val) > 0 and len(y_val) > 0
    ):  # Ensure val set is usable for tuning
        best_dt_model = tune_and_train_decision_tree(X_train, y_train, X_val, y_val)
    else:  # Fallback if validation set is empty or training set is empty
        if len(X_train) == 0:
            print("Training set is empty. Cannot train model.")
            return
        print(
            "Validation set is empty or training conditions not met for full tuning. Training on (train) data with default/balanced parameters."
        )
        best_dt_model = DecisionTreeClassifier(random_state=42, class_weight="balanced")
        best_dt_model.fit(X_train, y_train)

    # 6. Evaluate on Test Set
    if len(X_test) > 0 and len(y_test) > 0:
        print("\n--- Test Set Performance ---")
        y_test_pred = best_dt_model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_test_pred, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_test_pred))

        # Feature Importances (Optional)
        if hasattr(best_dt_model, "feature_importances_"):
            feature_names = get_feature_names()
            importances = best_dt_model.feature_importances_
            sorted_indices = np.argsort(importances)[::-1]
            print("\nTop Feature Importances:")
            for i in range(min(10, len(feature_names))):  # Print top 10 or fewer
                idx = sorted_indices[i]
                print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

            # Plot feature importances
            plt.figure(
                figsize=(10, max(8, len(feature_names) * 0.3))
            )  # Adjust height based on num features
            plt.title("Feature Importances from Decision Tree")
            plt.bar(
                range(X_train.shape[1]), importances[sorted_indices], align="center"
            )
            plt.xticks(
                range(X_train.shape[1]),
                np.array(feature_names)[sorted_indices],
                rotation=90,
            )
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "feature_importances.png"))
            plt.show()

        # 7. Visualize Test Set Results
        if IMAGE_PATH and trajs_test:  # Ensure trajs_test is not empty
            visualize_on_image(
                IMAGE_PATH,
                trajs_test,
                y_test,
                f"Sample {SAMPLE}: Test Set Ground Truth Labels",
                BINARY_COLORS_BGR,
                central_zone_polygon,
            )
            visualize_on_image(
                IMAGE_PATH,
                trajs_test,
                y_test_pred,
                f"Sample {SAMPLE}: Test Set Predicted Labels",
                BINARY_COLORS_BGR,
                central_zone_polygon,
            )
    else:
        print(
            "Test set is empty or no trajectories for test set. Skipping final evaluation and visualization."
        )

    print(f"\nAnalysis complete for sample {SAMPLE}. Outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
