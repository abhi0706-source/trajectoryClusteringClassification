#!/usr/bin/env python3
"""
Trajectory Classification Using a Refined Rule Engine and a Decision Tree Classifier,
with Hyperparameter Tuning on Train/Val/Test Splits and Visualization of Ground Truth and Predicted Trajectories.

This script:
- Loads and resamples trajectory data.
- Extracts enhanced features from each trajectory (including jerk, angular rates, deceleration).
- Uses either the ground truth labels or the refined rule engine to produce binary labels.
- Splits data into train/validation/test, performs hyperparameter tuning on train, evaluates on validation.
- Trains best model and evaluates on test set.
- Overlays the ground truth and predicted labels on a background image for the test set.
- Provides 2D visualization of trajectories colored by parameter (e.g., jerk).
- Provides 3D visualization of trajectories (x, y, time).
"""

import os
import re  # For filename sanitization
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from tslearn.preprocessing import TimeSeriesResampler

# -----------------------------
# PARAMETERS & PATHS
# -----------------------------

SAMPLE = "11"
FOLDER_PATH = SAMPLE
IMAGE_PATH = f"sample{SAMPLE}.jpg"  # Ensure this image exists or adjust path
GROUND_TRUTH_CSV_PATH = (
    "trajectory_images/combined_labels.csv"  # Ensure this CSV exists or adjust
)
N_POINTS = 50  # Number of points after resampling

OUTPUT_DIR = f"trajectory_DT_sample{SAMPLE}_enhanced_v4"  # Changed output dir name
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
# UTILITY FUNCTIONS
# -----------------------------
def sanitize_filename(filename_component: str) -> str:
    """Removes or replaces characters invalid in Windows filenames."""
    # Remove invalid characters like \ / : * ? " < > | ^
    sanitized = re.sub(r'[\\/*?:"<>|^]', "", filename_component)
    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    # Replace parentheses with underscores (or remove them)
    sanitized = sanitized.replace("(", "_").replace(")", "")
    # Ensure it's not too long (optional)
    # sanitized = sanitized[:100] # Example length limit
    return sanitized


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
        print(f"Error extracting trajectory from {csv_path}: {e}")
        return None


def resample_trajectory(traj: np.ndarray, n_points: int) -> Optional[np.ndarray]:
    if traj is None or traj.shape[0] < 2:
        return None
    xy = traj[:, 1:3]
    try:
        xy_contig = np.ascontiguousarray(xy, dtype=np.float64)
        resampler = TimeSeriesResampler(sz=n_points)
        resampled_xy = resampler.fit_transform(xy_contig.reshape(1, -1, 2))[0]
        frames = np.linspace(traj[0, 0], traj[-1, 0], n_points).astype(int)
        return np.column_stack((frames, resampled_xy))
    except Exception as e:
        print(f"Error resampling trajectory: {e}")
        return None


def load_trajectories(
    folder_path: str, n_points: int
) -> Tuple[List[np.ndarray], List[str]]:
    if not os.path.isdir(folder_path):
        print(f"Error: Folder {folder_path} not found.")
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
    print(f"Loaded {len(trajs)} trajectories from {len(files)} files in {folder_path}.")
    return trajs, names


def load_ground_truth_from_csv(gt_csv: str, fnames: List[str]) -> Optional[np.ndarray]:
    if not os.path.exists(gt_csv):
        print(f"Error: Ground truth CSV {gt_csv} not found.")
        return None
    try:
        df = pd.read_csv(gt_csv)
        if "filename" not in df.columns or "label" not in df.columns:
            print(
                "Error: Ground truth CSV must contain 'filename' and 'label' columns."
            )
            return None

        gt_map = {os.path.basename(row.filename): row.label for _, row in df.iterrows()}
        labels = []
        matched_count = 0
        for fname_full in fnames:
            base_fname = os.path.basename(fname_full)
            label = gt_map.get(base_fname)
            if label is not None:
                labels.append(label)
                matched_count += 1
            else:
                labels.append(-1)
        print(f"Matched GT for {matched_count} out of {len(fnames)} trajectories.")
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


def smallest_angle_diff(a1: float, a2: float) -> float:
    d = a1 - a2
    return (d + np.pi) % (2 * np.pi) - np.pi


def get_trajectory_angles(
    traj: np.ndarray, min_len: float
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    pts = traj[:, 1:3]
    if len(pts) < 2:
        return None, None, None
    overall_angle = angle_between_points(pts[0], pts[-1])
    start_angle = None
    for i in range(len(pts) - 1):
        if np.linalg.norm(pts[i + 1] - pts[i]) >= min_len:
            start_angle = angle_between_points(pts[i], pts[i + 1])
            break
    if start_angle is None:
        start_angle = angle_between_points(pts[0], pts[1])
    end_angle = None
    for i in range(len(pts) - 2, -1, -1):
        if np.linalg.norm(pts[i + 1] - pts[i]) >= min_len:
            end_angle = angle_between_points(pts[i], pts[i + 1])
            break
    if end_angle is None:
        end_angle = angle_between_points(pts[-2], pts[-1])
    return start_angle, end_angle, overall_angle


def get_direction(angle: Optional[float], tol: float) -> Optional[str]:
    if angle is None:
        return None
    if abs(smallest_angle_diff(angle, 0)) < tol:
        return "W_to_E"
    if abs(smallest_angle_diff(angle, np.pi)) < tol:
        return "E_to_W"
    if abs(smallest_angle_diff(angle, np.pi / 2)) < tol:
        return "S_to_N"
    if abs(smallest_angle_diff(angle, -np.pi / 2)) < tol:
        return "N_to_S"
    return None


def detect_sudden_change(
    traj: np.ndarray, change_threshold_rad: float
) -> Tuple[bool, float]:
    pts = traj[:, 1:3]
    if len(pts) < 3:
        return False, 0.0
    vectors = np.diff(pts, axis=0)
    segment_lengths = np.linalg.norm(vectors, axis=1)
    valid_vectors = vectors[segment_lengths > 1.0]
    if len(valid_vectors) < 2:
        return False, 0.0
    angles = np.arctan2(valid_vectors[:, 1], valid_vectors[:, 0])
    angle_differences = smallest_angle_diff(angles[1:], angles[:-1])
    max_abs_change = (
        np.max(np.abs(angle_differences)) if angle_differences.size > 0 else 0.0
    )
    return max_abs_change > change_threshold_rad, max_abs_change


def intersects_zone(traj: np.ndarray, zone_polygon: np.ndarray) -> bool:
    if (
        zone_polygon is None
        or len(zone_polygon) < 3
        or traj is None
        or traj.shape[0] == 0
    ):
        return False
    for pt in traj[:, 1:3].astype(np.float32):
        if cv2.pointPolygonTest(zone_polygon, tuple(pt), False) >= 0:
            return True
    return False


# -----------------------------
# REFINED RULE ENGINE
# -----------------------------


def assign_custom_cluster_refined(
    traj: np.ndarray, zone: np.ndarray, thr: Dict[str, Any]
) -> int:
    if traj is None or len(traj) < 2:
        return 19
    pts = traj[:, 1:3]
    direct_dist = np.linalg.norm(pts[-1] - pts[0])
    if direct_dist < thr["moving_threshold"]:
        return 18
    sa, ea, oa = get_trajectory_angles(traj, thr["min_segment_length"])
    if sa is None or ea is None or oa is None:
        return 19
    total_length = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    straightness_ratio = total_length / (direct_dist + 1e-6)
    overall_direction = get_direction(oa, thr["hv_angle_tolerance_rad"])
    if straightness_ratio < thr["straightness_threshold"]:
        if overall_direction == "W_to_E":
            return 1
        if overall_direction == "E_to_W":
            return 2
        if overall_direction == "N_to_S":
            return 3
        if overall_direction == "S_to_N":
            return 4
    start_dir = get_direction(sa, thr["turn_angle_tolerance_rad"])
    end_dir = get_direction(ea, thr["turn_angle_tolerance_rad"])
    if start_dir and end_dir:
        if (start_dir in ["W_to_E", "E_to_W"] and end_dir in ["N_to_S", "S_to_N"]) or (
            start_dir in ["N_to_S", "S_to_N"] and end_dir in ["W_to_E", "E_to_W"]
        ):
            return 10
    is_sudden_change, _ = detect_sudden_change(traj, thr["sudden_change_threshold_rad"])
    if is_sudden_change:
        return 17
    return 19


def map_clusters_to_binary_refined(cluster_labels: List[int]) -> np.ndarray:
    abnormal_clusters = {10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
    return np.array(
        [1 if label in abnormal_clusters else 0 for label in cluster_labels]
    )


# -----------------------------
# ENHANCED FEATURE EXTRACTION
# -----------------------------


def get_feature_names() -> List[str]:
    return [
        "direct_distance",
        "total_length",
        "straightness_ratio",
        "overall_angle",
        "start_angle",
        "end_angle",
        "turn_angle_difference",
        "is_sudden_change",
        "intersects_central_zone",
        "max_speed",
        "mean_speed",
        "speed_variance",
        "max_acceleration",
        "mean_acceleration",
        "acceleration_variance",
        "max_jerk",
        "mean_jerk",
        "max_angular_velocity",
        "max_angular_acceleration",
        "max_deceleration",
        "mean_deceleration",
    ]


def extract_enhanced_features(
    traj: np.ndarray, thr: Dict[str, Any], zone: np.ndarray
) -> List[float]:
    pts = traj[:, 1:3]
    timestamps = traj[:, 0]
    direct_dist = np.linalg.norm(pts[-1] - pts[0])
    segment_vectors = np.diff(pts, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    total_length = np.sum(segment_lengths)
    straightness_ratio = total_length / (direct_dist + 1e-6)
    sa_orig, ea_orig, oa_orig = get_trajectory_angles(traj, thr["min_segment_length"])
    overall_angle = oa_orig if oa_orig is not None else 0.0
    start_angle = sa_orig if sa_orig is not None else 0.0
    end_angle = ea_orig if ea_orig is not None else 0.0
    turn_angle_diff = (
        abs(smallest_angle_diff(sa_orig, ea_orig))
        if sa_orig is not None and ea_orig is not None
        else 0.0
    )
    is_sudden_change_flag, _ = detect_sudden_change(
        traj, thr["sudden_change_threshold_rad"]
    )
    intersects_zone_flag = 1 if intersects_zone(traj, zone) else 0
    max_speed, mean_speed, speed_variance = 0.0, 0.0, 0.0
    max_accel, mean_accel, accel_variance = 0.0, 0.0, 0.0
    max_jerk, mean_jerk = 0.0, 0.0
    max_angular_vel, max_angular_accel = 0.0, 0.0
    max_decel, mean_decel = 0.0, 0.0

    if len(pts) >= 2:
        dt = np.diff(timestamps)
        dt = np.where(dt == 0, 1e-6, dt)
        velocities_vec = segment_vectors / dt[:, np.newaxis]
        speeds = np.linalg.norm(velocities_vec, axis=1)
        if len(speeds) > 0:
            max_speed, mean_speed, speed_variance = (
                np.max(speeds),
                np.mean(speeds),
                np.var(speeds),
            )
        if len(velocities_vec) >= 2:
            dt_acc = np.diff(timestamps[:-1])
            dt_acc = np.where(dt_acc == 0, 1e-6, dt_acc)
            accelerations_vec = np.diff(velocities_vec, axis=0) / dt_acc[:, np.newaxis]
            accel_magnitudes = np.linalg.norm(accelerations_vec, axis=1)
            if len(accel_magnitudes) > 0:
                max_accel, mean_accel, accel_variance = (
                    np.max(accel_magnitudes),
                    np.mean(accel_magnitudes),
                    np.var(accel_magnitudes),
                )
            longitudinal_accels = [
                np.dot(
                    accelerations_vec[i],
                    velocities_vec[i] / (np.linalg.norm(velocities_vec[i]) + 1e-6),
                )
                for i in range(len(accelerations_vec))
            ]
            if longitudinal_accels:
                decelerations_only = [la for la in longitudinal_accels if la < 0]
                if decelerations_only:
                    max_decel, mean_decel = np.min(decelerations_only), np.mean(
                        decelerations_only
                    )
                else:
                    max_decel, mean_decel = 0.0, 0.0
            if len(accelerations_vec) >= 2:
                dt_jerk = np.diff(timestamps[:-2])
                dt_jerk = np.where(dt_jerk == 0, 1e-6, dt_jerk)
                jerks_vec = np.diff(accelerations_vec, axis=0) / dt_jerk[:, np.newaxis]
                jerk_magnitudes = np.linalg.norm(jerks_vec, axis=1)
                if len(jerk_magnitudes) > 0:
                    max_jerk, mean_jerk = np.max(jerk_magnitudes), np.mean(
                        jerk_magnitudes
                    )
        if len(segment_vectors) > 0:
            segment_angles = np.arctan2(segment_vectors[:, 1], segment_vectors[:, 0])
            if len(segment_angles) >= 2:
                angle_diffs = smallest_angle_diff(
                    segment_angles[1:], segment_angles[:-1]
                )
                dt_ang = dt[1:][: len(angle_diffs)]
                dt_ang = np.where(dt_ang == 0, 1e-6, dt_ang)
                angular_velocities = angle_diffs / dt_ang
                if len(angular_velocities) > 0:
                    max_angular_vel = np.max(np.abs(angular_velocities))
                if len(angular_velocities) >= 2:
                    dt_ang_acc = dt_ang[1:][: len(angular_velocities) - 1]
                    dt_ang_acc = np.where(dt_ang_acc == 0, 1e-6, dt_ang_acc)
                    angular_accelerations = np.diff(angular_velocities) / dt_ang_acc
                    if len(angular_accelerations) > 0:
                        max_angular_accel = np.max(np.abs(angular_accelerations))
    features = [
        direct_dist,
        total_length,
        straightness_ratio,
        overall_angle,
        start_angle,
        end_angle,
        turn_angle_diff,
        float(is_sudden_change_flag),
        float(intersects_zone_flag),
        max_speed,
        mean_speed,
        speed_variance,
        max_accel,
        mean_accel,
        accel_variance,
        max_jerk,
        mean_jerk,
        max_angular_vel,
        max_angular_accel,
        max_decel,
        mean_decel,
    ]
    return [np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0) for f in features]


def build_enhanced_feature_matrix(
    trajs: List[np.ndarray], thr: Dict[str, Any], zone: np.ndarray
) -> np.ndarray:
    return np.array([extract_enhanced_features(t, thr, zone) for t in trajs])


# -----------------------------
# HYPERPARAMETER TUNING + TRAIN
# -----------------------------


def tune_and_train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    inner_cv = StratifiedKFold(
        n_splits=min(
            5, np.min(np.bincount(y_train.astype(int))) if len(y_train) > 0 else 5
        ),
        shuffle=True,
        random_state=random_state,
    )  # Adjust n_splits
    param_grid = {
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": [None, "sqrt", "log2"],
        "class_weight": ["balanced", None],
        "criterion": ["gini", "entropy"],
    }
    base_estimator = DecisionTreeClassifier(random_state=random_state)
    grid_search = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=inner_cv,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("\nBest Hyperparameters Found:\n", grid_search.best_params_)
    if len(X_val) > 0 and len(y_val) > 0:
        y_val_pred = best_model.predict(X_val)
        print("\n--- Validation Set Performance (with best model) ---")
        print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
        print(classification_report(y_val, y_val_pred, zero_division=0))
        print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
    else:
        print("\nValidation set is empty, skipping validation performance printout.")
    return best_model


# -----------------------------
# VISUALIZATION
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
        points = traj[:, 1:3].astype(np.int32)
        label = int(labels[i])
        color = color_map.get(label, (128, 128, 128))
        cv2.polylines(
            img_copy, [points], isClosed=False, color=color, thickness=thickness
        )
    return img_copy


def show_image_plt(image_bgr: np.ndarray, title: str = ""):
    plt.figure(figsize=(12, 9))
    plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_on_image(
    base_image_path: str,
    trajectories: List[np.ndarray],
    labels: np.ndarray,
    title: str,
    color_map: Dict[int, Tuple[int, int, int]],
    output_dir: str,
    zone_polygon: Optional[np.ndarray] = None,
):
    background_image = (
        cv2.imread(base_image_path)
        if os.path.exists(base_image_path)
        else np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    )
    if not os.path.exists(base_image_path):
        print(f"Warning: BG image {base_image_path} not found.")
    overlay_image = draw_trajectories(background_image, trajectories, labels, color_map)
    if zone_polygon is not None:
        cv2.polylines(
            overlay_image,
            [zone_polygon.reshape(-1, 1, 2)],
            isClosed=True,
            color=(255, 0, 255),
            thickness=3,
        )
    save_path = os.path.join(output_dir, f"{sanitize_filename(title)}.png")
    cv2.imwrite(save_path, overlay_image)
    print(f"Saved visualization to {save_path}")
    show_image_plt(overlay_image, title)


# THIS FUNCTION CONTAINS THE FILENAME SANITIZATION FIX
def visualize_trajectory_by_parameter(
    base_image_path: str,
    trajectory: np.ndarray,
    parameter_values: np.ndarray,
    param_name: str,  # This is the display name, e.g., "Jerk (pixels/frame^3)"
    output_dir: str,
    trajectory_id: str,
    colormap: str = "jet",
    alpha: float = 0.7,
):
    """Visualize a single trajectory colored by a specific parameter value."""
    if os.path.exists(base_image_path):
        img = cv2.imread(base_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print(
            f"Warning: Background image {base_image_path} not found for param vis. Using blank white."
        )
        img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img)
    cmap = plt.get_cmap(colormap)

    if len(parameter_values) > 0:
        min_val, max_val = np.min(parameter_values), np.max(parameter_values)
        if min_val == max_val:
            normalized_params = np.zeros_like(parameter_values, dtype=float)
        else:
            normalized_params = (parameter_values - min_val) / (max_val - min_val)
    else:
        normalized_params, min_val, max_val = [], 0, 0

    pts = trajectory[:, 1:3]
    for i in range(len(pts) - 1):
        if i < len(normalized_params):
            color = cmap(normalized_params[i])
            ax.plot(
                [pts[i, 0], pts[i + 1, 0]],
                [pts[i, 1], pts[i + 1, 1]],
                color=color,
                linewidth=3,
                alpha=alpha,
            )
        else:
            ax.plot(
                [pts[i, 0], pts[i + 1, 0]],
                [pts[i, 1], pts[i + 1, 1]],
                color="gray",
                linewidth=2,
                alpha=0.5,
            )

    if len(parameter_values) > 0:
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=min_val, vmax=max_val)
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label(f"{param_name} (Range: {min_val:.2f} to {max_val:.2f})")

    ax.set_title(f"Trajectory {trajectory_id}: Coloured by {param_name}")
    ax.axis("off")
    fig.tight_layout()

    sanitized_param_name_for_file = sanitize_filename(param_name)
    save_path = os.path.join(
        output_dir,
        f"traj_{sanitize_filename(trajectory_id)}_{sanitized_param_name_for_file}.png",
    )
    plt.savefig(save_path, dpi=150)
    print(f"Saved parameter visualization to {save_path}")
    plt.show()


# END OF CORRECTED FUNCTION


def analyze_parameter_in_trajectory(
    traj: np.ndarray,
    image_path: str,
    output_dir: str,
    traj_id: str,
    param_to_analyze: str = "jerk",
):
    """Calculates and visualizes a specific parameter (jerk, speed, accel) along a trajectory."""
    pts = traj[:, 1:3]
    timestamps = traj[:, 0]
    parameter_values = []
    param_name_for_vis = "Parameter"
    if len(pts) < 2:
        print(f"Trajectory {traj_id} too short for analysis.")
        return
    dt = np.diff(timestamps)
    dt = np.where(dt == 0, 1e-6, dt)
    segment_vectors = np.diff(pts, axis=0)

    if param_to_analyze == "speed":
        if len(segment_vectors) > 0:
            velocities_vec = segment_vectors / dt[:, np.newaxis]
            parameter_values = np.linalg.norm(velocities_vec, axis=1)
            param_name_for_vis = "Speed (pixels/frame)"
    elif param_to_analyze == "acceleration":
        if len(segment_vectors) > 1:
            velocities_vec = segment_vectors / dt[:, np.newaxis]
            dt_acc = np.diff(timestamps[:-1])
            dt_acc = np.where(dt_acc == 0, 1e-6, dt_acc)
            accelerations_vec = np.diff(velocities_vec, axis=0) / dt_acc[:, np.newaxis]
            parameter_values = np.linalg.norm(accelerations_vec, axis=1)
            param_name_for_vis = "Acceleration (pixels/frame^2)"
    elif param_to_analyze == "jerk":
        if len(segment_vectors) > 2:
            velocities_vec = segment_vectors / dt[:, np.newaxis]
            dt_acc = np.diff(timestamps[:-1])
            dt_acc = np.where(dt_acc == 0, 1e-6, dt_acc)
            accelerations_vec = np.diff(velocities_vec, axis=0) / dt_acc[:, np.newaxis]
            if len(accelerations_vec) > 1:
                dt_jerk = np.diff(timestamps[:-2])
                dt_jerk = np.where(dt_jerk == 0, 1e-6, dt_jerk)
                jerks_vec = np.diff(accelerations_vec, axis=0) / dt_jerk[:, np.newaxis]
                parameter_values = np.linalg.norm(jerks_vec, axis=1)
                param_name_for_vis = "Jerk (pixels/frame^3)"
    else:
        print(f"Unknown parameter for analysis: {param_to_analyze}")
        return
    if not list(parameter_values):
        print(f"Could not compute {param_to_analyze} for {traj_id}.")
        return
    visualize_trajectory_by_parameter(
        image_path,
        traj,
        np.array(parameter_values),
        param_name_for_vis,
        output_dir,
        traj_id,
    )


def visualize_trajectory_3d(
    trajectory: np.ndarray,
    output_dir: str,
    trajectory_id: str,
    feature_values_for_color: Optional[np.ndarray] = None,
    feature_name: Optional[str] = None,
    title_suffix: str = "",
):
    if trajectory is None or len(trajectory) < 2:
        print(f"Traj {trajectory_id} too short for 3D vis.")
        return
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    time_coords, x_coords, y_coords = (
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
    )
    title = f"3D Trajectory: {sanitize_filename(trajectory_id)} {sanitize_filename(title_suffix)}"
    if (
        feature_values_for_color is not None
        and len(feature_values_for_color) == len(x_coords) - 1
    ):
        norm = plt.Normalize(
            feature_values_for_color.min(), feature_values_for_color.max()
        )
        cmap = plt.get_cmap("viridis")
        colors = cmap(norm(feature_values_for_color))
        for i in range(len(x_coords) - 1):
            ax.plot(
                x_coords[i : i + 2],
                y_coords[i : i + 2],
                time_coords[i : i + 2],
                color=colors[i],
                marker="o",
                markersize=2,
            )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
        if feature_name:
            cbar.set_label(feature_name)
            title += f" (Colored by {sanitize_filename(feature_name)})"
    else:
        ax.plot(
            x_coords, y_coords, time_coords, marker="o", linestyle="-", markersize=2
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Time")
    ax.set_title(title)
    ax.invert_yaxis()
    save_path = os.path.join(
        output_dir,
        f"traj3D_{sanitize_filename(trajectory_id)}_{sanitize_filename(title_suffix)}.png",
    )
    plt.savefig(save_path, dpi=150)
    print(f"Saved 3D vis to {save_path}")
    plt.show()


# -----------------------------
# MAIN
# -----------------------------


def main():
    print("Starting Trajectory Analysis Script...")
    if not os.path.exists(FOLDER_PATH) or not os.listdir(FOLDER_PATH):
        print(f"Warning: Trajectory folder '{FOLDER_PATH}' empty/not found.")

    trajs, fnames = load_trajectories(FOLDER_PATH, N_POINTS)
    if not trajs:
        print("No trajectories loaded. Exiting.")
        return

    y_labels = None
    if os.path.exists(GROUND_TRUTH_CSV_PATH):
        gt_labels_raw = load_ground_truth_from_csv(GROUND_TRUTH_CSV_PATH, fnames)
        if gt_labels_raw is not None:
            valid_gt_indices = [
                i for i, label in enumerate(gt_labels_raw) if label != -1
            ]
            if len(valid_gt_indices) < len(gt_labels_raw):
                print(
                    f"Filtered {len(gt_labels_raw) - len(valid_gt_indices)} trajs (missing GT)."
                )
            trajs = [trajs[i] for i in valid_gt_indices]
            fnames = [fnames[i] for i in valid_gt_indices]
            y_labels = gt_labels_raw[valid_gt_indices]
            if not trajs:
                print("No trajs after GT filtering. Exiting.")
                return
            print(f"Using {len(y_labels)} GT labels.")
    if y_labels is None:
        print("GT labels not available/failed. Using rule-based labels.")
        rule_based_cluster_labels = [
            assign_custom_cluster_refined(t, central_zone_polygon, THRESHOLD_PARAMS)
            for t in trajs
        ]
        y_labels = map_clusters_to_binary_refined(rule_based_cluster_labels)
        print(f"Generated {len(y_labels)} rule-based binary labels.")

    print("\nExtracting enhanced features...")
    X_features = build_enhanced_feature_matrix(
        trajs, THRESHOLD_PARAMS, central_zone_polygon
    )
    feature_names = get_feature_names()
    print(f"Feature matrix shape: {X_features.shape}")
    if len(X_features) < 20:
        print("Warning: Very few samples. Splitting might be unstable.")
        # Min samples for reliable split

    indices = np.arange(len(y_labels))
    try:
        train_val_indices, test_indices, y_train_val, _ = train_test_split(
            indices, y_labels, test_size=0.2, random_state=42, stratify=y_labels
        )
        train_indices, val_indices, _, _ = train_test_split(
            train_val_indices,
            y_train_val,
            test_size=0.25,
            random_state=42,
            stratify=y_train_val,
        )
    except (
        ValueError
    ) as e:  # Handles cases where a class has too few members for stratified split
        print(f"Stratified split failed: {e}. Falling back to non-stratified split.")
        train_val_indices, test_indices = train_test_split(
            indices, test_size=0.2, random_state=42
        )
        y_train_val = y_labels[train_val_indices]  # Need this for the next split
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=0.25, random_state=42
        )

    X_train, y_train = X_features[train_indices], y_labels[train_indices]
    X_val, y_val = X_features[val_indices], y_labels[val_indices]
    X_test, y_test = X_features[test_indices], y_labels[test_indices]
    trajs_train = [trajs[i] for i in train_indices]
    trajs_val = [trajs[i] for i in val_indices]
    trajs_test = [trajs[i] for i in test_indices]
    fnames_test = [fnames[i] for i in test_indices]

    print(f"\nData split: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
    if len(y_train) > 0:
        print(f"Train class dist: {np.bincount(y_train.astype(int))}")
    if len(y_val) > 0:
        print(f"Val class dist: {np.bincount(y_val.astype(int))}")
    if len(y_test) > 0:
        print(f"Test class dist: {np.bincount(y_test.astype(int))}")

    print("\n--- Hyperparameter Tuning and Training ---")
    if len(X_train) == 0:
        print("Training set empty. Cannot train. Exiting.")
        return
    best_dt_model = tune_and_train_decision_tree(X_train, y_train, X_val, y_val)

    print("\n--- Test Set Performance ---")
    if len(X_test) > 0:
        y_test_pred = best_dt_model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
        print(classification_report(y_test, y_test_pred, zero_division=0))
        print("CM:\n", confusion_matrix(y_test, y_test_pred))
        if hasattr(best_dt_model, "feature_importances_"):
            importances = best_dt_model.feature_importances_
            sorted_indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(12, max(6, len(feature_names) // 2)))
            plt.title("DT Feature Importances")
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
        print("\n--- Test Set Visualizations ---")
        if not os.path.exists(IMAGE_PATH):
            print(f"Warning: BG image {IMAGE_PATH} not found.")
        visualize_on_image(
            IMAGE_PATH,
            trajs_test,
            y_test,
            f"Sample {SAMPLE} Test GT",
            BINARY_COLORS_BGR,
            OUTPUT_DIR,
            central_zone_polygon,
        )
        visualize_on_image(
            IMAGE_PATH,
            trajs_test,
            y_test_pred,
            f"Sample {SAMPLE} Test Pred",
            BINARY_COLORS_BGR,
            OUTPUT_DIR,
            central_zone_polygon,
        )
        if trajs_test:
            print("\n--- Parameter-Specific Trajectory Visualizations ---")
            for i in range(min(3, len(trajs_test))):
                traj_id = sanitize_filename(fnames_test[i].split(".")[0])
                print(f"\nAnalyzing params for test traj: {traj_id}")
                analyze_parameter_in_trajectory(
                    trajs_test[i], IMAGE_PATH, OUTPUT_DIR, traj_id, "jerk"
                )
                analyze_parameter_in_trajectory(
                    trajs_test[i], IMAGE_PATH, OUTPUT_DIR, traj_id, "speed"
                )
                pts_3d, timestamps_3d = trajs_test[i][:, 1:3], trajs_test[i][:, 0]
                dt_3d = np.diff(timestamps_3d)
                dt_3d = np.where(dt_3d == 0, 1e-6, dt_3d)
                speeds_3d = []
                if len(pts_3d) >= 2:
                    speeds_3d = np.linalg.norm(
                        np.diff(pts_3d, axis=0) / dt_3d[:, np.newaxis], axis=1
                    )
                visualize_trajectory_3d(
                    trajs_test[i],
                    OUTPUT_DIR,
                    traj_id,
                    speeds_3d if len(speeds_3d) > 0 else None,
                    "Speed",
                    f"GT-{y_test[i]}_Pred-{y_test_pred[i]}",
                )
    else:
        print("Test set empty. Skipping test eval & vis.")
    print("\nScript finished.")


if __name__ == "__main__":
    if not os.path.exists(FOLDER_PATH):
        os.makedirs(FOLDER_PATH, exist_ok=True)
        print(f"Created dummy folder: {FOLDER_PATH}")
    if not os.listdir(FOLDER_PATH):
        dummy_data = {
            "frameNo": list(range(1, 21)),
            "left": list(range(100, 300, 10)),
            "top": list(range(100, 200, 5)),
            "w": [10] * 20,
            "h": [10] * 20,
        }
        df_dummy = pd.DataFrame(dummy_data)
        for i in range(30):  # Increased dummy files for robust splitting
            df_temp = df_dummy.copy()
            df_temp["left"] += np.random.randint(-20, 20, size=len(df_temp))
            df_temp["top"] += np.random.randint(-20, 20, size=len(df_temp))
            df_temp.to_csv(
                os.path.join(FOLDER_PATH, f"dummy_trajectory_{i+1:02d}.csv"),
                index=False,
            )
        print(f"Created 30 dummy trajectory files in {FOLDER_PATH}")
        if not os.path.exists(IMAGE_PATH):
            cv2.imwrite(IMAGE_PATH, np.ones((720, 1280, 3), dtype=np.uint8) * 200)
            print(f"Created dummy image: {IMAGE_PATH}")
        if not os.path.exists(GROUND_TRUTH_CSV_PATH):
            os.makedirs(os.path.dirname(GROUND_TRUTH_CSV_PATH), exist_ok=True)
            gt_fnames = [f"dummy_trajectory_{i+1:02d}.csv" for i in range(30)]
            gt_labels = np.random.randint(0, 2, size=30)
            pd.DataFrame({"filename": gt_fnames, "label": gt_labels}).to_csv(
                GROUND_TRUTH_CSV_PATH, index=False
            )
            print(f"Created dummy GT CSV: {GROUND_TRUTH_CSV_PATH} with 30 entries.")
        print("Created dummy files. Replace with actual data.")
    main()
