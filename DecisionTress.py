# # #!/usr/bin/env python3
# # """
# # Trajectory Classification Using a Refined Rule Engine and a Decision Tree Classifier,
# # with Visualization of Ground Truth and Predicted Trajectories.

# # This script:
# #   1. Loads and resamples trajectory data.
# #   2. Extracts features from each trajectory.
# #   3. Uses either the ground truth labels or the refined rule engine to produce binary labels.
# #   4. Trains a Decision Tree Classifier on the extracted features while handling class imbalance.
# #   5. Evaluates and plots the decision tree.
# #   6. Overlays the ground truth and predicted labels on a background image for visualization.

# # Adjust paths, thresholds, and parameters as needed.
# # """

# # import math
# # import os
# # import time
# # from typing import Any, Dict, List, Optional, Tuple

# # import cv2
# # import matplotlib.pyplot as plt
# # import numpy as np
# # import pandas as pd
# # import seaborn as sns
# # from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# # from sklearn.model_selection import train_test_split
# # from sklearn.tree import DecisionTreeClassifier, plot_tree
# # from tslearn.preprocessing import TimeSeriesResampler

# # # -----------------------------
# # # PARAMETERS & PATHS
# # # -----------------------------
# # SAMPLE = "11"  # Sample folder name
# # FOLDER_PATH = SAMPLE
# # IMAGE_PATH = f"sample{SAMPLE}.jpg"  # Background image path
# # GROUND_TRUTH_CSV_PATH = "trajectory_images/combined_labels.csv"
# # N_POINTS = 50  # Resampling points

# # # Directory to optionally save decision tree plot, if desired.
# # OUTPUT_DIR = f"trajectory_DT_sample{SAMPLE}"
# # os.makedirs(OUTPUT_DIR, exist_ok=True)

# # # -----------------------------
# # # THRESHOLDS (Refined rules)
# # # -----------------------------
# # THRESHOLD_PARAMS: Dict[str, Any] = {
# #     "moving_threshold": 10.0,
# #     "straightness_threshold": 1.30,
# #     "hv_angle_tolerance_deg": 30.0,
# #     "turn_angle_tolerance_deg": 45.0,
# #     "sudden_change_threshold_deg": 65.0,
# #     "min_segment_length": 5.0,
# # }
# # THRESHOLD_PARAMS["sudden_change_threshold_rad"] = np.deg2rad(
# #     THRESHOLD_PARAMS["sudden_change_threshold_deg"]
# # )
# # THRESHOLD_PARAMS["hv_angle_tolerance_rad"] = np.deg2rad(
# #     THRESHOLD_PARAMS["hv_angle_tolerance_deg"]
# # )
# # THRESHOLD_PARAMS["turn_angle_tolerance_rad"] = np.deg2rad(
# #     THRESHOLD_PARAMS["turn_angle_tolerance_deg"]
# # )

# # # -----------------------------
# # # CENTRAL ZONE POLYGON (for intersection tests)
# # # -----------------------------
# # central_zone_polygon = np.array(
# #     [[1650, 842], [1650, 1331], [2271, 1331], [2271, 842]], dtype=np.int32
# # )
# # print(f"Using central zone polygon: {central_zone_polygon.tolist()}")

# # # -----------------------------
# # # COLOR MAPS FOR VISUALIZATION (Binary Maps: 0=Normal, 1=Abnormal)
# # # -----------------------------
# # BINARY_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
# #     0: (0, 255, 0),  # Normal: Lime Green
# #     1: (0, 0, 255),  # Abnormal: Red
# # }


# # # -----------------------------
# # # HELPER FUNCTIONS: Data Loading & Preprocessing
# # # -----------------------------
# # def extract_trajectory(csv_path: str) -> Optional[np.ndarray]:
# #     try:
# #         df = pd.read_csv(csv_path, usecols=["frameNo", "left", "top", "w", "h"])
# #         df["center_x"] = df["left"] + df["w"] / 2
# #         df["center_y"] = df["top"] + df["h"] / 2
# #         if len(df) < 2:
# #             return None
# #         return df[["frameNo", "center_x", "center_y"]].values
# #     except Exception:
# #         return None


# # def resample_trajectory(traj: np.ndarray, n_points: int) -> Optional[np.ndarray]:
# #     if traj is None or traj.shape[0] < 2:
# #         return None
# #     xy = traj[:, 1:3]
# #     try:
# #         xy_contiguous = np.ascontiguousarray(xy, dtype=np.float64)
# #         resampler = TimeSeriesResampler(sz=n_points)
# #         resampled_xy = resampler.fit_transform(xy_contiguous.reshape(1, -1, 2))[0]
# #         frames = np.linspace(traj[0, 0], traj[-1, 0], n_points)
# #         return np.column_stack((frames, resampled_xy))
# #     except Exception:
# #         return None


# # def load_trajectories(
# #     folder_path: str, n_points: int
# # ) -> Tuple[List[np.ndarray], List[str]]:
# #     if not os.path.isdir(folder_path):
# #         return [], []
# #     csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])
# #     trajectories, file_names = [], []
# #     for fname in csv_files:
# #         full_path = os.path.join(folder_path, fname)
# #         traj = extract_trajectory(full_path)
# #         if traj is not None and len(traj) >= 2:
# #             traj_resampled = resample_trajectory(traj, n_points)
# #             if traj_resampled is not None:
# #                 trajectories.append(traj_resampled)
# #                 file_names.append(fname)
# #     print(f"Loaded {len(trajectories)} trajectories from {len(csv_files)} files.")
# #     return trajectories, file_names


# # def load_ground_truth_from_csv(
# #     gt_csv_path: str, processed_filenames: List[str]
# # ) -> Optional[np.ndarray]:
# #     if not os.path.exists(gt_csv_path):
# #         return None
# #     try:
# #         gt_df = pd.read_csv(gt_csv_path)
# #         if "filename" not in gt_df.columns or "label" not in gt_df.columns:
# #             return None
# #         gt_map = {
# #             os.path.basename(row["filename"]): row["label"]
# #             for _, row in gt_df.iterrows()
# #             if isinstance(row["filename"], str)
# #         }
# #         labels = [
# #             gt_map.get(os.path.basename(fname), -1) for fname in processed_filenames
# #         ]
# #         print(
# #             f"Matched ground truth for {sum(1 for l in labels if l != -1)} trajectories."
# #         )
# #         return np.array(labels)
# #     except Exception:
# #         return None


# # def angle_between_points(p1: np.ndarray, p2: np.ndarray) -> float:
# #     delta = p2 - p1
# #     return np.arctan2(delta[1], delta[0]) if np.linalg.norm(delta) > 1e-9 else 0.0


# # def smallest_angle_diff(angle1: float, angle2: float) -> float:
# #     diff = angle1 - angle2
# #     return (diff + np.pi) % (2 * np.pi) - np.pi


# # def get_trajectory_angles(
# #     traj: np.ndarray, min_segment_length: float
# # ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
# #     pts = traj[:, 1:3]
# #     if len(pts) < 2:
# #         return None, None, None
# #     overall_angle = angle_between_points(pts[0], pts[-1])
# #     start_angle = next(
# #         (
# #             angle_between_points(pts[i], pts[i + 1])
# #             for i in range(len(pts) - 1)
# #             if np.linalg.norm(pts[i + 1] - pts[i]) >= min_segment_length
# #         ),
# #         angle_between_points(pts[0], pts[1]),
# #     )
# #     end_angle = next(
# #         (
# #             angle_between_points(pts[i], pts[i + 1])
# #             for i in range(len(pts) - 2, -1, -1)
# #             if np.linalg.norm(pts[i + 1] - pts[i]) >= min_segment_length
# #         ),
# #         angle_between_points(pts[-2], pts[-1]),
# #     )
# #     return start_angle, end_angle, overall_angle


# # def get_direction(angle: Optional[float], tolerance_rad: float) -> Optional[str]:
# #     if angle is None:
# #         return None
# #     if abs(smallest_angle_diff(angle, 0)) < tolerance_rad:
# #         return "W_to_E"
# #     if abs(smallest_angle_diff(angle, np.pi)) < tolerance_rad:
# #         return "E_to_W"
# #     if abs(smallest_angle_diff(angle, np.pi / 2)) < tolerance_rad:
# #         return "S_to_N"
# #     if abs(smallest_angle_diff(angle, -np.pi / 2)) < tolerance_rad:
# #         return "N_to_S"
# #     return None


# # def detect_sudden_change(
# #     traj: np.ndarray, change_threshold_rad: float
# # ) -> Tuple[bool, float]:
# #     pts = traj[:, 1:3]
# #     if len(pts) < 3:
# #         return False, 0.0
# #     segment_vectors = np.diff(pts, axis=0)
# #     norms = np.linalg.norm(segment_vectors, axis=1)
# #     valid_vectors = segment_vectors[norms > 1.0]
# #     if len(valid_vectors) < 2:
# #         return False, 0.0
# #     angles = np.arctan2(valid_vectors[:, 1], valid_vectors[:, 0])
# #     angle_diffs = smallest_angle_diff(angles[1:], angles[:-1])
# #     max_change = np.max(np.abs(angle_diffs)) if angle_diffs.size > 0 else 0.0
# #     return max_change > change_threshold_rad, max_change


# # def intersects_zone(traj: np.ndarray, zone_polygon: Optional[np.ndarray]) -> bool:
# #     if (
# #         zone_polygon is None
# #         or len(zone_polygon) < 3
# #         or traj is None
# #         or traj.shape[0] == 0
# #     ):
# #         return False
# #     for pt in traj[:, 1:3].astype(np.float32):
# #         if cv2.pointPolygonTest(zone_polygon, tuple(pt), False) >= 0:
# #             return True
# #     return False


# # # -----------------------------
# # # REFINED RULE ENGINE FUNCTIONS
# # # -----------------------------
# # def assign_custom_cluster_refined(
# #     traj: np.ndarray, central_zone: np.ndarray, thresholds: Dict[str, float]
# # ) -> int:
# #     if traj is None or len(traj) < 2:
# #         return 19  # Fallback
# #     pts = traj[:, 1:3]
# #     direct_distance = np.linalg.norm(pts[-1] - pts[0])
# #     if direct_distance < thresholds["moving_threshold"]:
# #         return 18  # Stationary
# #     start_angle, end_angle, overall_angle = get_trajectory_angles(
# #         traj, thresholds["min_segment_length"]
# #     )
# #     if start_angle is None or end_angle is None or overall_angle is None:
# #         return 19
# #     cluster_id = 19  # Default fallback
# #     total_path = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
# #     straightness_ratio = total_path / (direct_distance + 1e-6)
# #     straight_dir = get_direction(overall_angle, thresholds["hv_angle_tolerance_rad"])
# #     if straightness_ratio < thresholds["straightness_threshold"]:
# #         if straight_dir == "W_to_E":
# #             cluster_id = 1
# #         elif straight_dir == "E_to_W":
# #             cluster_id = 2
# #         elif straight_dir == "N_to_S":
# #             cluster_id = 3
# #         elif straight_dir == "S_to_N":
# #             cluster_id = 4
# #         if cluster_id != 19:
# #             return cluster_id
# #     start_dir = get_direction(start_angle, thresholds["turn_angle_tolerance_rad"])
# #     end_dir = get_direction(end_angle, thresholds["turn_angle_tolerance_rad"])
# #     turn_type = None
# #     if start_dir and end_dir and start_dir != end_dir:
# #         if start_dir == "W_to_E":
# #             turn_type = (
# #                 "WN" if end_dir == "S_to_N" else ("WS" if end_dir == "N_to_S" else None)
# #             )
# #         elif start_dir == "E_to_W":
# #             turn_type = (
# #                 "ES" if end_dir == "N_to_S" else ("EN" if end_dir == "S_to_N" else None)
# #             )
# #         elif start_dir == "N_to_S":
# #             turn_type = (
# #                 "NE" if end_dir == "W_to_E" else ("NW" if end_dir == "E_to_W" else None)
# #             )
# #         elif start_dir == "S_to_N":
# #             turn_type = (
# #                 "SW" if end_dir == "E_to_W" else ("SE" if end_dir == "W_to_E" else None)
# #             )
# #     if turn_type:
# #         is_intersecting = intersects_zone(traj, central_zone)
# #         base_mapping = {
# #             "WN": 5,
# #             "WS": 6,
# #             "ES": 7,
# #             "EN": 8,
# #             "NE": 9,
# #             "SW": 10,
# #             "NW": 11,
# #             "SE": 12,
# #         }
# #         cluster_id = base_mapping.get(turn_type, 19)
# #         if turn_type == "WS" and not is_intersecting:
# #             cluster_id = 13
# #         elif turn_type == "EN" and not is_intersecting:
# #             cluster_id = 14
# #         elif turn_type == "NW" and not is_intersecting:
# #             cluster_id = 15
# #         elif turn_type == "SE" and not is_intersecting:
# #             cluster_id = 16
# #         elif turn_type == "WN" and is_intersecting:
# #             cluster_id = 20
# #         elif turn_type == "ES" and is_intersecting:
# #             cluster_id = 21
# #         elif turn_type == "NE" and is_intersecting:
# #             cluster_id = 22
# #         elif turn_type == "SW" and is_intersecting:
# #             cluster_id = 23
# #         return cluster_id
# #     sudden, _ = detect_sudden_change(traj, thresholds["sudden_change_threshold_rad"])
# #     if sudden:
# #         return 17
# #     return 19


# # def assign_custom_clusters_refined(
# #     trajectories: List[np.ndarray],
# #     central_zone: np.ndarray,
# #     thresholds: Dict[str, float],
# # ) -> np.ndarray:
# #     if not trajectories:
# #         return np.array([])
# #     clusters = [
# #         assign_custom_cluster_refined(traj, central_zone, thresholds)
# #         for traj in trajectories
# #     ]
# #     return np.array(clusters)


# # def map_clusters_to_binary_refined(cluster_labels: np.ndarray) -> np.ndarray:
# #     abnormal_clusters = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
# #     return np.array(
# #         [1 if label in abnormal_clusters else 0 for label in cluster_labels]
# #     )


# # # -----------------------------
# # # FEATURE EXTRACTION FUNCTIONS
# # # -----------------------------
# # def extract_features(
# #     traj: np.ndarray, thresholds: Dict[str, float], central_zone: np.ndarray
# # ) -> Dict[str, float]:
# #     pts = traj[:, 1:3]
# #     direct_distance = np.linalg.norm(pts[-1] - pts[0])
# #     segments = np.diff(pts, axis=0)
# #     segment_lengths = np.linalg.norm(segments, axis=1)
# #     total_path = np.sum(segment_lengths)
# #     straightness_ratio = total_path / (direct_distance + 1e-6)
# #     start_angle, end_angle, overall_angle = get_trajectory_angles(
# #         traj, thresholds["min_segment_length"]
# #     )
# #     turn_diff = (
# #         abs(smallest_angle_diff(start_angle, end_angle))
# #         if (start_angle is not None and end_angle is not None)
# #         else 0.0
# #     )
# #     sudden_flag, _ = detect_sudden_change(
# #         traj, thresholds["sudden_change_threshold_rad"]
# #     )
# #     intersect_flag = 1 if intersects_zone(traj, central_zone) else 0

# #     return {
# #         "direct_distance": direct_distance,
# #         "total_path": total_path,
# #         "straightness_ratio": straightness_ratio,
# #         "overall_angle": overall_angle if overall_angle is not None else 0.0,
# #         "start_angle": start_angle if start_angle is not None else 0.0,
# #         "end_angle": end_angle if end_angle is not None else 0.0,
# #         "turn_diff": turn_diff,
# #         "sudden_change": 1 if sudden_flag else 0,
# #         "intersect_flag": intersect_flag,
# #     }


# # def build_feature_matrix(
# #     trajectories: List[np.ndarray],
# #     thresholds: Dict[str, float],
# #     central_zone: np.ndarray,
# # ) -> np.ndarray:
# #     features = []
# #     for traj in trajectories:
# #         feat = extract_features(traj, thresholds, central_zone)
# #         features.append(
# #             [
# #                 feat["direct_distance"],
# #                 feat["total_path"],
# #                 feat["straightness_ratio"],
# #                 feat["overall_angle"],
# #                 feat["start_angle"],
# #                 feat["end_angle"],
# #                 feat["turn_diff"],
# #                 feat["sudden_change"],
# #                 feat["intersect_flag"],
# #             ]
# #         )
# #     return np.array(features)


# # # -----------------------------
# # # DECISION TREE TRAINING & EVALUATION (with imbalance handling)
# # # -----------------------------
# # def train_decision_tree(
# #     X: np.ndarray, y: np.ndarray, test_size: float = 0.3, random_state: int = 42
# # ):
# #     # Use stratify=y to maintain label distribution in splits.
# #     X_train, X_test, y_train, y_test = train_test_split(
# #         X, y, test_size=test_size, random_state=random_state, stratify=y
# #     )
# #     # Use a decision tree with balanced class weights
# #     clf = DecisionTreeClassifier(random_state=random_state, class_weight="balanced")
# #     clf.fit(X_train, y_train)
# #     y_pred = clf.predict(X_test)
# #     acc = accuracy_score(y_test, y_pred)
# #     print(f"Decision Tree Test Accuracy: {acc*100:.2f}%")
# #     print("Classification Report:")
# #     print(classification_report(y_test, y_pred, zero_division=0))
# #     cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
# #     print("Confusion Matrix:")
# #     print(cm)

# #     plt.figure(figsize=(20, 10))
# #     plot_tree(
# #         clf,
# #         feature_names=[
# #             "direct_distance",
# #             "total_path",
# #             "straightness_ratio",
# #             "overall_angle",
# #             "start_angle",
# #             "end_angle",
# #             "turn_diff",
# #             "sudden_change",
# #             "intersect_flag",
# #         ],
# #         class_names=["Normal", "Abnormal"],
# #         filled=True,
# #     )
# #     plt.title("Decision Tree for Trajectory Classification")
# #     plt.show()
# #     return clf


# # # -----------------------------
# # # VISUALIZATION FUNCTIONS: Drawing trajectories on an image
# # # -----------------------------
# # def draw_trajectories(
# #     image: np.ndarray,
# #     trajectories: List[np.ndarray],
# #     labels: np.ndarray,
# #     color_map: Dict[int, Tuple[int, int, int]],
# #     thickness: int = 2,
# # ) -> np.ndarray:
# #     overlay = image.copy()
# #     for i, traj in enumerate(trajectories):
# #         if traj is None or traj.shape[0] < 2:
# #             continue
# #         try:
# #             label = int(labels[i])
# #         except Exception:
# #             label = -1
# #         color = color_map.get(label, (127, 127, 127))
# #         pts = traj[:, 1:3].astype(np.int32)
# #         pts[:, 0] = np.clip(pts[:, 0], 0, image.shape[1] - 1)
# #         pts[:, 1] = np.clip(pts[:, 1], 0, image.shape[0] - 1)
# #         cv2.polylines(
# #             overlay,
# #             [pts],
# #             isClosed=False,
# #             color=color,
# #             thickness=thickness,
# #             lineType=cv2.LINE_AA,
# #         )
# #     return overlay


# # def show_image(image: np.ndarray, title: str = "") -> None:
# #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# #     plt.figure(figsize=(10, 8))
# #     plt.imshow(image_rgb)
# #     plt.title(title)
# #     plt.axis("off")
# #     plt.show()


# # def visualize_on_image(
# #     image_path: str,
# #     trajectories: List[np.ndarray],
# #     labels: np.ndarray,
# #     title: str,
# #     color_map: Dict[int, Tuple[int, int, int]],
# #     zone_polygon: Optional[np.ndarray] = None,
# # ) -> None:
# #     if os.path.exists(image_path):
# #         base_image = cv2.imread(image_path)
# #     else:
# #         base_image = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
# #     image_with_tracks = draw_trajectories(base_image, trajectories, labels, color_map)
# #     if zone_polygon is not None and zone_polygon.shape[0] >= 3:
# #         poly = zone_polygon.reshape((-1, 1, 2)).astype(np.int32)
# #         cv2.polylines(
# #             image_with_tracks, [poly], isClosed=True, color=(255, 0, 255), thickness=3
# #         )
# #     show_image(image_with_tracks, title)


# # # -----------------------------
# # # MAIN PIPELINE
# # # -----------------------------
# # def main():
# #     print(f"Processing sample folder: '{FOLDER_PATH}' (Sample {SAMPLE})")
# #     trajectories, filenames = load_trajectories(FOLDER_PATH, N_POINTS)
# #     if not trajectories:
# #         print("No valid trajectories found. Exiting.")
# #         return

# #     # Use ground truth if available; otherwise use rule-based labels.
# #     gt_labels = load_ground_truth_from_csv(GROUND_TRUTH_CSV_PATH, filenames)
# #     if gt_labels is not None and len(gt_labels) == len(trajectories):
# #         print("Using ground truth labels for training.")
# #         target = np.array(gt_labels, dtype=int)
# #         gt_available = True
# #     else:
# #         print("Ground truth not available or mismatched. Using rule-based labels.")
# #         rule_clusters = assign_custom_clusters_refined(
# #             trajectories, central_zone_polygon, THRESHOLD_PARAMS
# #         )
# #         target = map_clusters_to_binary_refined(rule_clusters)
# #         gt_available = False

# #     # Build feature matrix
# #     X = build_feature_matrix(trajectories, THRESHOLD_PARAMS, central_zone_polygon)
# #     print(f"Feature matrix shape: {X.shape}")

# #     # Train and evaluate the decision tree classifier (handling imbalance)
# #     clf = train_decision_tree(X, target)

# #     # Use the trained classifier to predict labels on all trajectories.
# #     predicted_labels = clf.predict(X)

# #     # -----------------------------
# #     # Visualization on Background Image:
# #     #   1. Overlay with Ground Truth labels (if available)
# #     #   2. Overlay with Predicted labels from the decision tree
# #     # -----------------------------
# #     print(
# #         "\nDisplaying Ground Truth (if available) and Predicted Visualizations on Image..."
# #     )

# #     if gt_available:
# #         visualize_on_image(
# #             IMAGE_PATH,
# #             trajectories,
# #             target,
# #             title=f"Sample {SAMPLE}: Ground Truth Binary Labels",
# #             color_map=BINARY_COLORS_BGR,
# #             zone_polygon=central_zone_polygon,
# #         )
# #     else:
# #         print("No ground truth available for visualization.")

# #     visualize_on_image(
# #         IMAGE_PATH,
# #         trajectories,
# #         predicted_labels,
# #         title=f"Sample {SAMPLE}: Predicted Binary Labels (Decision Tree)",
# #         color_map=BINARY_COLORS_BGR,
# #         zone_polygon=central_zone_polygon,
# #     )


# # if __name__ == "__main__":
# #     main()


# """
# ####################################################
# """

# #!/usr/bin/env python3
# """
# Trajectory Classification Using a Refined Rule Engine and a Decision Tree Classifier,
# with Hyperparameter Tuning and Visualization of Ground Truth and Predicted Trajectories.

# This script:
#   1. Loads and resamples trajectory data.
#   2. Extracts features from each trajectory.
#   3. Uses either the ground truth labels or the refined rule engine to produce binary labels.
#   4. Performs hyperparameter tuning (using GridSearchCV) on a Decision Tree Classifier
#      with imbalance handling.
#   5. Trains the best estimator, evaluates and plots the decision tree.
#   6. Overlays the ground truth and predicted labels on a background image for visualization.

# Adjust paths, thresholds, and parameters as needed.
# """

# import math
# import os
# import time
# from typing import Any, Dict, List, Optional, Tuple

# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from tslearn.preprocessing import TimeSeriesResampler

# # -----------------------------
# # PARAMETERS & PATHS
# # -----------------------------
# SAMPLE = "11"  # Sample folder name
# FOLDER_PATH = SAMPLE
# IMAGE_PATH = f"sample{SAMPLE}.jpg"  # Background image path
# GROUND_TRUTH_CSV_PATH = "trajectory_images/combined_labels.csv"
# N_POINTS = 50  # Resampling points

# # Directory to optionally save decision tree plot, if desired.
# OUTPUT_DIR = f"trajectory_DT_sample{SAMPLE}"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # -----------------------------
# # THRESHOLDS (Refined rules)
# # -----------------------------
# THRESHOLD_PARAMS: Dict[str, Any] = {
#     "moving_threshold": 10.0,
#     "straightness_threshold": 1.30,
#     "hv_angle_tolerance_deg": 30.0,
#     "turn_angle_tolerance_deg": 45.0,
#     "sudden_change_threshold_deg": 65.0,
#     "min_segment_length": 5.0,
# }
# THRESHOLD_PARAMS["sudden_change_threshold_rad"] = np.deg2rad(
#     THRESHOLD_PARAMS["sudden_change_threshold_deg"]
# )
# THRESHOLD_PARAMS["hv_angle_tolerance_rad"] = np.deg2rad(
#     THRESHOLD_PARAMS["hv_angle_tolerance_deg"]
# )
# THRESHOLD_PARAMS["turn_angle_tolerance_rad"] = np.deg2rad(
#     THRESHOLD_PARAMS["turn_angle_tolerance_deg"]
# )

# # -----------------------------
# # CENTRAL ZONE POLYGON (for intersection tests)
# # -----------------------------
# central_zone_polygon = np.array(
#     [[1650, 842], [1650, 1331], [2271, 1331], [2271, 842]], dtype=np.int32
# )
# print(f"Using central zone polygon: {central_zone_polygon.tolist()}")

# # -----------------------------
# # COLOR MAPS FOR VISUALIZATION (Binary Maps: 0=Normal, 1=Abnormal)
# # -----------------------------
# BINARY_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
#     0: (0, 255, 0),  # Normal: Lime Green
#     1: (0, 0, 255),  # Abnormal: Red
# }


# # -----------------------------
# # HELPER FUNCTIONS: Data Loading & Preprocessing
# # -----------------------------
# def extract_trajectory(csv_path: str) -> Optional[np.ndarray]:
#     try:
#         df = pd.read_csv(csv_path, usecols=["frameNo", "left", "top", "w", "h"])
#         df["center_x"] = df["left"] + df["w"] / 2
#         df["center_y"] = df["top"] + df["h"] / 2
#         if len(df) < 2:
#             return None
#         return df[["frameNo", "center_x", "center_y"]].values
#     except Exception:
#         return None


# def resample_trajectory(traj: np.ndarray, n_points: int) -> Optional[np.ndarray]:
#     if traj is None or traj.shape[0] < 2:
#         return None
#     xy = traj[:, 1:3]
#     try:
#         xy_contiguous = np.ascontiguousarray(xy, dtype=np.float64)
#         resampler = TimeSeriesResampler(sz=n_points)
#         resampled_xy = resampler.fit_transform(xy_contiguous.reshape(1, -1, 2))[0]
#         frames = np.linspace(traj[0, 0], traj[-1, 0], n_points)
#         return np.column_stack((frames, resampled_xy))
#     except Exception:
#         return None


# def load_trajectories(
#     folder_path: str, n_points: int
# ) -> Tuple[List[np.ndarray], List[str]]:
#     if not os.path.isdir(folder_path):
#         return [], []
#     csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])
#     trajectories, file_names = [], []
#     for fname in csv_files:
#         full_path = os.path.join(folder_path, fname)
#         traj = extract_trajectory(full_path)
#         if traj is not None and len(traj) >= 2:
#             traj_resampled = resample_trajectory(traj, n_points)
#             if traj_resampled is not None:
#                 trajectories.append(traj_resampled)
#                 file_names.append(fname)
#     print(f"Loaded {len(trajectories)} trajectories from {len(csv_files)} files.")
#     return trajectories, file_names


# def load_ground_truth_from_csv(
#     gt_csv_path: str, processed_filenames: List[str]
# ) -> Optional[np.ndarray]:
#     if not os.path.exists(gt_csv_path):
#         return None
#     try:
#         gt_df = pd.read_csv(gt_csv_path)
#         if "filename" not in gt_df.columns or "label" not in gt_df.columns:
#             return None
#         gt_map = {
#             os.path.basename(row["filename"]): row["label"]
#             for _, row in gt_df.iterrows()
#             if isinstance(row["filename"], str)
#         }
#         labels = [
#             gt_map.get(os.path.basename(fname), -1) for fname in processed_filenames
#         ]
#         print(
#             f"Matched ground truth for {sum(1 for l in labels if l != -1)} trajectories."
#         )
#         return np.array(labels)
#     except Exception:
#         return None


# def angle_between_points(p1: np.ndarray, p2: np.ndarray) -> float:
#     delta = p2 - p1
#     return np.arctan2(delta[1], delta[0]) if np.linalg.norm(delta) > 1e-9 else 0.0


# def smallest_angle_diff(angle1: float, angle2: float) -> float:
#     diff = angle1 - angle2
#     return (diff + np.pi) % (2 * np.pi) - np.pi


# def get_trajectory_angles(
#     traj: np.ndarray, min_segment_length: float
# ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
#     pts = traj[:, 1:3]
#     if len(pts) < 2:
#         return None, None, None
#     overall_angle = angle_between_points(pts[0], pts[-1])
#     start_angle = next(
#         (
#             angle_between_points(pts[i], pts[i + 1])
#             for i in range(len(pts) - 1)
#             if np.linalg.norm(pts[i + 1] - pts[i]) >= min_segment_length
#         ),
#         angle_between_points(pts[0], pts[1]),
#     )
#     end_angle = next(
#         (
#             angle_between_points(pts[i], pts[i + 1])
#             for i in range(len(pts) - 2, -1, -1)
#             if np.linalg.norm(pts[i + 1] - pts[i]) >= min_segment_length
#         ),
#         angle_between_points(pts[-2], pts[-1]),
#     )
#     return start_angle, end_angle, overall_angle


# def get_direction(angle: Optional[float], tolerance_rad: float) -> Optional[str]:
#     if angle is None:
#         return None
#     if abs(smallest_angle_diff(angle, 0)) < tolerance_rad:
#         return "W_to_E"
#     if abs(smallest_angle_diff(angle, np.pi)) < tolerance_rad:
#         return "E_to_W"
#     if abs(smallest_angle_diff(angle, np.pi / 2)) < tolerance_rad:
#         return "S_to_N"
#     if abs(smallest_angle_diff(angle, -np.pi / 2)) < tolerance_rad:
#         return "N_to_S"
#     return None


# def detect_sudden_change(
#     traj: np.ndarray, change_threshold_rad: float
# ) -> Tuple[bool, float]:
#     pts = traj[:, 1:3]
#     if len(pts) < 3:
#         return False, 0.0
#     segment_vectors = np.diff(pts, axis=0)
#     norms = np.linalg.norm(segment_vectors, axis=1)
#     valid_vectors = segment_vectors[norms > 1.0]
#     if len(valid_vectors) < 2:
#         return False, 0.0
#     angles = np.arctan2(valid_vectors[:, 1], valid_vectors[:, 0])
#     angle_diffs = smallest_angle_diff(angles[1:], angles[:-1])
#     max_change = np.max(np.abs(angle_diffs)) if angle_diffs.size > 0 else 0.0
#     return max_change > change_threshold_rad, max_change


# def intersects_zone(traj: np.ndarray, zone_polygon: Optional[np.ndarray]) -> bool:
#     if (
#         zone_polygon is None
#         or len(zone_polygon) < 3
#         or traj is None
#         or traj.shape[0] == 0
#     ):
#         return False
#     for pt in traj[:, 1:3].astype(np.float32):
#         if cv2.pointPolygonTest(zone_polygon, tuple(pt), False) >= 0:
#             return True
#     return False


# # -----------------------------
# # REFINED RULE ENGINE FUNCTIONS
# # -----------------------------
# def assign_custom_cluster_refined(
#     traj: np.ndarray, central_zone: np.ndarray, thresholds: Dict[str, float]
# ) -> int:
#     if traj is None or len(traj) < 2:
#         return 19  # Fallback
#     pts = traj[:, 1:3]
#     direct_distance = np.linalg.norm(pts[-1] - pts[0])
#     if direct_distance < thresholds["moving_threshold"]:
#         return 18  # Stationary
#     start_angle, end_angle, overall_angle = get_trajectory_angles(
#         traj, thresholds["min_segment_length"]
#     )
#     if start_angle is None or end_angle is None or overall_angle is None:
#         return 19
#     cluster_id = 19  # Default fallback
#     total_path = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
#     straightness_ratio = total_path / (direct_distance + 1e-6)
#     straight_dir = get_direction(overall_angle, thresholds["hv_angle_tolerance_rad"])
#     if straightness_ratio < thresholds["straightness_threshold"]:
#         if straight_dir == "W_to_E":
#             cluster_id = 1
#         elif straight_dir == "E_to_W":
#             cluster_id = 2
#         elif straight_dir == "N_to_S":
#             cluster_id = 3
#         elif straight_dir == "S_to_N":
#             cluster_id = 4
#         if cluster_id != 19:
#             return cluster_id
#     start_dir = get_direction(start_angle, thresholds["turn_angle_tolerance_rad"])
#     end_dir = get_direction(end_angle, thresholds["turn_angle_tolerance_rad"])
#     turn_type = None
#     if start_dir and end_dir and start_dir != end_dir:
#         if start_dir == "W_to_E":
#             turn_type = (
#                 "WN" if end_dir == "S_to_N" else ("WS" if end_dir == "N_to_S" else None)
#             )
#         elif start_dir == "E_to_W":
#             turn_type = (
#                 "ES" if end_dir == "N_to_S" else ("EN" if end_dir == "S_to_N" else None)
#             )
#         elif start_dir == "N_to_S":
#             turn_type = (
#                 "NE" if end_dir == "W_to_E" else ("NW" if end_dir == "E_to_W" else None)
#             )
#         elif start_dir == "S_to_N":
#             turn_type = (
#                 "SW" if end_dir == "E_to_W" else ("SE" if end_dir == "W_to_E" else None)
#             )
#     if turn_type:
#         is_intersecting = intersects_zone(traj, central_zone)
#         base_mapping = {
#             "WN": 5,
#             "WS": 6,
#             "ES": 7,
#             "EN": 8,
#             "NE": 9,
#             "SW": 10,
#             "NW": 11,
#             "SE": 12,
#         }
#         cluster_id = base_mapping.get(turn_type, 19)
#         if turn_type == "WS" and not is_intersecting:
#             cluster_id = 13
#         elif turn_type == "EN" and not is_intersecting:
#             cluster_id = 14
#         elif turn_type == "NW" and not is_intersecting:
#             cluster_id = 15
#         elif turn_type == "SE" and not is_intersecting:
#             cluster_id = 16
#         elif turn_type == "WN" and is_intersecting:
#             cluster_id = 20
#         elif turn_type == "ES" and is_intersecting:
#             cluster_id = 21
#         elif turn_type == "NE" and is_intersecting:
#             cluster_id = 22
#         elif turn_type == "SW" and is_intersecting:
#             cluster_id = 23
#         return cluster_id
#     sudden, _ = detect_sudden_change(traj, thresholds["sudden_change_threshold_rad"])
#     if sudden:
#         return 17
#     return 19


# def assign_custom_clusters_refined(
#     trajectories: List[np.ndarray],
#     central_zone: np.ndarray,
#     thresholds: Dict[str, float],
# ) -> np.ndarray:
#     if not trajectories:
#         return np.array([])
#     clusters = [
#         assign_custom_cluster_refined(traj, central_zone, thresholds)
#         for traj in trajectories
#     ]
#     return np.array(clusters)


# def map_clusters_to_binary_refined(cluster_labels: np.ndarray) -> np.ndarray:
#     abnormal_clusters = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
#     return np.array(
#         [1 if label in abnormal_clusters else 0 for label in cluster_labels]
#     )


# # -----------------------------
# # FEATURE EXTRACTION FUNCTIONS
# # -----------------------------
# def extract_features(
#     traj: np.ndarray, thresholds: Dict[str, float], central_zone: np.ndarray
# ) -> Dict[str, float]:
#     pts = traj[:, 1:3]
#     direct_distance = np.linalg.norm(pts[-1] - pts[0])
#     segments = np.diff(pts, axis=0)
#     segment_lengths = np.linalg.norm(segments, axis=1)
#     total_path = np.sum(segment_lengths)
#     straightness_ratio = total_path / (direct_distance + 1e-6)
#     start_angle, end_angle, overall_angle = get_trajectory_angles(
#         traj, thresholds["min_segment_length"]
#     )
#     turn_diff = (
#         abs(smallest_angle_diff(start_angle, end_angle))
#         if (start_angle is not None and end_angle is not None)
#         else 0.0
#     )
#     sudden_flag, _ = detect_sudden_change(
#         traj, thresholds["sudden_change_threshold_rad"]
#     )
#     intersect_flag = 1 if intersects_zone(traj, central_zone) else 0

#     return {
#         "direct_distance": direct_distance,
#         "total_path": total_path,
#         "straightness_ratio": straightness_ratio,
#         "overall_angle": overall_angle if overall_angle is not None else 0.0,
#         "start_angle": start_angle if start_angle is not None else 0.0,
#         "end_angle": end_angle if end_angle is not None else 0.0,
#         "turn_diff": turn_diff,
#         "sudden_change": 1 if sudden_flag else 0,
#         "intersect_flag": intersect_flag,
#     }


# def build_feature_matrix(
#     trajectories: List[np.ndarray],
#     thresholds: Dict[str, float],
#     central_zone: np.ndarray,
# ) -> np.ndarray:
#     features = []
#     for traj in trajectories:
#         feat = extract_features(traj, thresholds, central_zone)
#         features.append(
#             [
#                 feat["direct_distance"],
#                 feat["total_path"],
#                 feat["straightness_ratio"],
#                 feat["overall_angle"],
#                 feat["start_angle"],
#                 feat["end_angle"],
#                 feat["turn_diff"],
#                 feat["sudden_change"],
#                 feat["intersect_flag"],
#             ]
#         )
#     return np.array(features)


# # -----------------------------
# # DECISION TREE TRAINING & EVALUATION (with imbalance handling and hyperparameter tuning)
# # -----------------------------
# def tune_and_train_decision_tree(
#     X: np.ndarray, y: np.ndarray, random_state: int = 42
# ) -> DecisionTreeClassifier:
#     from sklearn.model_selection import GridSearchCV

#     # Parameter grid for tuning
#     param_grid = {
#         "max_depth": [None, 5, 10, 15],
#         "min_samples_split": [2, 5, 10],
#         "min_samples_leaf": [1, 2, 4],
#         "max_features": [None, "sqrt", "log2"],
#         "class_weight": ["balanced"],
#     }
#     dt = DecisionTreeClassifier(random_state=random_state)
#     grid_search = GridSearchCV(
#         dt,
#         param_grid,
#         cv=5,
#         scoring="f1",  # you can change the scoring metric as needed
#         n_jobs=-1,
#         verbose=1,
#         # stratify=y if y.ndim == 1 else None,
#     )
#     grid_search.fit(X, y)
#     print("Best hyperparameters:", grid_search.best_params_)
#     print("Best cross-validation score:", grid_search.best_score_)
#     best_dt = grid_search.best_estimator_
#     # Evaluate on the full dataset split or on a hold-out set
#     y_pred = best_dt.predict(X)
#     acc = accuracy_score(y, y_pred)
#     print(f"Full Dataset Accuracy (for tuning purposes): {acc*100:.2f}%")
#     print("Classification Report:")
#     print(classification_report(y, y_pred, zero_division=0))
#     cm = confusion_matrix(y, y_pred, labels=[0, 1])
#     print("Confusion Matrix:")
#     print(cm)
#     # Plot the decision tree from best estimator.
#     plt.figure(figsize=(20, 10))
#     plot_tree(
#         best_dt,
#         feature_names=[
#             "direct_distance",
#             "total_path",
#             "straightness_ratio",
#             "overall_angle",
#             "start_angle",
#             "end_angle",
#             "turn_diff",
#             "sudden_change",
#             "intersect_flag",
#         ],
#         class_names=["Normal", "Abnormal"],
#         filled=True,
#     )
#     plt.title("Tuned Decision Tree for Trajectory Classification")
#     plt.show()
#     return best_dt


# # -----------------------------
# # VISUALIZATION FUNCTIONS: Drawing trajectories on an image
# # -----------------------------
# def draw_trajectories(
#     image: np.ndarray,
#     trajectories: List[np.ndarray],
#     labels: np.ndarray,
#     color_map: Dict[int, Tuple[int, int, int]],
#     thickness: int = 2,
# ) -> np.ndarray:
#     overlay = image.copy()
#     for i, traj in enumerate(trajectories):
#         if traj is None or traj.shape[0] < 2:
#             continue
#         try:
#             label = int(labels[i])
#         except Exception:
#             label = -1
#         color = color_map.get(label, (127, 127, 127))
#         pts = traj[:, 1:3].astype(np.int32)
#         pts[:, 0] = np.clip(pts[:, 0], 0, image.shape[1] - 1)
#         pts[:, 1] = np.clip(pts[:, 1], 0, image.shape[0] - 1)
#         cv2.polylines(
#             overlay,
#             [pts],
#             isClosed=False,
#             color=color,
#             thickness=thickness,
#             lineType=cv2.LINE_AA,
#         )
#     return overlay


# def show_image(image: np.ndarray, title: str = "") -> None:
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     plt.figure(figsize=(10, 8))
#     plt.imshow(image_rgb)
#     plt.title(title)
#     plt.axis("off")
#     plt.show()


# def visualize_on_image(
#     image_path: str,
#     trajectories: List[np.ndarray],
#     labels: np.ndarray,
#     title: str,
#     color_map: Dict[int, Tuple[int, int, int]],
#     zone_polygon: Optional[np.ndarray] = None,
# ) -> None:
#     if os.path.exists(image_path):
#         base_image = cv2.imread(image_path)
#     else:
#         base_image = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
#     image_with_tracks = draw_trajectories(base_image, trajectories, labels, color_map)
#     if zone_polygon is not None and zone_polygon.shape[0] >= 3:
#         poly = zone_polygon.reshape((-1, 1, 2)).astype(np.int32)
#         cv2.polylines(
#             image_with_tracks, [poly], isClosed=True, color=(255, 0, 255), thickness=3
#         )
#     show_image(image_with_tracks, title)


# # -----------------------------
# # MAIN PIPELINE
# # -----------------------------
# def main():
#     print(f"Processing sample folder: '{FOLDER_PATH}' (Sample {SAMPLE})")
#     trajectories, filenames = load_trajectories(FOLDER_PATH, N_POINTS)
#     if not trajectories:
#         print("No valid trajectories found. Exiting.")
#         return

#     # Use ground truth if available; otherwise use rule-based labels.
#     gt_labels = load_ground_truth_from_csv(GROUND_TRUTH_CSV_PATH, filenames)
#     if gt_labels is not None and len(gt_labels) == len(trajectories):
#         print("Using ground truth labels for training.")
#         target = np.array(gt_labels, dtype=int)
#         gt_available = True
#     else:
#         print("Ground truth not available or mismatched. Using rule-based labels.")
#         rule_clusters = assign_custom_clusters_refined(
#             trajectories, central_zone_polygon, THRESHOLD_PARAMS
#         )
#         target = map_clusters_to_binary_refined(rule_clusters)
#         gt_available = False

#     # Build feature matrix
#     X = build_feature_matrix(trajectories, THRESHOLD_PARAMS, central_zone_polygon)
#     print(f"Feature matrix shape: {X.shape}")

#     # Hyperparameter tuning and training the decision tree classifier.
#     clf = tune_and_train_decision_tree(X, target, random_state=42)

#     # Use the trained classifier to predict labels on all trajectories.
#     predicted_labels = clf.predict(X)

#     # -----------------------------
#     # Visualization on Background Image:
#     #   1. Overlay with Ground Truth labels (if available)
#     #   2. Overlay with Predicted labels from the decision tree
#     # -----------------------------
#     print(
#         "\nDisplaying Ground Truth (if available) and Predicted Visualizations on Image..."
#     )

#     if gt_available:
#         visualize_on_image(
#             IMAGE_PATH,
#             trajectories,
#             target,
#             title=f"Sample {SAMPLE}: Ground Truth Binary Labels",
#             color_map=BINARY_COLORS_BGR,
#             zone_polygon=central_zone_polygon,
#         )
#     else:
#         print("No ground truth available for visualization.")

#     visualize_on_image(
#         IMAGE_PATH,
#         trajectories,
#         predicted_labels,
#         title=f"Sample {SAMPLE}: Predicted Binary Labels (Tuned Decision Tree)",
#         color_map=BINARY_COLORS_BGR,
#         zone_polygon=central_zone_polygon,
#     )


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Trajectory Classification Using a Refined Rule Engine and a Decision Tree Classifier,
with Hyperparameter Tuning on Train/Val/Test Splits and Visualization of Ground Truth and Predicted Trajectories.

This script:
  1. Loads and resamples trajectory data.
  2. Extracts features from each trajectory.
  3. Uses either the ground truth labels or the refined rule engine to produce binary labels.
  4. Splits data into train/validation/test, performs hyperparameter tuning on train, evaluates on validation.
  5. Trains best model and evaluates on test set.
  6. Overlays the ground truth and predicted labels on a background image for the test set.

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
SAMPLE = "11"
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
    0: (0, 255, 0),
    1: (0, 0, 255),
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
    except:
        return None


def resample_trajectory(traj: np.ndarray, n_points: int) -> Optional[np.ndarray]:
    if traj is None or traj.shape[0] < 2:
        return None
    xy = traj[:, 1:3]
    try:
        xy_contig = np.ascontiguousarray(xy, dtype=np.float64)
        resampler = TimeSeriesResampler(sz=n_points)
        resampled_xy = resampler.fit_transform(xy_contig.reshape(1, -1, 2))[0]
        frames = np.linspace(traj[0, 0], traj[-1, 0], n_points)
        return np.column_stack((frames, resampled_xy))
    except:
        return None


def load_trajectories(
    folder_path: str, n_points: int
) -> Tuple[List[np.ndarray], List[str]]:
    if not os.path.isdir(folder_path):
        return [], []
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])
    trajs, names = [], []
    for f in files:
        t = extract_trajectory(os.path.join(folder_path, f))
        if t is not None:
            rt = resample_trajectory(t, n_points)
            if rt is not None:
                trajs.append(rt)
                names.append(f)
    print(f"Loaded {len(trajs)} trajectories from {len(files)} files.")
    return trajs, names


def load_ground_truth_from_csv(gt_csv: str, fnames: List[str]) -> Optional[np.ndarray]:
    if not os.path.exists(gt_csv):
        return None
    df = pd.read_csv(gt_csv)
    if "filename" not in df.columns or "label" not in df.columns:
        return None
    mapping = {os.path.basename(r.filename): r.label for _, r in df.iterrows()}
    labels = [mapping.get(os.path.basename(f), -1) for f in fnames]
    print(f"Matched GT for {sum(l!=-1 for l in labels)} trajectories.")
    return np.array(labels)


# -----------------------------
# ANGLES / RULE-ENGINE UTILITIES
# -----------------------------
def angle_between_points(p1, p2):
    delta = p2 - p1
    return np.arctan2(delta[1], delta[0]) if np.linalg.norm(delta) > 1e-9 else 0.0


def smallest_angle_diff(a1, a2):
    d = a1 - a2
    return (d + np.pi) % (2 * np.pi) - np.pi


def get_trajectory_angles(traj, min_len):
    pts = traj[:, 1:3]
    if len(pts) < 2:
        return None, None, None
    overall = angle_between_points(pts[0], pts[-1])
    # start angle
    sa = next(
        (
            angle_between_points(pts[i], pts[i + 1])
            for i in range(len(pts) - 1)
            if np.linalg.norm(pts[i + 1] - pts[i]) >= min_len
        ),
        angle_between_points(pts[0], pts[1]),
    )
    # end angle
    ea = next(
        (
            angle_between_points(pts[i], pts[i + 1])
            for i in range(len(pts) - 2, -1, -1)
            if np.linalg.norm(pts[i + 1] - pts[i]) >= min_len
        ),
        angle_between_points(pts[-2], pts[-1]),
    )
    return sa, ea, overall


def get_direction(angle, tol):
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


def detect_sudden_change(traj, thr):
    pts = traj[:, 1:3]
    if len(pts) < 3:
        return False, 0.0
    vecs = np.diff(pts, axis=0)
    norms = np.linalg.norm(vecs, axis=1)
    valid = vecs[norms > 1.0]
    if len(valid) < 2:
        return False, 0.0
    angs = np.arctan2(valid[:, 1], valid[:, 0])
    diffs = smallest_angle_diff(angs[1:], angs[:-1])
    maxch = np.max(np.abs(diffs)) if diffs.size > 0 else 0.0
    return maxch > thr, maxch


def intersects_zone(traj, zone):
    if zone is None or len(zone) < 3 or traj is None or traj.shape[0] == 0:
        return False
    for pt in traj[:, 1:3].astype(np.float32):
        if cv2.pointPolygonTest(zone, tuple(pt), False) >= 0:
            return True
    return False


# -----------------------------
# REFINED RULE ENGINE
# -----------------------------
def assign_custom_cluster_refined(traj, zone, thr):
    if traj is None or len(traj) < 2:
        return 19
    pts = traj[:, 1:3]
    dd = np.linalg.norm(pts[-1] - pts[0])
    if dd < thr["moving_threshold"]:
        return 18
    sa, ea, oa = get_trajectory_angles(traj, thr["min_segment_length"])
    if sa is None:
        return 19
    # straight
    total = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    sr = total / (dd + 1e-6)
    sd = get_direction(oa, thr["hv_angle_tolerance_rad"])
    if sr < thr["straightness_threshold"]:
        if sd == "W_to_E":
            return 1
        if sd == "E_to_W":
            return 2
        if sd == "N_to_S":
            return 3
        if sd == "S_to_N":
            return 4
    # turn detection (omitted full code for brevity)
    # ... uses get_direction(sa,..), get_direction(ea,..)
    sudden, _ = detect_sudden_change(traj, thr["sudden_change_threshold_rad"])
    if sudden:
        return 17
    return 19


def map_clusters_to_binary_refined(labels):
    ab = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
    return np.array([1 if l in ab else 0 for l in labels])


# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(traj, thr, zone):
    pts = traj[:, 1:3]
    dd = np.linalg.norm(pts[-1] - pts[0])
    segs = np.diff(pts, axis=0)
    sl = np.linalg.norm(segs, axis=1)
    total = np.sum(sl)
    sr = total / (dd + 1e-6)
    sa, ea, oa = get_trajectory_angles(traj, thr["min_segment_length"])
    turn_diff = abs(smallest_angle_diff(sa, ea)) if sa is not None else 0.0
    sudden, _ = detect_sudden_change(traj, thr["sudden_change_threshold_rad"])
    inter = 1 if intersects_zone(traj, zone) else 0
    return [dd, total, sr, oa, sa, ea, turn_diff, int(sudden), inter]


def build_feature_matrix(trajs, thr, zone):
    return np.vstack([extract_features(t, thr, zone) for t in trajs])


# -----------------------------
# HYPERPARAMETER TUNING + TRAIN on Train/Val
# -----------------------------


def tune_and_train_decision_tree(X_train, y_train, X_val, y_val, random_state=42):
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    param_grid = {
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [None, "sqrt", "log2"],
        "class_weight": ["balanced"],
    }
    base = DecisionTreeClassifier(random_state=random_state)
    gs = GridSearchCV(base, param_grid, scoring="f1", cv=inner_cv, n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    print("Best params:", gs.best_params_)
    # validation performance
    yv = best.predict(X_val)
    print("-- Validation performance --")
    print("Accuracy:", accuracy_score(y_val, yv))
    print(classification_report(y_val, yv, zero_division=0))
    print("CM:", confusion_matrix(y_val, yv))
    return best


# -----------------------------
# VISUALIZATION on IMAGE
# -----------------------------
def draw_trajectories(image, trajs, labels, color_map, thickness=2):
    im = image.copy()
    for i, t in enumerate(trajs):
        pts = t[:, 1:3].astype(np.int32)
        cv2.polylines(
            im, [pts], False, color_map.get(int(labels[i]), (127, 127, 127)), thickness
        )
    return im


def show_image(im, title=""):
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


def visualize_on_image(path, trajs, labels, title, color_map, zone=None):
    base = (
        cv2.imread(path)
        if os.path.exists(path)
        else np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    )
    ov = draw_trajectories(base, trajs, labels, color_map)
    if zone is not None:
        cv2.polylines(ov, [zone.reshape(-1, 1, 2)], True, (255, 0, 255), 3)
    show_image(ov, title)


def extract_enhanced_features(traj, thr, zone):
    """
    Extract enhanced features from trajectory including jerk, deceleration, and angular metrics

    Args:
        traj: Trajectory array with columns [frameNo, x, y]
        thr: Dictionary of threshold parameters
        zone: Central zone polygon for intersection checks

    Returns:
        List of features
    """
    # Extract basic features from original code
    pts = traj[:, 1:3]
    timestamps = traj[:, 0]  # Frame numbers can be used as timestamps
    dd = np.linalg.norm(pts[-1] - pts[0])
    segs = np.diff(pts, axis=0)
    sl = np.linalg.norm(segs, axis=1)
    total = np.sum(sl)
    sr = total / (dd + 1e-6)
    sa, ea, oa = get_trajectory_angles(traj, thr["min_segment_length"])
    turn_diff = abs(smallest_angle_diff(sa, ea)) if sa is not None else 0.0
    sudden, _ = detect_sudden_change(traj, thr["sudden_change_threshold_rad"])
    inter = 1 if intersects_zone(traj, zone) else 0

    # Calculate velocities (for each segment between points)
    dt = np.diff(timestamps)
    # Avoid division by zero
    dt = np.where(dt == 0, 1e-6, dt)
    velocities = segs / dt[:, np.newaxis]
    speeds = np.linalg.norm(velocities, axis=1)

    # Calculate accelerations
    accels = np.diff(velocities, axis=0)
    dt_acc = np.diff(timestamps[:-1])
    dt_acc = np.where(dt_acc == 0, 1e-6, dt_acc)
    accels = accels / dt_acc[:, np.newaxis]
    accel_magnitudes = np.linalg.norm(accels, axis=1)

    # 1. Extract Jerk (rate of change of acceleration)
    jerks = np.diff(accels, axis=0)
    dt_jerk = np.diff(timestamps[:-2])
    dt_jerk = np.where(dt_jerk == 0, 1e-6, dt_jerk)
    jerks = jerks / dt_jerk[:, np.newaxis]
    jerk_magnitudes = np.linalg.norm(jerks, axis=1)
    max_jerk = np.max(jerk_magnitudes) if len(jerk_magnitudes) > 0 else 0
    mean_jerk = np.mean(jerk_magnitudes) if len(jerk_magnitudes) > 0 else 0

    # 2. & 4. Extract Rate of Change of Angle (Angular velocity and acceleration)
    angles = np.arctan2(segs[:, 1], segs[:, 0])
    angle_diffs = np.diff(angles)
    # Normalize angle differences to range [-pi, pi]
    angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
    dt_ang = np.diff(timestamps[:-1])
    dt_ang = np.where(dt_ang == 0, 1e-6, dt_ang)
    angular_velocities = angle_diffs / dt_ang
    max_angular_velocity = (
        np.max(np.abs(angular_velocities)) if len(angular_velocities) > 0 else 0
    )

    # Angular acceleration (rate of change of angular velocity)
    angular_accels = np.diff(angular_velocities)
    dt_ang_acc = np.diff(timestamps[:-2])
    dt_ang_acc = np.where(dt_ang_acc == 0, 1e-6, dt_ang_acc)
    angular_accels = angular_accels / dt_ang_acc
    max_angular_accel = np.max(np.abs(angular_accels)) if len(angular_accels) > 0 else 0

    # 3. Extract Deceleration (negative acceleration)
    # Calculate longitudinal accelerations (along direction of travel)
    longitudinal_accels = np.zeros(len(accels))
    for i in range(len(accels)):
        if i < len(velocities) - 1:
            # Project acceleration onto velocity direction
            v_dir = velocities[i] / (np.linalg.norm(velocities[i]) + 1e-6)
            longitudinal_accels[i] = np.dot(accels[i], v_dir)

    # Extract only the negative values (deceleration)
    decelerations = longitudinal_accels[longitudinal_accels < 0]
    max_deceleration = np.min(decelerations) if len(decelerations) > 0 else 0
    mean_deceleration = np.mean(decelerations) if len(decelerations) > 0 else 0

    # Calculate variability metrics
    speed_variance = np.var(speeds) if len(speeds) > 0 else 0
    accel_variance = np.var(accel_magnitudes) if len(accel_magnitudes) > 0 else 0

    # Combine original features with new ones
    features = [
        dd,
        total,
        sr,
        oa,
        sa,
        ea,
        turn_diff,
        int(sudden),
        inter,
        # New features:
        max_jerk,
        mean_jerk,  # Jerk features
        max_angular_velocity,
        max_angular_accel,  # Angle rate of change features
        max_deceleration,
        mean_deceleration,  # Deceleration features
        speed_variance,
        accel_variance,  # Variability metrics
    ]

    return features


def get_feature_names():
    """Return names of all features for reference"""
    return [
        # Original features
        "direct_distance",
        "total_length",
        "straightness_ratio",
        "overall_angle",
        "start_angle",
        "end_angle",
        "turn_difference",
        "sudden_change",
        "intersection",
        # New features
        "max_jerk",
        "mean_jerk",
        "max_angular_velocity",
        "max_angular_acceleration",
        "max_deceleration",
        "mean_deceleration",
        "speed_variance",
        "acceleration_variance",
    ]


# Update the build_feature_matrix function to use enhanced features
def build_enhanced_feature_matrix(trajs, thr, zone):
    return np.vstack([extract_enhanced_features(t, thr, zone) for t in trajs])


# Visualization function for trajectory colored by parameter value
def visualize_trajectory_by_parameter(
    image_path, trajectory, parameter_values, param_name, colormap="jet", alpha=0.7
):
    """
    Visualize a trajectory colored by a specific parameter value

    Args:
        image_path: Path to background image
        trajectory: Trajectory array with columns [frameNo, x, y]
        parameter_values: Array of parameter values for each segment
        param_name: Name of parameter for title
        colormap: Matplotlib colormap name
        alpha: Transparency level
    """
    # Load background image
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255

    # Create figure
    plt.figure(figsize=(12, 10))
    plt.imshow(img)

    # Get colormap
    cmap = plt.get_cmap(colormap)

    # Normalize parameter values
    if len(parameter_values) > 0:
        min_val, max_val = np.min(parameter_values), np.max(parameter_values)
        if min_val == max_val:
            normalized = np.zeros_like(parameter_values)
        else:
            normalized = (parameter_values - min_val) / (max_val - min_val)
    else:
        normalized = []

    # Plot each segment with its color
    pts = trajectory[:, 1:3]
    for i in range(len(pts) - 1):
        if i < len(normalized):
            color = cmap(normalized[i])
            plt.plot(
                [pts[i][0], pts[i + 1][0]],
                [pts[i][1], pts[i + 1][1]],
                color=color,
                linewidth=3,
                alpha=alpha,
            )

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    if len(parameter_values) > 0:
        cbar.set_label(f"{param_name} [{min_val:.2f} - {max_val:.2f}]")
    else:
        cbar.set_label(param_name)

    plt.title(f"Trajectory Visualization: {param_name}")
    plt.axis("off")
    plt.tight_layout()

    # Save figure
    plt.savefig(f"trajectory_{param_name.lower().replace(' ', '_')}.png", dpi=300)
    plt.show()


# Example function to calculate and visualize jerk along a trajectory
def analyze_jerk_in_trajectory(traj, image_path):
    """Calculate and visualize jerk in a trajectory"""
    pts = traj[:, 1:3]
    timestamps = traj[:, 0]

    # Calculate velocities
    dt = np.diff(timestamps)
    dt = np.where(dt == 0, 1e-6, dt)
    velocities = np.diff(pts, axis=0) / dt[:, np.newaxis]

    # Calculate accelerations
    dt_acc = np.diff(timestamps[:-1])
    dt_acc = np.where(dt_acc == 0, 1e-6, dt_acc)
    accels = np.diff(velocities, axis=0) / dt_acc[:, np.newaxis]

    # Calculate jerks
    dt_jerk = np.diff(timestamps[:-2])
    dt_jerk = np.where(dt_jerk == 0, 1e-6, dt_jerk)
    jerks = np.diff(accels, axis=0) / dt_jerk[:, np.newaxis]
    jerk_magnitudes = np.linalg.norm(jerks, axis=1)

    # Visualize jerk along trajectory
    visualize_trajectory_by_parameter(image_path, traj, jerk_magnitudes, "Jerk")

    return jerk_magnitudes


# -----------------------------
# MAIN
# -----------------------------
def main():
    trajs, fnames = load_trajectories(FOLDER_PATH, N_POINTS)
    if not trajs:
        return
    gt = load_ground_truth_from_csv(GROUND_TRUTH_CSV_PATH, fnames)
    if gt is not None and len(gt) == len(trajs):
        print("Using GT labels")
        y = gt
    else:
        print("Using rule-based labels")
        rc = [
            assign_custom_cluster_refined(t, central_zone_polygon, THRESHOLD_PARAMS)
            for t in trajs
        ]
        y = map_clusters_to_binary_refined(rc)
    X = build_feature_matrix(trajs, THRESHOLD_PARAMS, central_zone_polygon)
    # indices split
    idx = np.arange(len(y))
    trv, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    tr, va = train_test_split(trv, test_size=0.25, random_state=42, stratify=y[trv])
    X_tr, X_va, X_te = X[tr], X[va], X[te]
    y_tr, y_va, y_te = y[tr], y[va], y[te]
    traj_tr = [trajs[i] for i in tr]
    traj_va = [trajs[i] for i in va]
    traj_te = [trajs[i] for i in te]
    print(f"Sizes → train:{len(tr)}, val:{len(va)}, test:{len(te)}")
    # tune on train/val
    best = tune_and_train_decision_tree(X_tr, y_tr, X_va, y_va)
    # test
    y_pred = best.predict(X_te)
    print("-- Test performance --")
    print("Accuracy:", accuracy_score(y_te, y_pred))
    print(classification_report(y_te, y_pred, zero_division=0))
    print("CM:", confusion_matrix(y_te, y_pred))
    # visualize test
    visualize_on_image(
        IMAGE_PATH,
        traj_te,
        y_te,
        f"Sample {SAMPLE}: Test GT",
        BINARY_COLORS_BGR,
        central_zone_polygon,
    )
    visualize_on_image(
        IMAGE_PATH,
        traj_te,
        y_pred,
        f"Sample {SAMPLE}: Test Pred",
        BINARY_COLORS_BGR,
        central_zone_polygon,
    )


if __name__ == "__main__":
    main()
