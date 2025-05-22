#!/usr/bin/env python3
"""
Enhanced Trajectory Analysis: Angle and Rate of Change Analysis with 3D Visualization
and Performance Metrics Analysis.

This script extends the original trajectory classification to:
1. Extract and analyze angles and angular velocities for each trajectory
2. Create 3D visualizations of trajectory features
3. Perform detailed accuracy and precision analysis
4. Visualize feature importance and model performance
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tslearn.preprocessing import TimeSeriesResampler

# -----------------------------
# PARAMETERS & PATHS (Same as original)
# -----------------------------
SAMPLE = "11"
FOLDER_PATH = SAMPLE
IMAGE_PATH = f"sample{SAMPLE}.jpg"
GROUND_TRUTH_CSV_PATH = "trajectory_images/combined_labels.csv"
N_POINTS = 50

OUTPUT_DIR = f"trajectory_angle_analysis_sample{SAMPLE}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thresholds and parameters from original code
THRESHOLD_PARAMS: Dict[str, Any] = {
    "moving_threshold": 10.0,
    "straightness_threshold": 1.30,
    "hv_angle_tolerance_deg": 30.0,
    "turn_angle_tolerance_deg": 45.0,
    "sudden_change_threshold_deg": 65.0,
    "min_segment_length": 5.0,
}
THRESHOLD_PARAMS["sudden_change_threshold_rad"] = np.deg2rad(THRESHOLD_PARAMS["sudden_change_threshold_deg"])
THRESHOLD_PARAMS["hv_angle_tolerance_rad"] = np.deg2rad(THRESHOLD_PARAMS["hv_angle_tolerance_deg"])
THRESHOLD_PARAMS["turn_angle_tolerance_rad"] = np.deg2rad(THRESHOLD_PARAMS["turn_angle_tolerance_deg"])

# Central zone polygon
central_zone_polygon = np.array([[1650, 842], [1650, 1331], [2271, 1331], [2271, 842]], dtype=np.int32)

# Color maps
BINARY_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {0: (0, 255, 0), 1: (0, 0, 255)}

# -----------------------------
# UTILITY FUNCTIONS (From original code)
# -----------------------------
def extract_trajectory(csv_path: str) -> Optional[np.ndarray]:
    """
    Extracts the trajectory center points from a CSV file.
    
    Reads frame number and bounding box columns from the CSV, computes the center coordinates for each frame, and returns an array of frame numbers and center positions. Returns None if the file is invalid or contains fewer than two frames.
    
    Args:
        csv_path: Path to the CSV file containing trajectory data.
    
    Returns:
        A NumPy array of shape (n_frames, 3) with columns [frameNo, center_x, center_y], or None if extraction fails.
    """
    try:
        df = pd.read_csv(csv_path, usecols=["frameNo", "left", "top", "w", "h"])
        df["center_x"] = df["left"] + df["w"] / 2
        df["center_y"] = df["top"] + df["h"] / 2
        if len(df) < 2:
            return None
        return df[["frameNo", "center_x", "center_y"]].values
-    except:
-        return None
+    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
+        warnings.warn(f"Failed to extract trajectory from {csv_path}: {exc}")
+        return None

def resample_trajectory(traj: np.ndarray, n_points: int) -> Optional[np.ndarray]:
    """
    Resamples a trajectory to a fixed number of points using interpolation.
    
    Args:
        traj: Array of shape (N, 3) containing frame number and (x, y) coordinates.
        n_points: Desired number of points in the resampled trajectory.
    
    Returns:
        A NumPy array of shape (n_points, 3) with evenly spaced frame numbers and resampled (x, y) coordinates, or None if input is invalid or resampling fails.
    """
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

def load_trajectories(folder_path: str, n_points: int) -> Tuple[List[np.ndarray], List[str]]:
    """
    Loads and resamples all trajectory CSV files from a folder.
    
    Iterates through CSV files in the specified directory, extracts trajectory data, resamples each trajectory to a fixed number of points, and returns the list of valid resampled trajectories along with their corresponding filenames.
    
    Args:
        folder_path: Path to the folder containing trajectory CSV files.
        n_points: Number of points to resample each trajectory to.
    
    Returns:
        A tuple containing a list of resampled trajectories (as numpy arrays) and a list of their corresponding filenames.
    """
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
    """
    Loads ground truth labels from a CSV file and matches them to a list of filenames.
    
    Args:
        gt_csv: Path to the ground truth CSV file, which must contain 'filename' and 'label' columns.
        fnames: List of filenames to match with ground truth labels.
    
    Returns:
        An array of labels corresponding to the input filenames, or None if the CSV is missing or invalid.
        If a filename is not found in the CSV, its label is set to -1.
    """
    if not os.path.exists(gt_csv):
        return None
    df = pd.read_csv(gt_csv)
    if "filename" not in df.columns or "label" not in df.columns:
        return None
    mapping = {os.path.basename(r.filename): r.label for _, r in df.iterrows()}
    labels = [mapping.get(os.path.basename(f), -1) for f in fnames]
    print(f"Matched GT for {sum(l!=-1 for l in labels)} trajectories.")
    return np.array(labels)

def smallest_angle_diff(a1, a2):
    """
    Calculates the smallest signed difference between two angles in radians.
    
    The result is normalized to the range [-π, π], representing the shortest angular distance from a2 to a1.
    """
    d = a1 - a2
    return (d + np.pi) % (2 * np.pi) - np.pi

def intersects_zone(traj, zone):
    """
    Checks if any point in a trajectory lies within or on the boundary of a specified polygonal zone.
    
    Args:
        traj: Trajectory as a NumPy array with at least three columns (frame, x, y).
        zone: Polygonal zone as a NumPy array of shape (N, 2), where N ≥ 3.
    
    Returns:
        True if any trajectory point is inside or on the edge of the zone; otherwise, False.
    """
    if zone is None or len(zone) < 3 or traj is None or traj.shape[0] == 0:
        return False
    for pt in traj[:, 1:3].astype(np.float32):
        if cv2.pointPolygonTest(zone, tuple(pt), False) >= 0:
            return True
    return False

# -----------------------------
# ENHANCED ANGLE ANALYSIS FUNCTIONS
# -----------------------------
class TrajectoryAngleAnalyzer:
    """Class for comprehensive angle analysis of trajectories"""
    
    def __init__(self):
        """
        Initializes the class with an empty list to store trajectory data.
        """
        self.trajectory_data = []
        
    def analyze_trajectory_angles(self, traj: np.ndarray) -> Dict[str, Any]:
        """
        Performs comprehensive angle and curvature analysis on a single trajectory.
        
        Analyzes the input trajectory to compute detailed metrics including segment direction angles, angular differences, angular velocities, angular accelerations, curvature, turn statistics, smoothness, and complexity. Returns a dictionary containing all computed metrics for further feature extraction or visualization.
        
        Args:
            traj: Array of shape (N, 3) where each row contains [frame, x, y] for a trajectory.
        
        Returns:
            Dictionary with detailed angle, angular velocity, angular acceleration, curvature, turn, smoothness, and complexity metrics for the trajectory. If the trajectory is too short or invalid, returns a dictionary of zeroed/default metrics.
        """
        pts = traj[:, 1:3]
        timestamps = traj[:, 0]
        
        if len(pts) < 3:
            return self._empty_analysis()
        
        # Calculate segment vectors and angles
        segments = np.diff(pts, axis=0)
        segment_lengths = np.linalg.norm(segments, axis=1)
        
        # Filter out very small segments to avoid noise
        valid_segments = segment_lengths > 1.0
        if np.sum(valid_segments) < 2:
            return self._empty_analysis()
        
        valid_segs = segments[valid_segments]
        valid_lengths = segment_lengths[valid_segments]
        
        # Calculate angles for each segment (direction angles)
        angles = np.arctan2(valid_segs[:, 1], valid_segs[:, 0])
        angles_deg = np.degrees(angles)
        
        # Calculate angular differences (rate of change of angle)
        if len(angles) < 2:
            return self._empty_analysis()
            
        angle_diffs = np.diff(angles)
        # Normalize to [-pi, pi]
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
        angle_diffs_deg = np.degrees(angle_diffs)
        
        # Calculate time intervals for angular velocity
        valid_timestamps = timestamps[:-1][valid_segments]
        if len(valid_timestamps) < 2:
            dt_angles = np.ones(len(angle_diffs))
        else:
            dt_angles = np.diff(valid_timestamps)
            dt_angles = np.where(dt_angles == 0, 1e-6, dt_angles)
        
        # Angular velocities (rate of change of angle)
        angular_velocities = angle_diffs / dt_angles
        angular_velocities_deg = np.degrees(angular_velocities)
        
        # Calculate angular accelerations
        if len(angular_velocities) < 2:
            angular_accelerations = np.array([0])
            angular_accelerations_deg = np.array([0])
        else:
            dt_ang_acc = dt_angles[:-1] if len(dt_angles) > 1 else np.array([1.0])
            angular_accelerations = np.diff(angular_velocities) / dt_ang_acc
            angular_accelerations_deg = np.degrees(angular_accelerations)
        
        # Curvature analysis
        curvatures = self._calculate_curvature(pts)
        
        # Compile comprehensive analysis
        analysis = {
            # Basic angle metrics
            'angles': angles,
            'angles_deg': angles_deg,
            'mean_angle': np.mean(angles),
            'std_angle': np.std(angles),
            'angle_range': np.ptp(angles),
            
            # Angular velocity metrics
            'angular_velocities': angular_velocities,
            'angular_velocities_deg': angular_velocities_deg,
            'mean_angular_velocity': np.mean(np.abs(angular_velocities)),
            'max_angular_velocity': np.max(np.abs(angular_velocities)),
            'std_angular_velocity': np.std(angular_velocities),
            
            # Angular acceleration metrics
            'angular_accelerations': angular_accelerations,
            'angular_accelerations_deg': angular_accelerations_deg,
            'mean_angular_acceleration': np.mean(np.abs(angular_accelerations)),
            'max_angular_acceleration': np.max(np.abs(angular_accelerations)),
            'std_angular_acceleration': np.std(angular_accelerations),
            
            # Curvature metrics
            'curvatures': curvatures,
            'mean_curvature': np.mean(np.abs(curvatures)),
            'max_curvature': np.max(np.abs(curvatures)),
            'curvature_variance': np.var(curvatures),
            
            # Turn detection
            'sharp_turns': np.sum(np.abs(angle_diffs_deg) > 45),
            'moderate_turns': np.sum((np.abs(angle_diffs_deg) > 15) & (np.abs(angle_diffs_deg) <= 45)),
            'total_turn_angle': np.sum(np.abs(angle_diffs_deg)),
            
            # Smoothness metrics
            'angle_smoothness': self._calculate_smoothness(angles),
            'velocity_smoothness': self._calculate_smoothness(angular_velocities),
            
            # Trajectory characterization
            'is_predominantly_straight': np.sum(np.abs(angle_diffs_deg) < 5) / len(angle_diffs_deg) > 0.8,
            'has_sudden_changes': np.any(np.abs(angle_diffs_deg) > 90),
            'complexity_score': np.std(angle_diffs_deg) / (np.mean(np.abs(angle_diffs_deg)) + 1e-6)
        }
        
        return analysis
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """
        Returns a dictionary of zeroed and empty angle analysis metrics for invalid or missing trajectories.
        
        This provides a standardized output structure with default values when trajectory analysis cannot be performed.
        """
        return {
            'angles': np.array([]),
            'angles_deg': np.array([]),
            'mean_angle': 0,
            'std_angle': 0,
            'angle_range': 0,
            'angular_velocities': np.array([]),
            'angular_velocities_deg': np.array([]),
            'mean_angular_velocity': 0,
            'max_angular_velocity': 0,
            'std_angular_velocity': 0,
            'angular_accelerations': np.array([]),
            'angular_accelerations_deg': np.array([]),
            'mean_angular_acceleration': 0,
            'max_angular_acceleration': 0,
            'std_angular_acceleration': 0,
            'curvatures': np.array([]),
            'mean_curvature': 0,
            'max_curvature': 0,
            'curvature_variance': 0,
            'sharp_turns': 0,
            'moderate_turns': 0,
            'total_turn_angle': 0,
            'angle_smoothness': 0,
            'velocity_smoothness': 0,
            'is_predominantly_straight': True,
            'has_sudden_changes': False,
            'complexity_score': 0
        }
    
    def _calculate_curvature(self, pts: np.ndarray) -> np.ndarray:
        """
        Calculates the curvature at each point of a trajectory using three consecutive points.
        
        Args:
            pts: An array of 2D points representing the trajectory.
        
        Returns:
            An array of curvature values for each internal point in the trajectory.
        """
        if len(pts) < 3:
            return np.array([])
        
        curvatures = []
        for i in range(1, len(pts) - 1):
            p1, p2, p3 = pts[i-1], pts[i], pts[i+1]
            
            # Vectors from p2 to p1 and p2 to p3
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate cross product for curvature
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            
            # Calculate lengths
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            
            if len1 * len2 > 1e-9:
                curvature = cross / (len1 * len2)
            else:
                curvature = 0
            
            curvatures.append(curvature)
        
        return np.array(curvatures)
    
    def _calculate_smoothness(self, values: np.ndarray) -> float:
        """
        Calculates a smoothness score for a sequence as the inverse of its total variation.
        
        Args:
            values: Sequence of numeric values representing a trajectory feature.
        
        Returns:
            A float between 0 and 1, where higher values indicate smoother sequences.
        """
        if len(values) < 2:
            return 1.0
        variation = np.sum(np.abs(np.diff(values)))
        return 1.0 / (1.0 + variation)

# -----------------------------
# 3D VISUALIZATION FUNCTIONS
# -----------------------------
class Trajectory3DVisualizer:
    """Class for creating 3D visualizations of trajectory features"""
    
    def __init__(self, output_dir: str):
        """
        Initializes the visualizer with the specified output directory.
        
        Args:
            output_dir: Path to the directory where visualizations will be saved.
        """
        self.output_dir = output_dir
        
    def create_3d_feature_space(self, features: np.ndarray, labels: np.ndarray, 
                               feature_names: List[str], title: str = "3D Feature Space"):
        """
                               Creates a 3D scatter plot of three selected features, coloring points by label.
                               
                               Args:
                                   features: Feature matrix where each row corresponds to a trajectory and columns to features.
                                   labels: Array of class labels for each trajectory.
                                   feature_names: List of feature names corresponding to columns in the feature matrix.
                                   title: Title for the plot (default: "3D Feature Space").
                               
                               The plot is saved to the output directory as "3d_feature_space.png".
                               """
        # Select top 3 most important features for 3D visualization
        # For now, let's use angular velocity, curvature, and turn angle features
        if features.shape[1] >= 3:
            x_idx, y_idx, z_idx = 0, 1, 2  # You can modify these indices
            x_data = features[:, x_idx]
            y_data = features[:, y_idx] 
            z_data = features[:, z_idx]
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Color by labels
            colors = ['green' if l == 0 else 'red' for l in labels]
            scatter = ax.scatter(x_data, y_data, z_data, c=colors, alpha=0.6, s=50)
            
            ax.set_xlabel(feature_names[x_idx])
            ax.set_ylabel(feature_names[y_idx])
            ax.set_zlabel(feature_names[z_idx])
            ax.set_title(title)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='green', label='Normal'),
                             Patch(facecolor='red', label='Abnormal')]
            ax.legend(handles=legend_elements)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"3d_feature_space.png"), dpi=300)
            plt.show()
    
    def create_angle_evolution_3d(self, trajectories: List[np.ndarray], 
                                 angle_analyses: List[Dict], labels: np.ndarray):
        """
                                 Creates a 3D plot visualizing the evolution of angles and angular velocities over time for multiple trajectories.
                                 
                                 Each trajectory is plotted as a 3D line where the x-axis represents time steps, the y-axis shows the angle in degrees, and the z-axis displays angular velocity in degrees per step. Trajectories are colored by label (e.g., normal or abnormal) for comparison, and the plot is saved to the output directory.
                                 """
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, (traj, analysis, label) in enumerate(zip(trajectories, angle_analyses, labels)):
            if len(analysis['angles']) > 0:
                time_points = np.arange(len(analysis['angles']))
                angles_deg = analysis['angles_deg']
                angular_vel_deg = analysis['angular_velocities_deg']
                
                if len(angular_vel_deg) == len(angles_deg) - 1:
                    # Pad angular velocity to match angles length
                    angular_vel_deg = np.append(angular_vel_deg, angular_vel_deg[-1])
                elif len(angular_vel_deg) > len(angles_deg):
                    angular_vel_deg = angular_vel_deg[:len(angles_deg)]
                
                color = 'red' if label == 1 else 'green'
                alpha = 0.3 if i > 10 else 0.7  # Reduce opacity for too many trajectories
                
                ax.plot(time_points, angles_deg, angular_vel_deg, 
                       color=color, alpha=alpha, linewidth=1)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Angle (degrees)')
        ax.set_zlabel('Angular Velocity (deg/step)')
        ax.set_title('3D Evolution of Angles and Angular Velocities')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='green', label='Normal'),
                          Line2D([0], [0], color='red', label='Abnormal')]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "3d_angle_evolution.png"), dpi=300)
        plt.show()
    
    def create_trajectory_surface_plot(self, trajectories: List[np.ndarray], 
                                     angle_analyses: List[Dict], labels: np.ndarray):
        """
                                     Creates a 3D surface plot comparing normal and abnormal trajectory groups.
                                     
                                     Generates interactive surface plots for trajectories labeled as normal and abnormal, visualizing their spatial characteristics side by side. The resulting plot is saved as an HTML file and displayed.
                                     """
        # Prepare data for surface plot
        normal_trajs = [traj for traj, label in zip(trajectories, labels) if label == 0]
        abnormal_trajs = [traj for traj, label in zip(trajectories, labels) if label == 1]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Normal Trajectories', 'Abnormal Trajectories'],
            specs=[[{'type': 'surface'}, {'type': 'surface'}]]
        )
        
        # Process normal trajectories
        if normal_trajs:
            self._add_trajectory_surface(fig, normal_trajs, 'Normal', row=1, col=1)
        
        # Process abnormal trajectories  
        if abnormal_trajs:
            self._add_trajectory_surface(fig, abnormal_trajs, 'Abnormal', row=1, col=2)
        
        fig.update_layout(title_text="Trajectory Surface Analysis", height=600)
        fig.write_html(os.path.join(self.output_dir, "trajectory_surface.html"))
        fig.show()
    
    def _add_trajectory_surface(self, fig, trajectories, title, row, col):
        """
        Adds a 3D surface plot of interpolated trajectories to a subplot.
        
        Interpolates a subset of trajectories to a uniform length, constructs a surface representing their spatial evolution, and adds it to the specified subplot in the provided figure.
        """
        # Create a grid representing average trajectory characteristics
        max_len = max(len(traj) for traj in trajectories)
        
        # Interpolate all trajectories to same length
        interp_trajs = []
        for traj in trajectories[:min(10, len(trajectories))]:  # Limit for performance
            pts = traj[:, 1:3]
            if len(pts) < max_len:
                # Simple interpolation
                t_orig = np.linspace(0, 1, len(pts))
                t_new = np.linspace(0, 1, max_len)
                x_interp = np.interp(t_new, t_orig, pts[:, 0])
                y_interp = np.interp(t_new, t_orig, pts[:, 1])
                interp_trajs.append(np.column_stack([x_interp, y_interp]))
            else:
                interp_trajs.append(pts[:max_len])
        
        if interp_trajs:
            # Create surface from trajectory data
            traj_array = np.array(interp_trajs)
            x_surface = traj_array[:, :, 0]
            y_surface = traj_array[:, :, 1]
            z_surface = np.arange(len(interp_trajs))[:, np.newaxis] * np.ones((1, max_len))
            
            fig.add_trace(
                go.Surface(x=x_surface, y=y_surface, z=z_surface, 
                          name=title, showscale=False),
                row=row, col=col
            )

# -----------------------------
# PERFORMANCE ANALYSIS FUNCTIONS
# -----------------------------
class PerformanceAnalyzer:
    """Class for comprehensive performance analysis"""
    
    def __init__(self, output_dir: str):
        """
        Initializes the visualizer with the specified output directory for saving plots.
        
        Args:
            output_dir: Path to the directory where generated visualizations will be saved.
        """
        self.output_dir = output_dir
        
    def analyze_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                y_prob: np.ndarray = None, class_names: List[str] = None):
        """
                                Performs comprehensive evaluation of classification model performance.
                                
                                Calculates and prints overall accuracy, weighted precision, recall, and F1-score, displays a detailed classification report, visualizes the confusion matrix, and provides per-class metric analysis. If prediction probabilities are provided, plots ROC and Precision-Recall curves. Returns a dictionary of key performance metrics.
                                """
        if class_names is None:
            class_names = ['Normal', 'Abnormal']
            
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print("="*60)
        print("COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("="*60)
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        print()
        
        # Detailed classification report
        print("Detailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self._plot_confusion_matrix(cm, class_names)
        
        # Per-class analysis
        self._analyze_per_class_performance(y_true, y_pred, class_names)
        
        # ROC and Precision-Recall curves if probabilities available
        if y_prob is not None:
            self._plot_roc_and_pr_curves(y_true, y_prob)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str]):
        """
        Displays and saves a confusion matrix heatmap for classification results.
        
        Args:
            cm: Confusion matrix as a 2D NumPy array.
            class_names: List of class label names for axis annotation.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300)
        plt.show()
    
    def _analyze_per_class_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     class_names: List[str]):
        """
                                     Calculates and prints precision, recall, specificity, F1-score, and support for each class.
                                     
                                     Args:
                                         y_true: Array of true class labels.
                                         y_pred: Array of predicted class labels.
                                         class_names: List of class names corresponding to label indices.
                                     """
        print("\nPer-Class Analysis:")
        print("-" * 40)
        
        for i, class_name in enumerate(class_names):
            # True positives, false positives, false negatives
            tp = np.sum((y_true == i) & (y_pred == i))
            fp = np.sum((y_true != i) & (y_pred == i))
            fn = np.sum((y_true == i) & (y_pred != i))
            tn = np.sum((y_true != i) & (y_pred != i))
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"{class_name}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall (Sensitivity): {recall:.4f}")
            print(f"  Specificity: {specificity:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Support: {np.sum(y_true == i)}")
            print()
    
    def _plot_roc_and_pr_curves(self, y_true: np.ndarray, y_prob: np.ndarray):
        """
        Plots the ROC and Precision-Recall curves for binary classification results.
        
        Args:
            y_true: Ground truth binary labels.
            y_prob: Predicted probabilities for the positive class.
        
        The function saves the resulting plots as 'roc_pr_curves.png' in the output directory and displays them.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax1.legend(loc="lower right")
        ax1.grid(True)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        ax2.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR Curve (AUC = {pr_auc:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])  
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'roc_pr_curves.png'), dpi=300)
        plt.show()

# -----------------------------
# ENHANCED FEATURE EXTRACTION
# -----------------------------
def extract_comprehensive_features(traj: np.ndarray, thr: Dict, zone: np.ndarray, 
                                 analyzer: TrajectoryAngleAnalyzer) -> np.ndarray:
    """
                                 Extracts a comprehensive feature vector from a trajectory, combining geometric, intersection, and detailed angle-based metrics.
                                 
                                 The feature vector includes direct distance, total length, straightness ratio, intersection with a specified zone, and a suite of angle, angular velocity, angular acceleration, curvature, turn, smoothness, and complexity metrics derived from detailed angle analysis.
                                 """
    # Get angle analysis
    angle_analysis = analyzer.analyze_trajectory_angles(traj)
    
    # Basic trajectory features
    pts = traj[:, 1:3]
    dd = np.linalg.norm(pts[-1] - pts[0])
    segs = np.diff(pts, axis=0)
    sl = np.linalg.norm(segs, axis=1)
    total = np.sum(sl)
    sr = total / (dd + 1e-6)
    inter = 1 if intersects_zone(traj, zone) else 0
    
    # Compile all features
    features = [
        # Basic features
        dd,                                    # direct_distance
        total,                                 # total_length  
        sr,                                    # straightness_ratio
        inter,                                 # intersection
        
        # Angle features
        angle_analysis['mean_angle'],          # mean_angle
        angle_analysis['std_angle'],           # std_angle
        angle_analysis['angle_range'],         # angle_range
        
        # Angular velocity features
        angle_analysis['mean_angular_velocity'],      # mean_angular_velocity
        angle_analysis['max_angular_velocity'],       # max_angular_velocity
        angle_analysis['std_angular_velocity'],       # std_angular_velocity
        
        # Angular acceleration features
        angle_analysis['mean_angular_acceleration'],  # mean_angular_acceleration  
        angle_analysis['max_angular_acceleration'],   # max_angular_acceleration
        angle_analysis['std_angular_acceleration'],   # std_angular_acceleration
        
        # Curvature features
        angle_analysis['mean_curvature'],             # mean_curvature
        angle_analysis['max_curvature'],              # max_curvature
        angle_analysis['curvature_variance'],         # curvature_variance
        
        # Turn features
        angle_analysis['sharp_turns'],                # sharp_turns
        angle_analysis['moderate_turns'],             # moderate_turns
        angle_analysis['total_turn_angle'],           # total_turn_angle
        
        # Smoothness features
        angle_analysis['angle_smoothness'],           # angle_smoothness
        angle_analysis['velocity_smoothness'],        # velocity_smoothness
        
        # Complexity features
        float(angle_analysis['is_predominantly_straight']),  # is_predominantly_straight
        float(angle_analysis['has_sudden_changes']),         # has_sudden_changes
        angle_analysis['complexity_score'],                  # complexity_score
    ]
    
    return np.array(features)

def get_comprehensive_feature_names() -> List[str]:
    """
    Returns the list of feature names used in comprehensive trajectory analysis.
    
    The returned list includes basic geometric, angle-based, angular velocity, angular acceleration, curvature, turn, smoothness, and complexity features extracted from trajectories.
    """
    return [
        # Basic features
        'direct_distance',
        'total_length', 
        'straightness_ratio',
        'intersection',
        
        # Angle features
        'mean_angle',
        'std_angle',
        'angle_range',
        
        # Angular velocity features
        'mean_angular_velocity',
        'max_angular_velocity',
        'std_angular_velocity',
        
        # Angular acceleration features
        'mean_angular_acceleration',
        'max_angular_acceleration',
        'std_angular_acceleration',
        
        # Curvature features
        'mean_curvature',
        'max_curvature',
        'curvature_variance',
        
        # Turn features
        'sharp_turns',
        'moderate_turns',
        'total_turn_angle',
        
        # Smoothness features
        'angle_smoothness',
        'velocity_smoothness',
        
        # Complexity features
        'is_predominantly_straight',
        'has_sudden_changes',
        'complexity_score',
    ]

def build_comprehensive_feature_matrix(trajs: List[np.ndarray], thr: Dict, 
                                     zone: np.ndarray, analyzer: TrajectoryAngleAnalyzer) -> np.ndarray:
    """
                                     Builds a feature matrix by extracting comprehensive geometric and angle-based features from a list of trajectories.
                                     
                                     Each trajectory is processed to generate a feature vector that includes basic trajectory metrics and detailed angle analysis using the provided analyzer.
                                     
                                     Args:
                                         trajs: List of trajectory arrays to process.
                                         thr: Dictionary of threshold parameters for feature extraction.
                                         zone: Polygonal zone used for intersection checks.
                                         analyzer: Instance of TrajectoryAngleAnalyzer for angle-based feature extraction.
                                     
                                     Returns:
                                         A 2D NumPy array where each row corresponds to the feature vector of a trajectory.
                                     """
    return np.vstack([extract_comprehensive_features(t, thr, zone, analyzer) for t in trajs])

# -----------------------------
# RULE ENGINE (Simplified from original)
# -----------------------------
def assign_simple_binary_label(traj: np.ndarray, zone: np.ndarray, thr: Dict) -> int:
    """
    Assigns a binary label to a trajectory as normal (0) or abnormal (1) based on length, straightness, and sudden direction changes.
    
    A trajectory is labeled abnormal if it is too short, highly curved, or contains sharp turns exceeding 90 degrees.
    """
    if traj is None or len(traj) < 2:
        return 1  # Abnormal
    
    pts = traj[:, 1:3]
    dd = np.linalg.norm(pts[-1] - pts[0])
    
    # Very short trajectories are abnormal
    if dd < thr["moving_threshold"]:
        return 1
    
    # Calculate straightness
    segs = np.diff(pts, axis=0)
    sl = np.linalg.norm(segs, axis=1)
    total = np.sum(sl)
    sr = total / (dd + 1e-6)
    
    # Very curved trajectories are potentially abnormal
    if sr > 2.0:  # Much more curved than straight
        return 1
    
    # Check for sudden direction changes
    if len(segs) >= 2:
        angles = np.arctan2(segs[:, 1], segs[:, 0])
        angle_diffs = np.diff(angles)
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
        
        # Large sudden changes indicate abnormal behavior
        if np.any(np.abs(angle_diffs) > np.deg2rad(90)):
            return 1
    
    return 0  # Normal

# -----------------------------
# MAIN ANALYSIS FUNCTION
# -----------------------------
def main():
    """
    Executes the complete trajectory analysis pipeline, including data loading, feature extraction, angle analysis, model training with hyperparameter tuning, performance evaluation, and visualization.
    
    Returns:
        dict: A dictionary containing the trained model, performance metrics, feature importance, angle analyses, test predictions, and test labels.
    """
    print("Starting Comprehensive Trajectory Angle Analysis...")
    print("="*60)
    
    # Load data
    trajs, fnames = load_trajectories(FOLDER_PATH, N_POINTS)
    if not trajs:
        print("No trajectories loaded. Exiting.")
        return
    
    # Initialize analyzers
    angle_analyzer = TrajectoryAngleAnalyzer()
    visualizer = Trajectory3DVisualizer(OUTPUT_DIR)
    performance_analyzer = PerformanceAnalyzer(OUTPUT_DIR)
    
    # Analyze angles for all trajectories
    print(f"Analyzing angles for {len(trajs)} trajectories...")
    angle_analyses = []
    for i, traj in enumerate(trajs):
        analysis = angle_analyzer.analyze_trajectory_angles(traj)
        angle_analyses.append(analysis)
        if i % 50 == 0:
            print(f"Processed {i+1}/{len(trajs)} trajectories")
    
    # Load ground truth or use rule-based labels
    gt = load_ground_truth_from_csv(GROUND_TRUTH_CSV_PATH, fnames)
    if gt is not None and len(gt) == len(trajs):
        print("Using ground truth labels")
        y = gt
    else:
        print("Using rule-based labels")
        y = np.array([assign_simple_binary_label(t, central_zone_polygon, THRESHOLD_PARAMS) 
                     for t in trajs])
    
    print(f"Label distribution: Normal={np.sum(y==0)}, Abnormal={np.sum(y==1)}")
    
    # Extract comprehensive features
    print("Extracting comprehensive features...")
    X = build_comprehensive_feature_matrix(trajs, THRESHOLD_PARAMS, central_zone_polygon, angle_analyzer)
    feature_names = get_comprehensive_feature_names()
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    # Split data
    indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=42, stratify=y[train_val_idx])
    
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    
    trajs_train = [trajs[i] for i in train_idx]
    trajs_val = [trajs[i] for i in val_idx]
    trajs_test = [trajs[i] for i in test_idx]
    
    angle_analyses_train = [angle_analyses[i] for i in train_idx]
    angle_analyses_val = [angle_analyses[i] for i in val_idx]
    angle_analyses_test = [angle_analyses[i] for i in test_idx]
    
    print(f"Data split - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Train model with hyperparameter tuning
    print("Training Decision Tree with hyperparameter tuning...")
    best_model = tune_and_train_decision_tree(X_train, y_train, X_val, y_val)
    
    # Test model performance
    print("Evaluating on test set...")
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
    
    # Comprehensive performance analysis
    performance_metrics = performance_analyzer.analyze_model_performance(
        y_test, y_pred, y_prob, ['Normal', 'Abnormal']
    )
    
    # Feature importance analysis
    print("Analyzing feature importance...")
    feature_importance = best_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df.head(15), x='importance', y='feature')
    plt.title('Feature Importance (Top 15)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=300)
    plt.show()
    
    # Create 3D visualizations
    print("Creating 3D visualizations...")
    
    # 3D feature space visualization
    visualizer.create_3d_feature_space(X_test, y_test, feature_names, 
                                     "3D Feature Space (Test Set)")
    
    # 3D angle evolution visualization
    visualizer.create_angle_evolution_3d(trajs_test[:20], angle_analyses_test[:20], y_test[:20])
    
    # Create detailed angle analysis plots
    print("Creating detailed angle analysis plots...")
    create_detailed_angle_plots(angle_analyses_test, y_test, OUTPUT_DIR)
    
    # Trajectory surface visualization (using plotly)
    try:
        visualizer.create_trajectory_surface_plot(trajs_test[:10], angle_analyses_test[:10], y_test[:10])
    except Exception as e:
        print(f"Note: Could not create surface plot - {e}")
    
    # Analysis summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total trajectories analyzed: {len(trajs)}")
    print(f"Features extracted: {len(feature_names)}")
    print(f"Test set accuracy: {performance_metrics['accuracy']:.4f}")
    print(f"Test set precision: {performance_metrics['precision']:.4f}")
    print(f"Test set recall: {performance_metrics['recall']:.4f}")
    print(f"Test set F1-score: {performance_metrics['f1']:.4f}")
    print(f"\nTop 3 most important features:")
    for i, row in importance_df.head(3).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    
    return {
        'model': best_model,
        'performance': performance_metrics,
        'feature_importance': importance_df,
        'angle_analyses': angle_analyses,
        'test_predictions': y_pred,
        'test_labels': y_test
    }

def create_detailed_angle_plots(angle_analyses: List[Dict], labels: np.ndarray, output_dir: str):
    """
    Generates and saves comparative plots of angle-based trajectory metrics for normal and abnormal classes.
    
    Creates histograms and box plots to visualize and compare distributions of mean angular velocity, mean curvature, sharp turns, total turn angle, complexity score, and maximum angular acceleration between normal and abnormal trajectories. Plots are saved to the specified output directory.
    """
    
    # Prepare data for plotting
    normal_analyses = [analysis for analysis, label in zip(angle_analyses, labels) if label == 0]
    abnormal_analyses = [analysis for analysis, label in zip(angle_analyses, labels) if label == 1]
    
    # Plot 1: Distribution of angular velocities
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    normal_ang_vel = [analysis['mean_angular_velocity'] for analysis in normal_analyses if analysis['mean_angular_velocity'] > 0]
    abnormal_ang_vel = [analysis['mean_angular_velocity'] for analysis in abnormal_analyses if analysis['mean_angular_velocity'] > 0]
    
    plt.hist(normal_ang_vel, bins=20, alpha=0.7, label='Normal', color='green')
    plt.hist(abnormal_ang_vel, bins=20, alpha=0.7, label='Abnormal', color='red')
    plt.xlabel('Mean Angular Velocity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Mean Angular Velocities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of curvatures
    plt.subplot(2, 3, 2)
    normal_curv = [analysis['mean_curvature'] for analysis in normal_analyses]
    abnormal_curv = [analysis['mean_curvature'] for analysis in abnormal_analyses]
    
    plt.hist(normal_curv, bins=20, alpha=0.7, label='Normal', color='green')
    plt.hist(abnormal_curv, bins=20, alpha=0.7, label='Abnormal', color='red')
    plt.xlabel('Mean Curvature')
    plt.ylabel('Frequency')
    plt.title('Distribution of Mean Curvatures')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Sharp turns comparison
    plt.subplot(2, 3, 3)
    normal_turns = [analysis['sharp_turns'] for analysis in normal_analyses]
    abnormal_turns = [analysis['sharp_turns'] for analysis in abnormal_analyses]
    
    plt.hist(normal_turns, bins=10, alpha=0.7, label='Normal', color='green')
    plt.hist(abnormal_turns, bins=10, alpha=0.7, label='Abnormal', color='red')
    plt.xlabel('Number of Sharp Turns')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sharp Turns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Total turn angle comparison
    plt.subplot(2, 3, 4)
    normal_total_turn = [analysis['total_turn_angle'] for analysis in normal_analyses]
    abnormal_total_turn = [analysis['total_turn_angle'] for analysis in abnormal_analyses]
    
    plt.hist(normal_total_turn, bins=20, alpha=0.7, label='Normal', color='green')
    plt.hist(abnormal_total_turn, bins=20, alpha=0.7, label='Abnormal', color='red')
    plt.xlabel('Total Turn Angle (degrees)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Total Turn Angles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Complexity score comparison
    plt.subplot(2, 3, 5)
    normal_complexity = [analysis['complexity_score'] for analysis in normal_analyses if not np.isnan(analysis['complexity_score'])]
    abnormal_complexity = [analysis['complexity_score'] for analysis in abnormal_analyses if not np.isnan(analysis['complexity_score'])]
    
    plt.hist(normal_complexity, bins=20, alpha=0.7, label='Normal', color='green')
    plt.hist(abnormal_complexity, bins=20, alpha=0.7, label='Abnormal', color='red')
    plt.xlabel('Complexity Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Complexity Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Angular acceleration comparison
    plt.subplot(2, 3, 6)
    normal_ang_acc = [analysis['max_angular_acceleration'] for analysis in normal_analyses if analysis['max_angular_acceleration'] > 0]
    abnormal_ang_acc = [analysis['max_angular_acceleration'] for analysis in abnormal_analyses if analysis['max_angular_acceleration'] > 0]
    
    plt.hist(normal_ang_acc, bins=20, alpha=0.7, label='Normal', color='green')
    plt.hist(abnormal_ang_acc, bins=20, alpha=0.7, label='Abnormal', color='red')
    plt.xlabel('Max Angular Acceleration')
    plt.ylabel('Frequency')
    plt.title('Distribution of Max Angular Accelerations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_angle_analysis.png'), dpi=300)
    plt.show()
    
    # Create box plots for better comparison
    plt.figure(figsize=(15, 8))
    
    # Prepare data for box plots
    metrics = ['mean_angular_velocity', 'mean_curvature', 'sharp_turns', 'total_turn_angle', 'complexity_score']
    metric_labels = ['Mean Angular\nVelocity', 'Mean\nCurvature', 'Sharp\nTurns', 'Total Turn\nAngle', 'Complexity\nScore']
    
    data_for_boxplot = []
    labels_for_boxplot = []
    positions = []
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        normal_values = [analysis[metric] for analysis in normal_analyses 
                        if not np.isnan(analysis[metric]) and not np.isinf(analysis[metric])]
        abnormal_values = [analysis[metric] for analysis in abnormal_analyses 
                          if not np.isnan(analysis[metric]) and not np.isinf(analysis[metric])]
        
        data_for_boxplot.extend([normal_values, abnormal_values])
        labels_for_boxplot.extend([f'{label}\nNormal', f'{label}\nAbnormal'])
        positions.extend([i*2 + 1, i*2 + 2])
    
    bp = plt.boxplot(data_for_boxplot, positions=positions, patch_artist=True)
    
    # Color the boxes
    colors = ['lightgreen', 'lightcoral'] * len(metrics)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.xticks(positions, labels_for_boxplot, rotation=45, ha='right')
    plt.ylabel('Value')
    plt.title('Comparison of Angle-Related Metrics Between Normal and Abnormal Trajectories')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'angle_metrics_boxplot.png'), dpi=300)
    plt.show()

def tune_and_train_decision_tree(X_train, y_train, X_val, y_val, random_state=42):
    """
    Performs hyperparameter tuning and training of a decision tree classifier using grid search.
    
    Fits a decision tree with cross-validated grid search over multiple hyperparameters, selects the best model based on weighted F1-score, evaluates its performance on a validation set, and returns the trained classifier.
    
    Args:
    	X_train: Training feature matrix.
    	y_train: Training labels.
    	X_val: Validation feature matrix.
    	y_val: Validation labels.
    	random_state: Random seed for reproducibility.
    
    Returns:
    	The best trained DecisionTreeClassifier instance.
    """
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    param_grid = {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': [None, 'sqrt', 'log2', 0.8],
        'class_weight': ['balanced', None],
        'criterion': ['gini', 'entropy']
    }
    
    base = DecisionTreeClassifier(random_state=random_state)
    gs = GridSearchCV(base, param_grid, scoring='f1_weighted', cv=inner_cv, n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    
    best = gs.best_estimator_
    print("Best parameters:", gs.best_params_)
    
    # Validation performance
    y_val_pred = best.predict(X_val)
    print("-- Validation Performance --")
    print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"F1-Score: {f1_score(y_val, y_val_pred, average='weighted'):.4f}")
    print(f"Precision: {precision_score(y_val, y_val_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_val, y_val_pred, average='weighted'):.4f}")
    
    return best

if __name__ == "__main__":
                results = main()
