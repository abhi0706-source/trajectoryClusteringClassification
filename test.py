import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.manifold import TSNE

def load_trajectory(file_path):
    """
    Load trajectory data from CSV file
    """
    try:
        df = pd.read_csv(file_path)
        # Calculate center coordinates
        df['center_x'] = df['left'] + df['w'] / 2
        df['center_y'] = df['top'] + df['h'] / 2
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def visualize_trajectory(file_path, background_image=None, title=None, ax=None):
    """
    Visualize a single trajectory on a given axis
    """
    df = load_trajectory(file_path)
    if df is None:
        return
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # If background image is provided, display it
    if background_image is not None:
        ax.imshow(background_image)
    
    # Plot the trajectory
    ax.plot(df['center_x'], df['center_y'], 'b-', linewidth=2)
    
    # Mark start and end points
    ax.plot(df['center_x'].iloc[0], df['center_y'].iloc[0], 'go', markersize=10)  # Start (green)
    ax.plot(df['center_x'].iloc[-1], df['center_y'].iloc[-1], 'ro', markersize=10)  # End (red)
    
    # Add time indicators
    n_points = len(df)
    stride = max(1, n_points // 10)  # Show at most 10 time indicators
    for i in range(0, n_points, stride):
        frame = df['frameNo'].iloc[i]
        ax.annotate(f"{frame}", (df['center_x'].iloc[i], df['center_y'].iloc[i]), 
                   fontsize=8, color='k', backgroundcolor='w')
    
    if title:
        ax.set_title(title)
    
    ax.set_aspect('equal')
    return ax

def visualize_normal_vs_abnormal(results_df, dataset_folders=['10', '11', '12'], n_samples=3):
    """
    Visualize examples of normal and abnormal trajectories
    """
    # Load sample background images
    background_images = {}
    for folder in dataset_folders:
        image_path = f"sample{folder}.jpg"
        if os.path.exists(image_path):
            background_images[folder] = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    
    for folder in dataset_folders:
        # Filter trajectories for this folder
        folder_results = results_df[results_df['file_path'].str.startswith(folder)]
        
        # Get some normal and abnormal examples
        normal_examples = folder_results[folder_results['true_label'] == 0].head(n_samples)
        abnormal_examples = folder_results[folder_results['true_label'] == 1].head(n_samples)
        
        # Get background image
        bg_image = background_images.get(folder, None)
        
        # Visualize normal examples
        fig, axes = plt.subplots(1, n_samples, figsize=(15, 5))
        fig.suptitle(f'Normal Trajectories (Dataset {folder})', fontsize=16)
        
        for i, (_, row) in enumerate(normal_examples.iterrows()):
            title = f"Normal: {os.path.basename(row['file_path'])}"
            visualize_trajectory(row['file_path'], bg_image, title, axes[i] if n_samples > 1 else axes)
            
        plt.tight_layout()
        plt.savefig(f'normal_trajectories_{folder}.png', dpi=300)
        
        # Visualize abnormal examples
        fig, axes = plt.subplots(1, n_samples, figsize=(15, 5))
        fig.suptitle(f'Abnormal Trajectories (Dataset {folder})', fontsize=16)
        
        for i, (_, row) in enumerate(abnormal_examples.iterrows()):
            title = f"Abnormal: {os.path.basename(row['file_path'])}"
            visualize_trajectory(row['file_path'], bg_image, title, axes[i] if n_samples > 1 else axes)
            
        plt.tight_layout()
        plt.savefig(f'abnormal_trajectories_{folder}.png', dpi=300)

def visualize_feature_distributions(features_df):
    """
    Visualize distributions of key features for normal vs abnormal trajectories
    """
    # Select a subset of important features
    features = [
        'path_length', 'bbox_area', 'avg_speed', 'max_speed', 
        'std_speed', 'mean_accel', 'total_direction_change'
    ]
    
    # Create figure
    fig, axes = plt.subplots(len(features), 1, figsize=(12, 4*len(features)))
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        # Plot distributions
        normal_values = features_df[features_df['label'] == 0][feature]
        abnormal_values = features_df[features_df['label'] == 1][feature]
        
        ax.hist(normal_values, bins=30, alpha=0.5, label='Normal', color='green')
        ax.hist(abnormal_values, bins=30, alpha=0.5, label='Abnormal', color='red')
        
        ax.set_title(f'Distribution of {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Calculate and display statistics
        normal_mean = normal_values.mean()
        abnormal_mean = abnormal_values.mean()
        
        ax.axvline(normal_mean, color='green', linestyle='--', alpha=0.8)
        ax.axvline(abnormal_mean, color='red', linestyle='--', alpha=0.8)
        
        # Add text with statistics
        stats_text = (
            f"Normal - Mean: {normal_mean:.2f}, Std: {normal_values.std():.2f}\n"
            f"Abnormal - Mean: {abnormal_mean:.2f}, Std: {abnormal_values.std():.2f}"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300)
    plt.show()

def visualize_feature_space(features_df):
    """
    Visualize trajectory features in a reduced dimensionality space using t-SNE
    """
    # Extract feature matrix
    X = features_df.drop(['label', 'file_path'], axis=1)
    X = X.fillna(X.mean())
    
    # Apply t-SNE for visualization
    print("Applying t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    
    # Get labels
    labels = features_df['label'].values
    
    # Plot normal trajectories
    normal_mask = labels == 0
    plt.scatter(X_tsne[normal_mask, 0], X_tsne[normal_mask, 1], 
                c='green', label='Normal', alpha=0.7, s=50, edgecolor='w')
    
    # Plot abnormal trajectories
    abnormal_mask = labels == 1
    plt.scatter(X_tsne[abnormal_mask, 0], X_tsne[abnormal_mask, 1], 
                c='red', label='Abnormal', alpha=0.7, s=50, edgecolor='w')
    
    plt.title('t-SNE Visualization of Trajectory Features', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(fontsize=12)
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('tsne_visualization.png', dpi=300)
    plt.show()

def analyze_misclassifications(results_df, features_df):
    """
    Analyze trajectories that were misclassified
    """
    # Merge results with features
    merged_df = pd.merge(
        results_df, 
        features_df,
        left_on='file_path', 
        right_on='file_path'
    )
    
    # Find misclassified trajectories
    misclassified = merged_df[merged_df['true_label'] != merged_df['predicted_label']]
    
    print(f"Number of misclassified trajectories: {len(misclassified)}")
    print(f"False positives (normal classified as abnormal): {sum((misclassified['true_label'] == 0) & (misclassified['predicted_label'] == 1))}")
    print(f"False negatives (abnormal classified as normal): {sum((misclassified['true_label'] == 1) & (misclassified['predicted_label'] == 0))}")
    
    # Select a few misclassifications to visualize
    false_positives = misclassified[(misclassified['true_label'] == 0) & (misclassified['predicted_label'] == 1)].head(3)
    false_negatives = misclassified[(misclassified['true_label'] == 1) & (misclassified['predicted_label'] == 0)].head(3)
    
    # Load background images
    bg_images = {}
    for folder in ['10', '11', '12']:
        image_path = f"sample{folder}.jpg"
        if os.path.exists(image_path):
            bg_images[folder] = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    
    # Visualize false positives
    if len(false_positives) > 0:
        n_samples = len(false_positives)
        fig, axes = plt.subplots(1, n_samples, figsize=(15, 5))
        fig.suptitle('False Positives (Normal classified as Abnormal)', fontsize=16)
        
        for i, (_, row) in enumerate(false_positives.iterrows()):
            folder = row['file_path'].split(os.path.sep)[0]
            bg_image = bg_images.get(folder, None)
            title = f"FP: {os.path.basename(row['file_path'])}"
            ax = axes[i] if n_samples > 1 else axes
            visualize_trajectory(row['file_path'], bg_image, title, ax)
            
        plt.tight_layout()
        plt.savefig('false_positives.png', dpi=300)
    
    # Visualize false negatives
    if len(false_negatives) > 0:
        n_samples = len(false_negatives)
        fig, axes = plt.subplots(1, n_samples, figsize=(15, 5))
        fig.suptitle('False Negatives (Abnormal classified as Normal)', fontsize=16)
        
        for i, (_, row) in enumerate(false_negatives.iterrows()):
            folder = row['file_path'].split(os.path.sep)[0]
            bg_image = bg_images.get(folder, None)
            title = f"FN: {os.path.basename(row['file_path'])}"
            ax = axes[i] if n_samples > 1 else axes
            visualize_trajectory(row['file_path'], bg_image, title, ax)
            
        plt.tight_layout()
        plt.savefig('false_negatives.png', dpi=300)

def main():
    # Load results and features
    if not os.path.exists('classification_results.csv') or not os.path.exists('trajectory_features.csv'):
        print("Please run the main DBSCAN analysis script first.")
        return
    
    results_df = pd.read_csv('classification_results.csv')
    features_df = pd.read_csv('trajectory_features.csv')
    
    # Visualize normal vs abnormal examples
    print("Visualizing normal vs abnormal trajectories...")
    visualize_normal_vs_abnormal(results_df)
    
    # Visualize feature distributions
    print("Visualizing feature distributions...")
    visualize_feature_distributions(features_df)
    
    # Visualize feature space
    print("Visualizing feature space...")
    visualize_feature_space(features_df)
    
    # Analyze misclassifications
    print("Analyzing misclassifications...")
    analyze_misclassifications(results_df, features_df)
    
    print("Analysis complete. Visualization images saved.")

if __name__ == "__main__":
    main()