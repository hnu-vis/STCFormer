import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict
import torch

def visualize_predictions(returns_all: Dict, save_dir: str, model_name: str, num_samples: int = 7):
    """Visualize prediction results at selected timesteps and stations.
    
    Args:
        returns_all (Dict): Dictionary containing 'prediction', 'target', 'inputs'
        save_dir (str): Directory to save the visualization PDFs
        model_name (str): Name of the model for the title
        num_samples (int): Number of samples to take along each dimension (default: 7)
    """
    
    # Create save directory if not exists
    vis_save_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_save_dir, exist_ok=True)
    
    # Extract data
    inputs = returns_all['inputs']  # (total_steps, hist_len, num_stations, 1)
    target = returns_all['target']  # (total_steps, pred_len, num_stations, 1)
    prediction = returns_all['prediction']  # (total_steps, pred_len, num_stations, 1)
    
    # Get dimensions
    total_steps = inputs.shape[0]
    hist_len = inputs.shape[1]
    pred_len = target.shape[1]
    num_stations = inputs.shape[2]
    
    # Convert to numpy if tensors
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    
    # Select indices
    step_indices = np.linspace(0, total_steps - 1, num_samples, dtype=int)
    station_indices = np.linspace(0, num_stations - 1, num_samples, dtype=int)
    
    # Set matplotlib style for better appearance
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 20,
        'axes.titlesize': 22,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 18,
        'figure.figsize': (12, 7),
        'lines.linewidth': 3,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.facecolor': '#f8f9fa',
        'figure.facecolor': 'white'
    })
    
    print(f"Generating {num_samples * num_samples} visualization plots...")
    
    # Generate plots
    for i, step_idx in enumerate(step_indices):
        for j, station_idx in enumerate(station_indices):
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Prepare data for this step and station
            hist_data = inputs[step_idx, :, station_idx, 0]  # (hist_len,)
            target_data = target[step_idx, :, station_idx, 0]  # (pred_len,)
            pred_data = prediction[step_idx, :, station_idx, 0]  # (pred_len,)
            
            # Create x-axis
            hist_x = np.arange(0, hist_len)
            pred_x = np.arange(hist_len, hist_len + pred_len)
            
            # Plot historical data with smooth line
            ax.plot(hist_x, hist_data, color='#7f8c8d', linewidth=3.5, 
                   label='Historical', alpha=0.8, linestyle='-')
            
            # Plot ground truth and prediction with smooth lines
            ax.plot(pred_x, target_data, color='#e67e22', linewidth=3.5, 
                   label='Ground Truth', alpha=0.9, linestyle='-')
            ax.plot(pred_x, pred_data, color='#3498db', linewidth=3.5, 
                   label='Prediction', alpha=0.9, linestyle='-')
            
            # Add vertical line to separate history and future (without label)
            ax.axvline(x=hist_len - 0.5, color='#e74c3c', linestyle='--', 
                      linewidth=2.5, alpha=0.6)
            
            # Customize axes
            ax.set_xlabel('Time', fontsize=20, fontweight='bold')
            ax.set_ylabel('Value', fontsize=20, fontweight='bold')
            ax.set_title(f'{model_name}', 
                        fontsize=24, fontweight='bold', pad=20)
            
            # Legend with better positioning
            legend = ax.legend(fontsize=18, loc='best', frameon=True, 
                             shadow=True, fancybox=True, framealpha=0.95)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('#bdc3c7')
            
            # Grid styling
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=1)
            ax.set_axisbelow(True)
            
            # Spine styling
            for spine in ax.spines.values():
                spine.set_edgecolor('#bdc3c7')
                spine.set_linewidth(1.5)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            filename = f'pred_step{i+1}_station{j+1}.pdf'
            save_path = os.path.join(vis_save_dir, filename)
            plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            # Print progress
            if (i * num_samples + j + 1) % 10 == 0:
                print(f"  Generated {i * num_samples + j + 1}/{num_samples * num_samples} plots")
    
    print(f"All visualizations saved to: {vis_save_dir}")
    print(f"Total plots generated: {num_samples * num_samples}")
    
    # Reset style
    plt.style.use('default')
    plt.rcParams.update(plt.rcParamsDefault)