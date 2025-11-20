import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import re

def plot_comparison(log_files):
    """
    Reads multiple training log files and plots a comparison of their
    training and test losses on two separate subplots.
    """
    if not log_files:
        print("Error: No log files provided.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    for log_file in log_files:
        # Check if file exists
        if not os.path.exists(log_file):
            print(f"Warning: Log file not found at {log_file}, skipping.")
            continue

        # Read the data
        df = pd.read_csv(log_file)

        # Create a label for the plot from the filename (e.g., "ResNet-20")
        match = re.search(r'log_(resnet|plainnet)_n(\d+)\.csv', os.path.basename(log_file))
        if match:
            model_type = match.group(1).capitalize()
            n_size = int(match.group(2))
            layers = 6 * n_size + 2
            label = f'{model_type}-{layers}'
        else:
            label = os.path.basename(log_file)

        # Plot Training Loss
        ax1.plot(df['epoch'], df['train_loss'], marker='o', linestyle='-', label=label)
        
        # Plot Test Loss
        ax2.plot(df['epoch'], df['test_loss'], marker='o', linestyle='-', label=label)

    # --- Formatting for Training Loss Plot ---
    ax1.set_title('Training Loss Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    # --- Formatting for Test Loss Plot ---
    ax2.set_title('Test Loss Comparison')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend()
    
    # Adjust layout and save the plot
    plt.tight_layout()
    output_filename = 'comparison_loss_graph.png'
    plt.savefig(output_filename)
    print(f"Comparison graph saved to {output_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training/test loss comparison from multiple log files.')
    parser.add_argument('log_files', nargs='+', type=str, help='Paths to the training log CSV files.')
    args = parser.parse_args()
    
    plot_comparison(args.log_files)
