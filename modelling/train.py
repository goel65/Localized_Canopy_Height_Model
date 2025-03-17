import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils.data_loader import SatelliteDataLoader
from models.multi_modal_lstm import MultiModalLSTM

def parse_args():
    parser = argparse.ArgumentParser(description='Train the multi-modal LSTM model')
    parser.add_argument('--run_number', type=int, default=1,
                      help='Run number for organizing outputs (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate for training (default: 0.001)')
    parser.add_argument('--base_dir', type=str, default='prelim_outputs',
                      help='Base directory containing the data (default: prelim_outputs)')
    return parser.parse_args()

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    output_dir = os.path.join('results', f'run{args.run_number}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data loader
    data_loader = SatelliteDataLoader(args.base_dir)
    
    # Load training data
    print("Loading training data...")
    training_data = data_loader.load_training_data()
    
    # Normalize GEDI data
    gedi_normalized = data_loader.normalize_gedi(training_data['gedi'])
    
    # Create bins for stratification
    bins = np.linspace(0, 1.5, 10)
    y_binned = np.digitize(gedi_normalized, bins)
    
    # Split data into training and validation sets
    print("Splitting data into training and validation sets...")
    split_data = {}
    (
        split_data['landsat_train'], split_data['landsat_val'],
        split_data['sentinel2_train'], split_data['sentinel2_val'],
        split_data['sentinel1_train'], split_data['sentinel1_val'],
        split_data['nlcd_train'], split_data['nlcd_val'],
        split_data['gedi_train'], split_data['gedi_val']
    ) = train_test_split(
        training_data['landsat_features'],
        training_data['sentinel2_features'],
        training_data['sentinel1_features'],
        training_data['nlcd'],
        gedi_normalized,
        test_size=0.2,
        stratify=y_binned
    )
    
    # Prepare training and validation data dictionaries
    train_dict = {
        'landsat_features': split_data['landsat_train'],
        'sentinel2_features': split_data['sentinel2_train'],
        'sentinel1_features': split_data['sentinel1_train'],
        'nlcd': split_data['nlcd_train'],
        'gedi': split_data['gedi_train']
    }
    
    val_dict = {
        'landsat_features': split_data['landsat_val'],
        'sentinel2_features': split_data['sentinel2_val'],
        'sentinel1_features': split_data['sentinel1_val'],
        'nlcd': split_data['nlcd_val'],
        'gedi': split_data['gedi_val']
    }
    
    # Initialize model
    print("Initializing model...")
    model = MultiModalLSTM(
        landsat_timesteps=64,
        sentinel1_timesteps=85,
        sentinel2_timesteps=87,
        landsat_features=11,
        sentinel1_features=2,
        sentinel2_features=14,
        nlcd_classes=10
    )
    
    # Train model
    print("Training model...")
    history = model.train(
        train_dict,
        val_dict,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    model_path = os.path.join(output_dir, 'multi_modal_lstm.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.npz')
    np.savez(history_path, **history)
    print(f"Training history saved to {history_path}")

if __name__ == '__main__':
    main() 