import os
import argparse
import glob
from utils.data_loader import SatelliteDataLoader
from utils.raster_producer import RasterProducer
from models.multi_modal_lstm import MultiModalLSTM

def parse_args():
    parser = argparse.ArgumentParser(description='Generate predictions using the trained multi-modal LSTM model')
    parser.add_argument('--run_number', type=int, default=1,
                      help='Run number for organizing outputs (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for prediction (default: 32)')
    parser.add_argument('--base_dir', type=str, default='prelim_outputs',
                      help='Base directory containing the data (default: prelim_outputs)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    Piute = {
        "Xmax" : -111.76628273950942,
        "Xmin" : -112.5197057381439,
        "Ymax" : 38.51244321480291,
        "Ymin" : 38.145371577379684,
        "spatial_resolution" :  0.00031021562683723573
    }

    Piute['ncols'] = int((Piute['Xmax'] - Piute['Xmin'])/ Piute['spatial_resolution'])
    Piute['nrows'] = int((Piute['Ymax'] - Piute['Ymin'])/ Piute['spatial_resolution'])

    Location = Piute
    # Create output directory
    output_dir = os.path.join('results', f'run{args.run_number}')
    predictions_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Initialize data loader and raster producer
    data_loader = SatelliteDataLoader(args.base_dir)
    raster_producer = RasterProducer(Location)
    
    # Load the trained model
    model_path = os.path.join(output_dir, 'multi_modal_lstm.h5')
    print(f"Loading model from {model_path}")
    model = MultiModalLSTM.load(model_path)
    
    # Get test data files
    test_files = {
        'coord_files': sorted(glob.glob(os.path.join(args.base_dir, 'landsat/test_files/test_coords_*.npy'))),
        'nlcd_files': sorted(glob.glob(os.path.join(args.base_dir, 'landsat/test_files/nlcd_test_samples_*.npy'))),
        'lan_features_files': sorted(glob.glob(os.path.join(args.base_dir, 'landsat/test_files/test_features_*.npy'))),
        'sen_features_files': sorted(glob.glob(os.path.join(args.base_dir, 'sentinel2/test_files/test_features_*.npy'))),
        'sen1_features_files': sorted(glob.glob(os.path.join(args.base_dir, 'sentinel1/test_files/test_features_*.npy')))
    }
    
    # Process predictions
    print("Generating predictions...")
    raster_producer.process_predictions(
        model=model,
        results_dir=predictions_dir,
        coord_files=test_files['coord_files'],
        nlcd_files=test_files['nlcd_files'],
        lan_features_files=test_files['lan_features_files'],
        sen_features_files=test_files['sen_features_files'],
        sen1_features_files=test_files['sen1_features_files'],
        scaling_params=data_loader.scaling_parameters,
        batch_size=args.batch_size
    )
    
    # Compile all predictions and coordinates
    print("Compiling predictions and coordinates...")
    prediction_files = sorted(glob.glob(os.path.join(predictions_dir, 'predictions_*.npy')))
    
    all_predictions = raster_producer.compile_data_from_lists(prediction_files)
    all_coords = raster_producer.compile_data_from_lists(test_files['coord_files'])
    
    # Produce final raster
    print("Producing final GeoTIFF raster...")
    raster_producer.produce_raster(
        all_coords, 
        all_predictions, 
        os.path.join(predictions_dir, 'Piute_prediction_CHM_v1')
    )
    
    print(f"Predictions and raster completed and saved to {predictions_dir}")

if __name__ == '__main__':
    main() 