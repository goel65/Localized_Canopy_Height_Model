# Multi-Modal LSTM Model for Satellite Data Processing

This repository contains a multi-modal LSTM model for processing satellite data, including Landsat, Sentinel-1, and Sentinel-2 data. The model is designed to predict Canopy Height Model (CHM) values using a combination of satellite imagery and ground truth data.

## Repository Structure

```
.
├── preprocessing/                    # Preprocessing notebooks
│   ├── landsat_preprocessing.ipynb   # Landsat data preprocessing
│   ├── sentinel2_preprocessing.ipynb # Sentinel-2 data preprocessing
│   └── s1_creating_array.ipynb       # Sentinel-1 data preprocessing
├── data_processing/                  # Data processing scripts
│   ├── utils/                        # Utility functions
│   │   ├── GEDI_functions.py        # GEDI data processing utilities
│   │   ├── HLS_functions.py         # Landsat data processing utilities
│   │   ├── NLCD_functions.py        # NLCD data processing utilities
│   │   └── SEN1_functions.py        # Sentinel-1 data processing utilities
│   ├── gedi_nlcd_processing.py      # GEDI and NLCD data processing
│   ├── landsat_processing.py        # Landsat data processing
│   ├── sentinel1_processing.py      # Sentinel-1 data processing
│   └── sentinel2_processing.py      # Sentinel-2 data processing
├── modelling/                        # Model-related code
│   ├── utils/                       # Model-specific utilities
│   ├── models/                      # Model architecture
│   ├── train.py                     # Training script
│   └── predict.py                   # Prediction script
├── rs.py                           # Remote sensing utilities
└── requirements.txt                 # Project dependencies
```

## Workflow Overview

The project follows a two-phase workflow:

### Phase 1: Manual Data Preprocessing (Jupyter Notebooks)
The Jupyter notebooks in the `preprocessing/` directory are used manually to clean and prepare raw satellite data. These notebooks handle:
- Data cleaning and validation
- Format conversion
- Initial feature extraction
- Quality checks

Run these notebooks in the following order:
1. `landsat_preprocessing.ipynb`
2. `sentinel2_preprocessing.ipynb`
3. `s1_creating_array.ipynb`

**Note**: These notebooks are meant to be run interactively to ensure data quality and proper preprocessing. Once you have clean data in the required format, you can proceed to Phase 2.

### Phase 2: Automated Processing Pipeline
After obtaining clean data from Phase 1, use the automated processing pipeline:

1. **Data Processing** (using scripts in `data_processing/`):
   - `gedi_nlcd_processing.py`: Processes ground truth data
   - `landsat_processing.py`: Processes Landsat feature arrays
   - `sentinel1_processing.py`: Processes Sentinel-1 feature arrays
   - `sentinel2_processing.py`: Processes Sentinel-2 feature arrays

2. **Model Training**:
   - Uses `modelling/train.py` to train the multi-modal LSTM model
   - Supports command-line arguments for run number and training parameters

3. **Model Prediction**:
   - Uses `modelling/predict.py` to generate predictions
   - Creates GeoTIFF raster outputs for visualization

## Usage

### Phase 1: Manual Preprocessing
1. Open and run each preprocessing notebook in order:
   ```bash
   jupyter notebook preprocessing/landsat_preprocessing.ipynb
   jupyter notebook preprocessing/sentinel2_preprocessing.ipynb
   jupyter notebook preprocessing/s1_creating_array.ipynb
   ```
2. Verify the output data format matches the requirements in the Data Organization section
3. Once clean data is obtained, proceed to Phase 2

### Phase 2: Automated Processing

```bash
# Process GEDI and NLCD data
python data_processing/gedi_nlcd_processing.py

# Process Landsat data
python data_processing/landsat_processing.py

# Process Sentinel-2 data
python data_processing/sentinel2_processing.py

# Process Sentinel-1 data
python data_processing/sentinel1_processing.py

# Train the model
python modelling/train.py --run_number 1 --batch_size 32 --epochs 100 --learning_rate 0.001

# Generate predictions
python modelling/predict.py --run_number 1 --batch_size 32
```

## Command-Line Arguments

### Training Script (`modelling/train.py`)
- `--run_number`: Integer for organizing outputs (default: 1)
- `--batch_size`: Integer for batch size (default: 32)
- `--epochs`: Integer for number of training epochs (default: 100)
- `--learning_rate`: Float for learning rate (default: 0.001)
- `--base_dir`: String for base directory containing data (default: 'prelim_output')

### Prediction Script (`modelling/predict.py`)
- `--run_number`: Integer for organizing outputs (default: 1)
- `--batch_size`: Integer for batch size (default: 32)
- `--base_dir`: String for base directory containing data (default: 'prelim_output')

## Output Structure

```
results/
└── run{run_number}/
    ├── multi_modal_lstm.h5          # Trained model
    ├── training_history.npz         # Training history
    └── predictions/
        └── Piute_prediction_CHM_v1.tif  # Final prediction raster
```

## Dependencies

The project requires the following Python packages:
- numpy
- pandas
- tensorflow
- rasterio
- scikit-learn
- matplotlib

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Notes

- The `--run_number` parameter is required for proper organization of model outputs and predictions
- Different run numbers will create separate directories for models and predictions
- The preprocessing notebooks handle the conversion of raw satellite data into feature arrays
- The model architecture is defined in `modelling/models/multi_modal_lstm.py`

## Prerequisites

Before starting the processing pipeline, ensure you have the following data:

1. **Satellite Data Arrays**:
   - Landsat arrays (11 bands) for 64 timestamps
   - Sentinel-2 arrays (14 bands) for 87 timestamps
   - Sentinel-1 arrays (2 bands) for 85 timestamps
   - Each array should be a numpy array with shape (nrows, ncols, nbands)

2. **Raw Data Files**:
   - GEDI raw files (.h5 format)
   - NLCD raw files (2019 land cover .tif format)

## Data Organization

The data should be organized in the following structure:
```
data/
├── landsat_arrays/
│   └── *.npy (64 timestamps)
├── sentinel1_arrays/
│   ├── band1/
│   │   └── *.npy (85 timestamps)
│   └── band2/
│       └── *.npy (85 timestamps)
├── sentinel2_arrays/
│   └── *.npy (87 timestamps)
├── GEDI_raw_files/
│   └── *.h5
└── NLCD_raw_files/
    └── nlcd_2019_land_cover.tif
```

## Model Architecture

The multi-modal LSTM model consists of:
- Three LSTM branches for processing Landsat, Sentinel-1, and Sentinel-2 data
- A dense network branch for processing NLCD data
- A merged network that combines all branches for final prediction 