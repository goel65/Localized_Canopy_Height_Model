import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

class MultiModalLSTM:
    def __init__(self, config=None):
        """
        Initialize the multi-modal LSTM model
        """
        self.config = config or {
            'landsat_timesteps': 64,
            'landsat_features': 11,
            'sentinel2_timesteps': 87,
            'sentinel2_features': 14,
            'sentinel1_timesteps': 85,
            'sentinel1_features': 2,
            'nlcd_classes': 10
        }
        self.model = None
        
    def build_model(self):
        """
        Build the multi-modal LSTM model architecture
        """
        # Input layers
        landsat_input = Input(shape=(self.config['landsat_timesteps'], self.config['landsat_features']))
        sentinel2_input = Input(shape=(self.config['sentinel2_timesteps'], self.config['sentinel2_features']))
        sentinel1_input = Input(shape=(self.config['sentinel1_timesteps'], self.config['sentinel1_features']))
        nlcd_input = Input(shape=(self.config['nlcd_classes'],))
        
        # Landsat branch
        lstm1 = LSTM(units=160, return_sequences=False)(landsat_input)
        lstm1 = Dropout(0.2)(lstm1)
        dense1 = Dense(units=320, activation='relu')(lstm1)
        
        # Sentinel-2 branch
        lstm2 = LSTM(units=96, return_sequences=False)(sentinel2_input)
        lstm2 = Dropout(0.15)(lstm2)
        dense2 = Dense(units=448, activation='relu')(lstm2)
        
        # Sentinel-1 branch
        lstm3 = LSTM(units=128, return_sequences=False)(sentinel1_input)
        lstm3 = Dropout(0.15)(lstm3)
        dense3 = Dense(units=384, activation='relu')(lstm3)
        
        # NLCD branch
        dense4 = Dense(units=256, activation='relu')(nlcd_input)
        
        # Merge all branches
        concatenated = Concatenate()([dense1, dense2, dense3, dense4])
        fc1 = Dense(units=480, activation='relu')(concatenated)
        fc2 = Dense(units=64, activation='relu')(fc1)
        fc2 = Dropout(0.05)(fc2)
        
        # Output layer
        output = Dense(units=1, activation='linear')(fc2)
        
        # Create model
        self.model = Model(
            inputs=[landsat_input, sentinel2_input, sentinel1_input, nlcd_input],
            outputs=output
        )
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='mean_absolute_error',
            metrics=['mean_absolute_error', 'mean_squared_error']
        )
        
        return self.model
    
    def train(self, train_data, validation_data, epochs=60, batch_size=32):
        """
        Train the model with early stopping
        """
        if self.model is None:
            self.build_model()
            
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            min_delta=0.001,
            mode='min',
            verbose=1,
            restore_best_weights=True
        )
        
        X_train = [
            train_data['landsat_features'],
            train_data['sentinel2_features'],
            train_data['sentinel1_features'],
            train_data['nlcd']
        ]
        
        X_val = [
            validation_data['landsat_features'],
            validation_data['sentinel2_features'],
            validation_data['sentinel1_features'],
            validation_data['nlcd']
        ]
        
        history = self.model.fit(
            X_train, train_data['gedi'],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, validation_data['gedi']),
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history
    
    def predict(self, test_data):
        """
        Make predictions on test data
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained yet")
            
        X_test = [
            test_data['landsat_features'],
            test_data['sentinel2_features'],
            test_data['sentinel1_features'],
            test_data['nlcd']
        ]
        
        return self.model.predict(X_test)
    
    def save(self, filepath):
        """
        Save the model to disk
        """
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath):
        """
        Load a saved model from disk
        """
        instance = cls()
        instance.model = tf.keras.models.load_model(filepath)
        return instance 