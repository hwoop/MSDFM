# data_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import Config

def load_data():
    """
    Loads NASA FD001 dataset.
    Assumes files are in the path specified in Config.
    Returns:
        train_df, test_df, test_rul
    """
    # Load Training Data
    try:
        train_df = pd.read_csv(Config.DATASET_PATH, sep=r'\s+', header=None, 
                               names=Config.INDEX_COLS + Config.SETTING_COLS + Config.SENSOR_COLS)
        test_df = pd.read_csv(Config.TEST_PATH, sep=r'\s+', header=None, 
                              names=Config.INDEX_COLS + Config.SETTING_COLS + Config.SENSOR_COLS)
        test_rul = pd.read_csv(Config.RUL_PATH, sep=r'\s+', header=None, names=['RUL'])
    except FileNotFoundError:
        print("Error: Dataset files not found. Please ensure NASA C-MAPSS data is in 'data/' folder.")
        raise

    # Normalization (MinMax Scaling)
    # Note: The paper implies using degradation signals derived from sensors.
    # We apply MinMax scaling to map sensor values roughly to 0-1 range relative to their max degradation.
    # However, strictly speaking, MSDFM maps State(0~1) to Sensor values.
    # We normalize sensors to make optimization stable.
    
    scaler = MinMaxScaler()
    train_df[Config.SENSOR_COLS] = scaler.fit_transform(train_df[Config.SENSOR_COLS])
    test_df[Config.SENSOR_COLS] = scaler.transform(test_df[Config.SENSOR_COLS])

    return train_df, test_df, test_rul

def get_lifetimes(train_df):
    """Extracts lifetimes (max cycles) for each unit."""
    return train_df.groupby('unit_nr')['time_cycles'].max().values