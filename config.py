# config.py
class Config:
    # Hyperparameters
    NUM_PARTICLES = 1000  # N_s: Number of particles (User requested parameter)
    
    # Dataset Config
    DATASET_PATH = 'data/train_FD001.txt'  # Path to training data
    TEST_PATH = 'data/test_FD001.txt'      # Path to test data
    RUL_PATH = 'data/RUL_FD001.txt'        # Path to RUL truth
    
    # Model Config
    FAILURE_THRESHOLD = 1.0     # D = 1 (Normalized state)
    DT = 1.0                    # Time interval (cycle)
    
    # Sensor Config (NASA C-MAPSS has 21 sensors)
    TOTAL_SENSORS = 21
    # Columns in C-MAPSS FD001
    INDEX_COLS = ['unit_nr', 'time_cycles']
    SETTING_COLS = ['setting_1', 'setting_2', 'setting_3']
    SENSOR_COLS = ['s_{}'.format(i) for i in range(1, 22)]
    
    # Optimization
    SMOOTHING_FRAC = 0.1 # Lowess smoothing fraction