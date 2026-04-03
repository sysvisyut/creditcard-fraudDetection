DATA_PATH = "data/creditcard.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
N_ESTIMATORS = 100

OUTPUT_PLOTS = "outputs/plots/"
OUTPUT_MODELS = "outputs/models/"

# Features array to be populated dynamically by feature selection downstream
SELECTED_FEATURES = []

# Other constants replacing magic numbers originally mapped in main.py
SAMPLING_N_SAMPLES_PCA = 5000
RF_MAX_DEPTH = 5
LR_MAX_ITER = 1000
TRAIN_TEST_SPLIT_TEMP = 0.30
TRAIN_TEST_SPLIT_VAL = 0.50
LR_C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100]
PCA_COMPONENTS = 2
