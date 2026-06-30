# ==========================================================
# Dataset Configuration
# ==========================================================
DATA_PATH = "data_B (1).csv"
TARGET_COLUMN = "Credit_Score"

# Columns dropped immediately (identifiers / leakage / no predictive value)
DROP_COLUMNS = [
    "Unnamed: 0",
    "ID",
    "Customer_ID",
    "Name",
    "SSN",
    "Month"
]

# ==========================================================
# Train Test Split
# ==========================================================
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

# ==========================================================
# Feature Engineering
# ==========================================================
EPSILON = 1e-6

# ==========================================================
# Model Configuration
# ==========================================================
MODEL_PATH = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

# ==========================================================
# Models
# ==========================================================
MODELS = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "KNN",
    "Naive Bayes",
    "SVM"
]