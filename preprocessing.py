import re

import joblib
import numpy as np
import pandas as pd

from config import *

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


class Preprocessing:
    """
    Handles cleaning, feature engineering and encoding of the
    credit score dataset. All fitted transformers (encoders,
    scaler, one-hot encoder) are stored on the instance so the
    exact same transformation can be replayed at inference time.
    """

    def __init__(self):

        self.label_encoder = LabelEncoder()

        self.preprocessor = None

    def load_data(self):

        return pd.read_csv(DATA_PATH)

    def _parse_credit_history_age(self, value):
        """Convert '3 Years and 2 Months' -> 38 (total months)."""

        if pd.isna(value):
            return np.nan

        match = re.match(
            r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?",
            str(value)
        )

        if not match:
            return np.nan

        years, months = int(match.group(1)), int(match.group(2))

        return years * 12 + months

    def clean_data(self, df):

        df = df.copy()

        df.drop_duplicates(inplace=True)

        df.drop(
            columns=[c for c in DROP_COLUMNS if c in df.columns],
            inplace=True
        )

        # Strip stray underscores that appear inside numeric-looking strings
        df.replace("_", "", regex=True, inplace=True)

        df["Occupation"] = df["Occupation"].replace("", np.nan)

        df["Credit_Mix"] = df["Credit_Mix"].replace("", np.nan)

        df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("NM", np.nan)

        df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace("NM", np.nan)

        # --- Credit_History_Age: "X Years and Y Months" -> total months ---
        df["Credit_History_Age"] = df["Credit_History_Age"].apply(
            self._parse_credit_history_age
        )

        # --- Type_of_Loan: high-cardinality free text -> numeric count of loans ---
        df["Num_Loan_Types"] = df["Type_of_Loan"].apply(
            lambda x: 0 if pd.isna(x) else len(
                [t for t in str(x).replace("and ", "").split(",") if t.strip()]
            )
        )

        df.drop(columns=["Type_of_Loan"], inplace=True)

        numeric_columns = [
            "Age",
            "Annual_Income",
            "Monthly_Inhand_Salary",
            "Num_Bank_Accounts",
            "Num_Credit_Card",
            "Interest_Rate",
            "Num_of_Loan",
            "Delay_from_due_date",
            "Num_of_Delayed_Payment",
            "Changed_Credit_Limit",
            "Num_Credit_Inquiries",
            "Outstanding_Debt",
            "Credit_Utilization_Ratio",
            "Credit_History_Age",
            "Total_EMI_per_month",
            "Amount_invested_monthly",
            "Monthly_Balance"
        ]

        for col in numeric_columns:

            df[col] = pd.to_numeric(

                df[col],

                errors="coerce"

            )

        df.loc[

            (df["Age"] < 18) |

            (df["Age"] > 100),

            "Age"

        ] = np.nan

        invalid_columns = [
            "Num_Bank_Accounts",
            "Num_Credit_Card",
            "Interest_Rate",
            "Num_of_Loan",
            "Delay_from_due_date",
            "Num_of_Delayed_Payment",
            "Changed_Credit_Limit",
            "Num_Credit_Inquiries",
            "Outstanding_Debt",
            "Credit_Utilization_Ratio",
            "Total_EMI_per_month",
            "Amount_invested_monthly",
            "Monthly_Balance"
        ]

        for col in invalid_columns:

            df.loc[

                df[col] < 0,

                col

            ] = np.nan

        numerical = df.select_dtypes(

            include=np.number

        ).columns

        categorical = df.select_dtypes(

            include="object"

        ).columns

        for col in numerical:

            df[col] = df[col].fillna(df[col].median())

        for col in categorical:

            if col == TARGET_COLUMN:
                continue

            df[col] = df[col].fillna(df[col].mode()[0])

        return df

    def feature_engineering(self, df):

        df = df.copy()

        df[TARGET_COLUMN] = self.label_encoder.fit_transform(

            df[TARGET_COLUMN]

        )

        df["Debt_to_Income_Ratio"] = (

            df["Outstanding_Debt"]

            /

            (df["Annual_Income"] + EPSILON)

        )

        df["Loan_per_Bank_Account"] = (

            df["Num_of_Loan"]

            /

            (df["Num_Bank_Accounts"] + EPSILON)

        )

        df["EMI_to_Income_Ratio"] = (

            df["Total_EMI_per_month"]

            /

            (df["Annual_Income"] + EPSILON)

        )

        df["Investment_to_Income_Ratio"] = (

            df["Amount_invested_monthly"]

            /

            (df["Annual_Income"] + EPSILON)

        )

        df["Credit_per_Account"] = (

            df["Num_Credit_Card"]

            /

            (df["Num_Bank_Accounts"] + EPSILON)

        )

        df["Inquiry_per_Loan"] = (

            df["Num_Credit_Inquiries"]

            /

            (df["Num_of_Loan"] + EPSILON)

        )

        df["Total_Financial_Accounts"] = (

            df["Num_Bank_Accounts"]

            +

            df["Num_Credit_Card"]

        )

        return df

    def split_data(self, df):

        X = df.drop(

            columns=[TARGET_COLUMN]

        )

        y = df[TARGET_COLUMN]

        return train_test_split(

            X,

            y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE,

            stratify=y

        )

    def preprocess_data(

        self,

        X_train,

        X_test

    ):

        categorical_features = X_train.select_dtypes(

            include="object"

        ).columns

        numerical_features = X_train.select_dtypes(

            include=["int64", "float64"]

        ).columns

        self.preprocessor = ColumnTransformer(

            transformers=[

                (

                    "num",

                    StandardScaler(),

                    numerical_features

                ),

                (

                    "cat",

                    OneHotEncoder(

                        handle_unknown="ignore"

                    ),

                    categorical_features

                )

            ]

        )

        X_train = self.preprocessor.fit_transform(

            X_train

        )

        X_test = self.preprocessor.transform(

            X_test

        )

        return X_train, X_test

    def save_artifacts(self):

        joblib.dump(

            self.preprocessor,

            PREPROCESSOR_PATH

        )

        joblib.dump(

            self.label_encoder,

            LABEL_ENCODER_PATH

        )

    def run(self):

        df = self.load_data()

        df = self.clean_data(df)

        df = self.feature_engineering(df)

        X_train, X_test, y_train, y_test = self.split_data(

            df

        )

        X_train, X_test = self.preprocess_data(

            X_train,

            X_test

        )

        self.save_artifacts()

        return (

            X_train,

            X_test,

            y_train,

            y_test,

            self.label_encoder

        )