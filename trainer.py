import os
import time
import joblib
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import issparse

from config import *

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class Trainer:
    """
    Trains a panel of candidate classifiers and keeps the model with
    the best weighted F1 score as the production candidate.

    NOTE: MLflow tracking has been removed on purpose. MLflow needs a
    tracking server / local `mlruns` folder that is not reliably
    available inside a SageMaker Studio / Training Job container, so
    every run is logged with plain prints + a results DataFrame and
    artifacts (classification report, confusion matrix image) saved
    directly to disk under `artifacts/<model_name>/` instead.
    """

    def __init__(self, artifacts_dir="artifacts"):

        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)

        self.models = {

            "Logistic Regression": LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000
            ),

            "Decision Tree": DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                max_depth=10
            ),

            "Random Forest": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_estimators=200,
                max_depth=15
            ),

            "Gradient Boosting": GradientBoostingClassifier(
                random_state=RANDOM_STATE,
                learning_rate=0.05,
                n_estimators=200
            ),

            "KNN": KNeighborsClassifier(
                n_neighbors=5
            ),

            "Naive Bayes": GaussianNB(),

            "SVM": SVC(
                probability=True,
                random_state=RANDOM_STATE
            )

        }

        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.results = []

    def _to_dense(self, X):
        """Convert to dense array only if X is actually sparse."""
        return X.toarray() if issparse(X) else X

    def fit_model(self, model, name, X_train, y_train):

        if name == "Naive Bayes":
            X_train = self._to_dense(X_train)

        model.fit(X_train, y_train)

    def predict(self, model, name, X_test):

        if name == "Naive Bayes":
            X_test = self._to_dense(X_test)

        return model.predict(X_test)

    def calculate_metrics(self, y_test, prediction):

        return {

            "Accuracy": accuracy_score(
                y_test,
                prediction
            ),

            "Precision": precision_score(
                y_test,
                prediction,
                average="weighted"
            ),

            "Recall": recall_score(
                y_test,
                prediction,
                average="weighted"
            ),

            "F1 Score": f1_score(
                y_test,
                prediction,
                average="weighted"
            )

        }

    def cross_validate(self, model, name, X_train, y_train):
        """5-fold stratified CV F1 score, used to justify model choice
        beyond a single train/test split and to flag overfitting."""

        cv = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        X = self._to_dense(X_train) if name == "Naive Bayes" else X_train

        scores = cross_val_score(
            model,
            X,
            y_train,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1
        )

        return scores.mean(), scores.std()

    def log_artifacts(
        self,
        name,
        model,
        metrics,
        y_test,
        prediction,
        cv_mean,
        cv_std
    ):
        """Persist per-model evidence to disk instead of MLflow."""

        model_dir = os.path.join(
            self.artifacts_dir,
            name.replace(" ", "_")
        )
        os.makedirs(model_dir, exist_ok=True)

        # classification report
        report = classification_report(y_test, prediction)
        with open(os.path.join(model_dir, "classification_report.txt"), "w") as f:
            f.write(report)

        # confusion matrix
        cm = confusion_matrix(y_test, prediction)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(name)
        plt.savefig(
            os.path.join(model_dir, "confusion_matrix.png"),
            bbox_inches="tight"
        )
        plt.close()

        # metrics + params as plain text/csv
        params_df = pd.DataFrame(
            list(model.get_params().items()),
            columns=["param", "value"]
        )
        params_df.to_csv(os.path.join(model_dir, "params.csv"), index=False)

        metrics_full = {
            **metrics,
            "CV_F1_Mean": cv_mean,
            "CV_F1_Std": cv_std
        }
        pd.DataFrame([metrics_full]).to_csv(
            os.path.join(model_dir, "metrics.csv"), index=False
        )

        # model artifact itself, useful for inspection even though the
        # single "best" model is the one actually deployed
        joblib.dump(model, os.path.join(model_dir, "model.pkl"))

    def update_best_model(
        self,
        model,
        name,
        score
    ):

        if score > self.best_score:

            self.best_score = score

            self.best_model = model

            self.best_model_name = name

    def train_models(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ):

        self.results = []

        print(f"\nTraining {len(self.models)} candidate models, please wait...\n")

        for name, model in self.models.items():

            print(f"  -> Training {name} ...", end=" ", flush=True)

            start = time.time()

            self.fit_model(
                model,
                name,
                X_train,
                y_train
            )

            prediction = self.predict(
                model,
                name,
                X_test
            )

            metrics = self.calculate_metrics(
                y_test,
                prediction
            )

            cv_mean, cv_std = self.cross_validate(
                model,
                name,
                X_train,
                y_train
            )

            self.log_artifacts(
                name,
                model,
                metrics,
                y_test,
                prediction,
                cv_mean,
                cv_std
            )

            self.results.append({

                "Model": name,

                **metrics,

                "CV_F1_Mean": cv_mean,

                "CV_F1_Std": cv_std

            })

            self.update_best_model(

                model,

                name,

                metrics["F1 Score"]

            )

            elapsed = time.time() - start

            print(f"done in {elapsed:.1f}s (F1={metrics['F1 Score']:.4f})")

        print(f"\nBest Model : {self.best_model_name} (F1={self.best_score:.4f})")

        self.results = pd.DataFrame(

            self.results

        ).sort_values(

            by="F1 Score",

            ascending=False

        )

        self.results.to_csv(
            os.path.join(self.artifacts_dir, "all_models_summary.csv"),
            index=False
        )

        return self.results

    def save_best_model(self):

        os.makedirs(
            os.path.dirname(MODEL_PATH),
            exist_ok=True
        )

        joblib.dump(

            self.best_model,

            MODEL_PATH

        )

    def get_best_model(self):

        return self.best_model

    def get_best_model_name(self):

        return self.best_model_name