import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class Evaluator:
    """Evaluates a single trained model on the held-out test set."""

    def __init__(self, model, model_name):

        self.model = model

        self.model_name = model_name

    def predict(self, X_test):

        if self.model_name == "Naive Bayes":

            return self.model.predict(
                X_test.toarray()
            )

        return self.model.predict(
            X_test
        )

    def evaluate(self, y_test, prediction):

        metrics = {

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

        return pd.DataFrame([metrics])

    def get_classification_report(
        self,
        y_test,
        prediction
    ):

        return classification_report(
            y_test,
            prediction,
            output_dict=True
        )

    def print_classification_report(
        self,
        y_test,
        prediction
    ):

        print(

            classification_report(

                y_test,

                prediction

            )

        )

    def plot_confusion_matrix(
        self,
        y_test,
        prediction,
        save_path="models/best_model_confusion_matrix.png"
    ):

        cm = confusion_matrix(

            y_test,

            prediction

        )

        plt.figure(figsize=(6,5))

        sns.heatmap(

            cm,

            annot=True,

            fmt="d",

            cmap="Blues"

        )

        plt.title(

            f"Confusion Matrix - {self.model_name}"

        )

        plt.xlabel(

            "Predicted"

        )

        plt.ylabel(

            "Actual"

        )

        plt.tight_layout()

        plt.savefig(save_path, bbox_inches="tight")

        plt.close()

    def evaluate_all(
        self,
        X_test,
        y_test
    ):

        prediction = self.predict(

            X_test

        )

        metric = self.evaluate(

            y_test,

            prediction

        )

        self.print_classification_report(

            y_test,

            prediction

        )

        self.plot_confusion_matrix(

            y_test,

            prediction

        )

        return metric