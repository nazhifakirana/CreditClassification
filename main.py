from preprocessing import Preprocessing
from trainer import Trainer
from evaluator import Evaluator


class CreditScorePipeline:

    def __init__(self):
        self.preprocessing = Preprocessing()
        self.trainer = Trainer()

    def run(self):
        print("=" * 70)
        print("Credit Score Classification Training Pipeline")
        print("=" * 70)

        print("\nStep 1 : Data Preprocessing")
        (
            X_train,
            X_test,
            y_train,
            y_test,
            _
        ) = self.preprocessing.run()
        print("Preprocessing Completed")

        print("\nStep 2 : Model Training")
        results = self.trainer.train_models(
            X_train,
            y_train,
            X_test,
            y_test
        )
        print(results)

        self.trainer.save_best_model()
        print("\nTraining Completed")

        best_model = self.trainer.get_best_model()
        best_model_name = self.trainer.get_best_model_name()
        print(f"\nBest Model : {best_model_name}")

        print("\nStep 3 : Model Evaluation")
        evaluator = Evaluator(
            best_model,
            best_model_name
        )
        evaluation = evaluator.evaluate_all(
            X_test,
            y_test
        )
        print("\nEvaluation Metric")
        print(evaluation)

        print("\nPipeline Finished Successfully")
        print("All per-model artifacts (metrics, params, confusion matrix, "
              "classification report) are saved under `artifacts/<model_name>/`.")


if __name__ == "__main__":
    pipeline = CreditScorePipeline()
    pipeline.run()