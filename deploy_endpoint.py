

import os
import json
import tarfile

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# ==========================================================
# Configuration
# ==========================================================

BUCKET = "nazhifa-credit-score-uas"
MODEL_ARTIFACTS_DIR = "models"         
MODEL_TAR_NAME = "model.tar.gz"
MODEL_S3_KEY = "credit-score/model.tar.gz"
ENDPOINT_NAME = "credit-score-endpoint"
REGION = "us-east-1"
INSTANCE_TYPE = "ml.m5.large"
FRAMEWORK_VERSION = "1.2-1"


GIT_CONFIG = {
    "repo": "https://github.com/nazhifakirana/CreditClassification.git",
    "branch": "main",
}
ENTRY_POINT = "inference.py" 


def get_lab_role_arn():
    iam = boto3.client("iam")
    return iam.get_role(RoleName="LabRole")["Role"]["Arn"]


def build_model_tarball():
    """Pack models/best_model.pkl, preprocessor.pkl, label_encoder.pkl
    into model.tar.gz with a flat structure, exactly what model_fn()
    in src/inference.py expects to find in model_dir."""

    required_files = ["best_model.pkl", "preprocessor.pkl", "label_encoder.pkl"]

    for f in required_files:
        path = os.path.join(MODEL_ARTIFACTS_DIR, f)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run `python main.py` first to train and "
                f"save the model artifacts."
            )

    with tarfile.open(MODEL_TAR_NAME, "w:gz") as tar:
        for f in required_files:
            tar.add(os.path.join(MODEL_ARTIFACTS_DIR, f), arcname=f)

    print(f"Built {MODEL_TAR_NAME} with: {', '.join(required_files)}")

    return MODEL_TAR_NAME


def upload_model_to_s3(tar_path):
    s3 = boto3.client("s3", region_name=REGION)
    s3.upload_file(tar_path, BUCKET, MODEL_S3_KEY)
    s3_uri = f"s3://{BUCKET}/{MODEL_S3_KEY}"
    print(f"Uploaded model artifact to {s3_uri}")
    return s3_uri


def invoke_test_cases(runtime):
    """Send one test case per Credit_Score class through the deployed
    endpoint, for the screenshots required by the assignment."""

    test_cases = {
        "Expected POOR": {
            "Age": 25, "Occupation": "Mechanic", "Annual_Income": 18000,
            "Monthly_Inhand_Salary": 1300, "Num_Bank_Accounts": 8,
            "Num_Credit_Card": 9, "Interest_Rate": 28, "Num_of_Loan": 6,
            "Type_of_Loan": "Personal Loan, Payday Loan, Auto Loan, Credit-Builder Loan",
            "Delay_from_due_date": 25, "Num_of_Delayed_Payment": 18,
            "Changed_Credit_Limit": -5.0, "Num_Credit_Inquiries": 12,
            "Credit_Mix": "Bad", "Outstanding_Debt": 4200.0,
            "Credit_Utilization_Ratio": 42.0,
            "Credit_History_Age": "1 Years and 2 Months",
            "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 600.0,
            "Amount_invested_monthly": 20.0,
            "Payment_Behaviour": "High_spent_Large_value_payments",
            "Monthly_Balance": 50.0,
        },
        "Expected STANDARD": {
            "Age": 35, "Occupation": "Teacher", "Annual_Income": 45000,
            "Monthly_Inhand_Salary": 3500, "Num_Bank_Accounts": 4,
            "Num_Credit_Card": 4, "Interest_Rate": 14, "Num_of_Loan": 2,
            "Type_of_Loan": "Auto Loan, Student Loan",
            "Delay_from_due_date": 10, "Num_of_Delayed_Payment": 6,
            "Changed_Credit_Limit": 3.0, "Num_Credit_Inquiries": 4,
            "Credit_Mix": "Standard", "Outstanding_Debt": 1200.0,
            "Credit_Utilization_Ratio": 30.0,
            "Credit_History_Age": "8 Years and 4 Months",
            "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 200.0,
            "Amount_invested_monthly": 150.0,
            "Payment_Behaviour": "Low_spent_Medium_value_payments",
            "Monthly_Balance": 350.0,
        },
        "Expected GOOD": {
            "Age": 42, "Occupation": "Lawyer", "Annual_Income": 120000,
            "Monthly_Inhand_Salary": 9000, "Num_Bank_Accounts": 2,
            "Num_Credit_Card": 2, "Interest_Rate": 5, "Num_of_Loan": 1,
            "Type_of_Loan": "Mortgage Loan",
            "Delay_from_due_date": 0, "Num_of_Delayed_Payment": 0,
            "Changed_Credit_Limit": 1.0, "Num_Credit_Inquiries": 1,
            "Credit_Mix": "Good", "Outstanding_Debt": 200.0,
            "Credit_Utilization_Ratio": 18.0,
            "Credit_History_Age": "20 Years and 6 Months",
            "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 50.0,
            "Amount_invested_monthly": 1500.0,
            "Payment_Behaviour": "Low_spent_Small_value_payments",
            "Monthly_Balance": 2500.0,
        },
    }

    for name, case in test_cases.items():
        body = json.dumps({"instances": [case]})

        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=body,
        )

        result = json.loads(response["Body"].read().decode("utf-8"))

        print("=" * 70)
        print(f"Test Case  : {name}")
        print(json.dumps(result, indent=2))
        print("=" * 70)
        print()


# ==========================================================
# Main
# ==========================================================

def main():

    boto3.setup_default_session(region_name=REGION)
    sm_session = sagemaker.Session()
    role_arn = get_lab_role_arn()

    tar_path = build_model_tarball()
    model_s3_uri = upload_model_to_s3(tar_path)

    print("=" * 70)
    print("Credit Score SageMaker Deployment")
    print("=" * 70)
    print(f"\nRole          : {role_arn}")
    print(f"Bucket        : {BUCKET}")
    print(f"Model URI     : {model_s3_uri}")
    print(f"Endpoint Name : {ENDPOINT_NAME}")
    print(f"Region        : {REGION}")
    print(f"Source repo   : {GIT_CONFIG['repo']} ({GIT_CONFIG['branch']})")

    model = SKLearnModel(
        model_data=model_s3_uri,
        role=role_arn,
        entry_point=ENTRY_POINT,
        git_config=GIT_CONFIG,
        framework_version=FRAMEWORK_VERSION,
        sagemaker_session=sm_session,
    )

    print("\nDeploying Endpoint...\n")

    model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
    )

    runtime = boto3.client("sagemaker-runtime", region_name=REGION)

    print("\nRunning test cases against the deployed endpoint...\n")
    invoke_test_cases(runtime)

    print("Deployment Completed Successfully.")
    print(f"Endpoint Name : {ENDPOINT_NAME}")
    print("\nDelete the endpoint after testing to avoid AWS charges:")
    print(f'  boto3.client("sagemaker").delete_endpoint(EndpointName="{ENDPOINT_NAME}")')

    # Uncomment to auto-delete right after this script's test run:
    # sm_session.delete_endpoint(ENDPOINT_NAME)


if __name__ == "__main__":
    main()
