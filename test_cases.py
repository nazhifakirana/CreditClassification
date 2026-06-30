"""
Test cases that are designed to represent each Credit_Score class
(Poor / Standard / Good). Run this after the model has been trained
and the artifacts (model, preprocessor, label encoder) are saved in
`models/`, to verify the deployed inference logic end to end and to
produce the evidence (console output / screenshots) required for
the assignment.

Usage:
    python test_cases.py
"""

from inference import CreditScoreInference

# A test case intended to lean towards a "Poor" credit score profile:
# high debt, many delayed payments, high credit utilization, bad credit mix.
case_poor = {
    "Age": 25,
    "Occupation": "Mechanic",
    "Annual_Income": 18000,
    "Monthly_Inhand_Salary": 1300,
    "Num_Bank_Accounts": 8,
    "Num_Credit_Card": 9,
    "Interest_Rate": 28,
    "Num_of_Loan": 6,
    "Type_of_Loan": "Personal Loan, Payday Loan, Auto Loan, Credit-Builder Loan",
    "Delay_from_due_date": 25,
    "Num_of_Delayed_Payment": 18,
    "Changed_Credit_Limit": -5.0,
    "Num_Credit_Inquiries": 12,
    "Credit_Mix": "Bad",
    "Outstanding_Debt": 4200.0,
    "Credit_Utilization_Ratio": 42.0,
    "Credit_History_Age": "1 Years and 2 Months",
    "Payment_of_Min_Amount": "Yes",
    "Total_EMI_per_month": 600.0,
    "Amount_invested_monthly": 20.0,
    "Payment_Behaviour": "High_spent_Large_value_payments",
    "Monthly_Balance": 50.0,
}

# A test case intended to lean towards a "Standard" credit score profile:
# moderate values across the board.
case_standard = {
    "Age": 35,
    "Occupation": "Teacher",
    "Annual_Income": 45000,
    "Monthly_Inhand_Salary": 3500,
    "Num_Bank_Accounts": 4,
    "Num_Credit_Card": 4,
    "Interest_Rate": 14,
    "Num_of_Loan": 2,
    "Type_of_Loan": "Auto Loan, Student Loan",
    "Delay_from_due_date": 10,
    "Num_of_Delayed_Payment": 6,
    "Changed_Credit_Limit": 3.0,
    "Num_Credit_Inquiries": 4,
    "Credit_Mix": "Standard",
    "Outstanding_Debt": 1200.0,
    "Credit_Utilization_Ratio": 30.0,
    "Credit_History_Age": "8 Years and 4 Months",
    "Payment_of_Min_Amount": "No",
    "Total_EMI_per_month": 200.0,
    "Amount_invested_monthly": 150.0,
    "Payment_Behaviour": "Low_spent_Medium_value_payments",
    "Monthly_Balance": 350.0,
}

# A test case intended to lean towards a "Good" credit score profile:
# high income, low debt, long credit history, good credit mix.
case_good = {
    "Age": 42,
    "Occupation": "Lawyer",
    "Annual_Income": 120000,
    "Monthly_Inhand_Salary": 9000,
    "Num_Bank_Accounts": 2,
    "Num_Credit_Card": 2,
    "Interest_Rate": 5,
    "Num_of_Loan": 1,
    "Type_of_Loan": "Mortgage Loan",
    "Delay_from_due_date": 0,
    "Num_of_Delayed_Payment": 0,
    "Changed_Credit_Limit": 1.0,
    "Num_Credit_Inquiries": 1,
    "Credit_Mix": "Good",
    "Outstanding_Debt": 200.0,
    "Credit_Utilization_Ratio": 18.0,
    "Credit_History_Age": "20 Years and 6 Months",
    "Payment_of_Min_Amount": "No",
    "Total_EMI_per_month": 50.0,
    "Amount_invested_monthly": 1500.0,
    "Payment_Behaviour": "Low_spent_Small_value_payments",
    "Monthly_Balance": 2500.0,
}


def run_test_case(name, raw_input, engine):

    label, probabilities = engine.predict(raw_input)

    print("=" * 70)
    print(f"Test Case  : {name}")
    print(f"Prediction : {label}")
    print("Probabilities:")

    for cls, p in sorted(probabilities.items(), key=lambda x: -x[1]):
        print(f"   {cls:<10s}: {p:.4f}")

    print("=" * 70)
    print()


if __name__ == "__main__":

    engine = CreditScoreInference()

    run_test_case("Expected POOR", case_poor, engine)
    run_test_case("Expected STANDARD", case_standard, engine)
    run_test_case("Expected GOOD", case_good, engine)