import streamlit as st
import pandas as pd
import plotly.express as px
import boto3
import json

st.set_page_config(
    page_title="Credit Score Classification",
    page_icon="💳",
    layout="centered"
)

# ================================
# SageMaker Endpoint Configuration
# ================================
ENDPOINT_NAME = "credit-score-endpoint"
REGION = "us-east-1"

runtime = boto3.client(
    "sagemaker-runtime",
    region_name=REGION
)

st.title("💳 Credit Score Classification")

st.caption(
    "Masukkan data nasabah untuk memprediksi kategori credit score "
    "(Poor / Standard / Good) menggunakan model yang telah dideploy di AWS SageMaker."
)

with st.form("credit_score_form"):

    st.subheader("Data Demografi & Pekerjaan")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)

    with col2:
        occupation = st.text_input("Occupation", value="Engineer")

    st.subheader("Pendapatan & Tabungan")

    col1, col2 = st.columns(2)

    with col1:
        annual_income = st.number_input(
            "Annual Income",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
        )

        monthly_inhand_salary = st.number_input(
            "Monthly Inhand Salary",
            min_value=0.0,
            value=4000.0,
            step=100.0,
        )

    with col2:
        monthly_balance = st.number_input(
            "Monthly Balance",
            min_value=0.0,
            value=300.0,
            step=10.0,
        )

        amount_invested_monthly = st.number_input(
            "Amount Invested Monthly",
            min_value=0.0,
            value=100.0,
            step=10.0,
        )

    st.subheader("Rekening & Kartu")

    col1, col2, col3 = st.columns(3)

    with col1:
        num_bank_accounts = st.number_input(
            "Num Bank Accounts",
            min_value=0,
            value=3
        )

    with col2:
        num_credit_card = st.number_input(
            "Num Credit Card",
            min_value=0,
            value=2
        )

    with col3:
        interest_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.0,
            value=10.0
        )

    st.subheader("Pinjaman & Keterlambatan")

    col1, col2 = st.columns(2)

    with col1:

        num_of_loan = st.number_input(
            "Num of Loan",
            min_value=0,
            value=1
        )

        type_of_loan = st.text_input(
            "Type of Loan",
            value="Personal Loan, Auto Loan"
        )

        delay_from_due_date = st.number_input(
            "Delay from Due Date",
            min_value=0,
            value=5
        )

    with col2:

        num_of_delayed_payment = st.number_input(
            "Num of Delayed Payment",
            min_value=0,
            value=2
        )

        changed_credit_limit = st.number_input(
            "Changed Credit Limit",
            value=5.0
        )

        num_credit_inquiries = st.number_input(
            "Num Credit Inquiries",
            min_value=0,
            value=2
        )

    st.subheader("Profil Kredit")

    col1, col2 = st.columns(2)

    with col1:

        credit_mix = st.selectbox(
            "Credit Mix",
            ["Good", "Standard", "Bad"]
        )

        outstanding_debt = st.number_input(
            "Outstanding Debt",
            min_value=0.0,
            value=1000.0
        )

        credit_utilization_ratio = st.number_input(
            "Credit Utilization Ratio",
            min_value=0.0,
            value=30.0
        )

    with col2:

        credit_history_age = st.text_input(
            "Credit History Age",
            value="5 Years and 3 Months"
        )

        payment_of_min_amount = st.selectbox(
            "Payment of Min Amount",
            ["Yes", "No"]
        )

        payment_behaviour = st.selectbox(
            "Payment Behaviour",
            [
                "Low_spent_Small_value_payments",
                "Low_spent_Medium_value_payments",
                "Low_spent_Large_value_payments",
                "High_spent_Small_value_payments",
                "High_spent_Medium_value_payments",
                "High_spent_Large_value_payments",
            ],
        )

    total_emi_per_month = st.number_input(
        "Total EMI per Month",
        min_value=0.0,
        value=150.0,
        step=10.0,
    )

    submitted = st.form_submit_button("Predict Credit Score")

if submitted:

    raw_input = {
        "Age": age,
        "Occupation": occupation,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_inhand_salary,
        "Num_Bank_Accounts": num_bank_accounts,
        "Num_Credit_Card": num_credit_card,
        "Interest_Rate": interest_rate,
        "Num_of_Loan": num_of_loan,
        "Type_of_Loan": type_of_loan,
        "Delay_from_due_date": delay_from_due_date,
        "Num_of_Delayed_Payment": num_of_delayed_payment,
        "Changed_Credit_Limit": changed_credit_limit,
        "Num_Credit_Inquiries": num_credit_inquiries,
        "Credit_Mix": credit_mix,
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": credit_utilization_ratio,
        "Credit_History_Age": credit_history_age,
        "Payment_of_Min_Amount": payment_of_min_amount,
        "Total_EMI_per_month": total_emi_per_month,
        "Amount_invested_monthly": amount_invested_monthly,
        "Payment_Behaviour": payment_behaviour,
        "Monthly_Balance": monthly_balance,
    }

    with st.spinner("Predicting..."):

        body = json.dumps({
            "instances": [raw_input]
        })

        try:

            response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType="application/json",
                Accept="application/json",
                Body=body,
            )

            result = json.loads(
                response["Body"].read().decode("utf-8")
            )

            prediction = result["predictions"][0]

            label = prediction["prediction"]
            probabilities = prediction["probabilities"]

            st.success(f"### Predicted Credit Score : **{label}**")

            df = pd.DataFrame(
                {
                    "Class": list(probabilities.keys()),
                    "Probability": list(probabilities.values()),
                }
            )

            fig = px.bar(
                df,
                x="Class",
                y="Probability",
                color="Class",
                text_auto=".2%",
                title="Prediction Probability",
            )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Raw Prediction"):
                st.json(result)

        except Exception as e:
            st.error(f"Error saat memanggil SageMaker Endpoint:\n\n{e}")
