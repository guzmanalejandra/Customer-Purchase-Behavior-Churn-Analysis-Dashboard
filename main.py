import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Behavior & Churn Analysis Dashboard", layout="wide")
st.title("Customer Purchase Behavior & Churn Analysis Dashboard")

# -------------------------------------------
# Data Loading Function
# -------------------------------------------
def load_data():
    file_path = "customer_data.csv"
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        st.success("Loaded data from customer_data.csv")
    else:
        st.warning("File customer_data.csv not found. Using sample data.")
        # Create a sample DataFrame with two sample rows.
        sample_data = {
            "Customer ID": ["0002-ORFBO", "0003-XYZABC", "0004-TEST01", "0005-TEST02"],
            "Gender": ["Female", "Male", "Female", "Male"],
            "Age": [37, 45, 29, 52],
            "Married": ["Yes", "No", "Yes", "No"],
            "Number of Dependents": [0, 1, 2, 0],
            "City": ["Frazier Park", "SomeCity", "OtherCity", "AnotherCity"],
            "Zip Code": [93225, 12345, 54321, 67890],
            "Latitude": [34.827662, 40.12345, 35.00000, 41.00000],
            "Longitude": [-118.999073, -75.12345, -117.00000, -73.00000],
            "Number of Referrals": [2, 1, 3, 2],
            "Tenure in Months": [9, 12, 3, 18],
            "Offer": [None, None, None, None],
            "Phone Service": ["Yes", "Yes", "Yes", "No"],
            "Avg Monthly Long Distance Charges": [42.39, 35.00, 30.00, 25.00],
            "Multiple Lines": ["No", "No", "Yes", "No"],
            "Internet Service": ["Yes", "Yes", "Yes", "Yes"],
            "Internet Type": ["Cable", "Fiber", "Cable", "Fiber"],
            "Avg Monthly GB Download": [16, 20, 15, 22],
            "Online Security": ["No", "Yes", "Yes", "No"],
            "Online Backup": ["Yes", "No", "Yes", "Yes"],
            "Device Protection Plan": ["No", "Yes", "No", "No"],
            "Premium Tech Support": ["Yes", "No", "Yes", "No"],
            "Streaming TV": ["Yes", "No", "Yes", "No"],
            "Streaming Movies": ["No", "Yes", "No", "Yes"],
            "Streaming Music": ["No", "Yes", "No", "Yes"],
            "Unlimited Data": ["Yes", "Yes", "Yes", "No"],
            "Contract": ["One Year", "Two Year", "Month-to-Month", "Two Year"],
            "Paperless Billing": ["Yes", "No", "Yes", "Yes"],
            "Payment Method": ["Credit Card", "Debit Card", "Credit Card", "Bank Transfer"],
            "Monthly Charge": [65.6, 70.0, 55.0, 80.0],
            "Total Charges": [593.3, 840.0, 165.0, 1440.0],
            "Total Refunds": [0, 0, 0, 0],
            "Total Extra Data Charges": [0, 0, 0, 0],
            "Total Long Distance Charges": [381.51, 600.0, 200.0, 700.0],
            "Total Revenue": [974.81, 1440.0, 220.0, 2140.0],
            "Customer Status": ["Stayed", "Churned", "Churned", "Churned"],
            "Churn Category": ["", "Voluntary", "Involuntary", "Voluntary"],
            "Churn Reason": ["", "Found a better offer", "Failed credit card payment", "Not satisfied with service"]
        }
        data = pd.DataFrame(sample_data)
    return data

# Load the dataset
data = load_data()

# Display a preview of the dataset
st.header("Dataset Preview")
st.dataframe(data.head())

# -------------------------------------------
# Data Processing & Feature Engineering
# -------------------------------------------
# Calculate Historical Average Monthly Revenue using Total Revenue and Tenure in Months.
# If Tenure is 0 or missing, use the Monthly Charge as a fallback.
data['Historical_Avg'] = data['Total Revenue'] / data['Tenure in Months']
data['Historical_Avg'] = data['Historical_Avg'].replace([np.inf, -np.inf], np.nan)
data['Historical_Avg'] = data['Historical_Avg'].fillna(data['Monthly Charge'])


# Calculate the ratio of the current Monthly Charge to the Historical Average.
data['Charge_Ratio'] = data['Monthly Charge'] / data['Historical_Avg']

st.header("Analysis Metrics")
st.write("Summary statistics for Monthly Charge and Historical Average:")
st.write(data[['Monthly Charge', 'Historical_Avg']].describe())

# -------------------------------------------
# App Settings & Alerts for Purchase Behavior
# -------------------------------------------
st.sidebar.header("Alert Settings")
threshold = st.sidebar.slider(
    "Select drop ratio threshold (Monthly Charge / Historical Avg)", 
    min_value=0.0, max_value=1.0, value=0.8, step=0.05
)
data['At_Risk'] = data['Charge_Ratio'] < threshold

st.subheader("At Risk Customers")
at_risk_customers = data[data['At_Risk']]
st.dataframe(at_risk_customers[['Customer ID', 'Monthly Charge', 'Historical_Avg', 'Charge_Ratio']])

# -------------------------------------------
# Visualizations for Purchase Behavior
# -------------------------------------------
st.subheader("Distribution of Charge Ratio")
fig, ax = plt.subplots()
ax.hist(data['Charge_Ratio'], bins=20, edgecolor='black')
ax.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f"Threshold: {threshold}")
ax.set_xlabel("Monthly Charge / Historical Average")
ax.set_ylabel("Frequency")
ax.set_title("Charge Ratio Distribution")
ax.legend()
st.pyplot(fig)

st.subheader("Monthly Charge vs Historical Average")
fig2, ax2 = plt.subplots()
colors = np.where(data['At_Risk'], 'red', 'blue')
ax2.scatter(data['Historical_Avg'], data['Monthly Charge'], c=colors, alpha=0.6)
min_val = min(data['Historical_Avg'].min(), data['Monthly Charge'].min())
max_val = max(data['Historical_Avg'].max(), data['Monthly Charge'].max())
ax2.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
ax2.set_xlabel("Historical Average")
ax2.set_ylabel("Monthly Charge")
ax2.set_title("Monthly Charge vs Historical Average")
st.pyplot(fig2)

st.write("Note: Customers shown in red are flagged as at risk based on the current threshold.")

# -------------------------------------------
# Churn Analysis Section
# -------------------------------------------
st.header("Churn Analysis")
show_churn = st.sidebar.checkbox("Show Churn Analysis")

if show_churn:
    if "Customer Status" in data.columns:
        # Filter churned customers (assuming churned status is labeled 'Churned', case-insensitive)
        churned = data[data['Customer Status'].str.lower() == "churned"]
        total_customers = data.shape[0]
        churn_count = churned.shape[0]
        churn_rate = (churn_count / total_customers) * 100 if total_customers > 0 else 0

        st.subheader("Overall Churn Metrics")
        st.write(f"Total Customers: {total_customers}")
        st.write(f"Churned Customers: {churn_count}")
        st.write(f"Churn Rate: {churn_rate:.2f}%")
        
        if churn_count > 0:
            # Analyze churn by tenure (e.g., early drop-offs)
            st.subheader("Churn by Tenure")
            fig3, ax3 = plt.subplots()
            ax3.hist(churned['Tenure in Months'], bins=10, edgecolor='black')
            ax3.set_xlabel("Tenure in Months")
            ax3.set_ylabel("Number of Churned Customers")
            ax3.set_title("Distribution of Tenure among Churned Customers")
            st.pyplot(fig3)
            
            # Calculate percentage of churned customers with tenure less than a given threshold (e.g., 1 month)
            tenure_threshold = 1  # in months
            early_churn = churned[churned['Tenure in Months'] < tenure_threshold]
            early_churn_pct = (early_churn.shape[0] / churn_count) * 100
            st.write(f"{early_churn.shape[0]} out of {churn_count} churned customers ({early_churn_pct:.2f}%) churned before {tenure_threshold} month(s).")
            
            # -------------------------
            # Churn Reasons and Types Analysis
            # -------------------------
            if "Churn Reason" in churned.columns:
                st.subheader("Churn Reasons")
                # Display count of each churn reason
                churn_reason_counts = churned['Churn Reason'].value_counts()
                st.write("Churn Reason Counts:")
                st.write(churn_reason_counts)
                
                # Plot a bar chart of churn reasons
                st.subheader("Most Common Churn Reasons")
                fig4, ax4 = plt.subplots()
                churn_reason_counts.plot(kind='bar', ax=ax4)
                ax4.set_xlabel("Churn Reason")
                ax4.set_ylabel("Count")
                ax4.set_title("Most Given Reasons for Churn")
                st.pyplot(fig4)
                
                # Categorize churn as voluntary or involuntary using basic keyword matching.
                def categorize_churn(reason):
                    if pd.isna(reason) or reason.strip() == "":
                        return "Unknown"
                    reason_lower = reason.lower()
                    if "failed" in reason_lower or "payment" in reason_lower or "credit" in reason_lower:
                        return "Involuntary"
                    return "Voluntary"
                
                churned = churned.copy()  # Create an explicit copy to avoid the warning
                churned['Churn Type'] = churned['Churn Reason'].apply(categorize_churn)
                st.write("Churn Type Counts:")
                st.write(churned['Churn Type'].value_counts())
                
                # -------------------------
                # Additional Graphs for Churn Analysis
                # -------------------------
                # 1. Pie chart for churn type breakdown.
                st.subheader("Churn Type Breakdown")
                churn_type_counts = churned['Churn Type'].value_counts()
                fig5, ax5 = plt.subplots()
                ax5.pie(churn_type_counts, labels=churn_type_counts.index, autopct='%1.1f%%', startangle=90)
                ax5.set_title("Churn Type Distribution")
                st.pyplot(fig5)
                
                # 2. Histogram of churned customers' ages.
                if "Age" in churned.columns:
                    st.subheader("Churned Customers by Age")
                    fig6, ax6 = plt.subplots()
                    ax6.hist(churned['Age'], bins=10, edgecolor='black')
                    ax6.set_xlabel("Age")
                    ax6.set_ylabel("Count")
                    ax6.set_title("Age Distribution of Churned Customers")
                    st.pyplot(fig6)
                
                # 3. Bar chart for churn counts by gender.
                if "Gender" in churned.columns:
                    st.subheader("Churn by Gender")
                    gender_counts = churned['Gender'].value_counts()
                    fig7, ax7 = plt.subplots()
                    gender_counts.plot(kind='bar', ax=ax7)
                    ax7.set_xlabel("Gender")
                    ax7.set_ylabel("Count")
                    ax7.set_title("Churned Customers by Gender")
                    st.pyplot(fig7)
            else:
                st.info("No 'Churn Reason' column found to analyze specific churn complaints.")
            
            # Note on seasonality: if you have a 'Churn Month' or date column, you can extend the analysis.
            if "Churn Month" in churned.columns:
                st.subheader("Churn Seasonality")
                fig8, ax8 = plt.subplots()
                churn_by_month = churned['Churn Month'].value_counts().sort_index()
                churn_by_month.plot(kind='bar', ax=ax8)
                ax8.set_xlabel("Month")
                ax8.set_ylabel("Churn Count")
                ax8.set_title("Churn Counts by Month")
                st.pyplot(fig8)
            else:
                st.info("Seasonality analysis requires a 'Churn Month' column. Add this column for further insights.")
        else:
            st.info("No churned customers found in the dataset.")
    else:
        st.error("The dataset does not contain a 'Customer Status' column for churn analysis.")


# -------------------------------------------
# Option to Show Full Processed Dataset
# -------------------------------------------
if st.checkbox("Show full processed dataset"):
    st.dataframe(data)
