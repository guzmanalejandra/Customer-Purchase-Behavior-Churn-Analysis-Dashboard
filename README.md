# Customer Purchase Behavior & Churn Analysis Dashboard

This project is an interactive dashboard built using [Streamlit](https://streamlit.io/) that analyzes customer purchase behavior and churn patterns. The dashboard helps you monitor customer spending, flag at-risk customers, and understand why customers churn using various visualizations and metrics.

## Features

- **Data Loading & Preview:**
  - Loads customer data from a CSV file (`customer_data.csv`).
  - Falls back to a sample dataset if the CSV file is not found.
  - Displays a preview of the loaded dataset.

- **Purchase Behavior Analysis:**
  - Calculates each customer's historical average monthly revenue.
  - Computes a ratio comparing the current monthly charge to the historical average.
  - Flags customers as "at risk" based on a user-adjustable threshold.
  - Visualizes the distribution of charge ratios and compares monthly charges with historical averages.

- **Churn Analysis:**
  - Segments churned customers and calculates overall churn metrics.
  - Analyzes churn by customer tenure.
  - Visualizes the most common churn reasons using bar charts.
  - Categorizes churn as voluntary or involuntary using keyword matching.
  - Additional graphs:
    - **Pie Chart:** Breakdown of churn types.
    - **Histogram:** Age distribution of churned customers.
    - **Bar Chart:** Churn counts by gender.
  
- **Interactive Controls:**
  - Sidebar options to adjust thresholds and toggle churn analysis graphs.
  - Options to display the full processed dataset.

## Requirements

- Python 3.7+
- [Streamlit](https://streamlit.io/)
- Pandas
- Matplotlib
- Numpy

## Installation

**Running the App**

   ```bash
   streamlit run main.py

