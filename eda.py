import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

def run_eda(df, rfm, rfm_log):

    logger.info("=========== EDA & Visualization ==========")

    logger.info("1. Correlation Matrix between RFM Features")
    correlation = rfm_log.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(8, 5))
    sns.heatmap(correlation, annot=True, cmap='RdYlGn', center=0)
    plt.title("Correlation Heatmap: Recency vs Frequency vs Monetary")
    plt.show()

    logger.info("2. Top 10 Customers by Monetary Value")
    top_10_customers = rfm.nlargest(10, 'Monetary')
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_10_customers.index.astype(str), y=top_10_customers['Monetary'], palette='magma')
    plt.title("Revenue Contribution of Top 10 Customers")
    plt.xlabel("Customer ID")
    plt.show()

    logger.info("3. Statistical Summary of Customer Recency")
    plt.figure(figsize=(10, 6))
    sns.histplot(rfm['Recency'], bins=40, kde=True, color='teal')
    plt.axvline(rfm['Recency'].median(), color='red', linestyle='--')
    plt.title("Distribution of Days Since Last Purchase")
    plt.show()

    logger.info("4. Pairplot: Visualizing Natural Groups")
    sns.pairplot(rfm_log, diag_kind='kde', plot_kws={'alpha': 0.4})
    plt.show()

    df_temp = df.reset_index() if 'CustomerID' not in df.columns else df
    df_unique_customers = df_temp.drop_duplicates(subset=['CustomerID'])[['CustomerID', 'Country']]
    rfm_context = rfm_log.reset_index().merge(df_unique_customers, on='CustomerID', how='left')

    logger.info("--- 5. Most Purchased Products by Top Spenders")
    top_customer_ids = rfm.nlargest(400, 'Monetary').index  # كبار العملاء
    top_products = df[df['CustomerID'].isin(top_customer_ids)]['Description'].value_counts().head(10)

    plt.figure(figsize=(10, 5))
    top_products.plot(kind='barh', color='gold')
    plt.title("Top 10 Products for High-Value Customers")
    plt.show()


    country_analysis = rfm_context.groupby('Country')[['Recency', 'Frequency', 'Monetary']].mean().sort_values(
        by='Monetary', ascending=False).head(10)
    logger.info(f"--- 6. Average RFM Metrics per Top Countries ---\n{country_analysis.head().to_string()}")

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Country', y='Monetary', data=rfm_context[rfm_context['Country'].isin(country_analysis.index)])
    plt.xticks(rotation=45)
    plt.title("Monetary Distribution (Log) by Top Countries")
    plt.show()

    logger.info("--- 7. Monthly Activity Pattern ---")
    monthly_activity = df.groupby('Month')['InvoiceNo'].nunique()

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=monthly_activity.index, y=monthly_activity.values, marker='o', color='red')
    plt.title("Total Transactions (Unique Invoices) per Month")
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 13))
    plt.show()

    logger.info("--- 8. Daily Sales Frequency ---")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df.drop_duplicates('InvoiceNo'), x='DayOfWeek', order=day_order, palette='coolwarm')
    plt.title("Traffic Distribution by Day of Week (Unique Invoices)")
    plt.show()