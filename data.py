import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


def get_processed_data():
    logger.info("Loading data from Excel...")
    df = pd.read_excel("D:\ALL Projects\Segmentation\Online Retail.xlsx")
    pd.set_option('display.width', None)
    print(df.head(25))
    logger.info("Dataset Loading Successful...")

    print("=========== Basic Functions ==========")
    logger.info("information about data:")
    print(df.info())

    logger.info("Statistical Operations:")
    print(df.describe().round(2))

    logger.info("Columns of Data:")
    print(df.columns)

    logger.info("number of rows & columns:")
    print(df.shape)

    logger.info("Column types:")
    print(df.dtypes)

    logger.info("=========== Data Cleaning ==========")
    #['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate','UnitPrice', 'CustomerID', 'Country']

    # Validation DT
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%Y%m%d', errors='coerce')

    # Remove Cancelled Invoice (C) ------->Identifier Feature not numeric
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df = df[~df['InvoiceNo'].str.startswith('C')]

    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    logger.info("Number of Frequency Rows")
    print(df.duplicated().sum()) # 5226
    df =df.drop_duplicates()
    logger.info("After Removing Duplicates",df.shape)
    print('-------------')

    logger.info("Missing Values")
    print(df.isnull().sum())

    # we found miss value in CustomerID : 132186 from 541909
    df.dropna(subset=['CustomerID'], inplace=True)
    print(df['CustomerID'].isnull().sum())
    # CustomerID is Identifier Feature not numeric
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)

    # Convert Series to DataFrame ,we just have CustomerID contain miss value from 8 Feature
    #missing_data = df.isnull().sum().to_frame()
    #sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    #plt.title('Missing Values Count')
    #plt.show()

    print(df.isnull().sum())

    print(df.describe().round(2))

    print("=========== Preprocessin ==========")

    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['Day'] = df['InvoiceDate'].dt.day
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    df['WeekDay_Name'] = df['InvoiceDate'].dt.weekday

    print(df.head(25))

    df['TotalSum'] = df['Quantity'] * df['UnitPrice']


    # Setting a reference date (Snapshot Date)
    # We assume we are analyzing the data the day after the last existing purchase transaction
    import datetime as dt
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

    # Data aggregation for each client Recency Monetary (The RFM Core)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalSum': 'sum'
    })

    # New table : rfm
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalSum': 'Monetary'
    }, inplace=True)

    print(rfm.head(30))
    print(rfm.columns)
    print("---------------------")
    logger.info("Original Skewness:\n", rfm.skew())

    # we found :
    #  Recency       1.246048
    # Frequency    12.067031
    # Monetary     19.339368
    # Right Skew Value -----------> Log Transformation

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    sns.histplot(rfm['Recency'], kde=True, color='skyblue')
    plt.title(f'Recency Distribution\nSkew: {rfm["Recency"].skew():.2f}')

    plt.subplot(1, 3, 2)
    sns.histplot(rfm['Frequency'], kde=True, color='salmon')
    plt.title(f'Frequency Distribution\nSkew: {rfm["Frequency"].skew():.2f}')

    plt.subplot(1, 3, 3)
    sns.histplot(rfm['Monetary'], kde=True, color='lightgreen')
    plt.title(f'Monetary Distribution\nSkew: {rfm["Monetary"].skew():.2f}')

    plt.tight_layout()
    plt.show()

    rfm_log = np.log1p(rfm)
    logger.info("\nSkewness after Treatment:\n", rfm_log.skew())


    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.histplot(rfm_log['Recency'], kde=True)
    plt.title("Recency After Log")

    plt.subplot(1, 3, 2)
    sns.histplot(rfm_log['Frequency'], kde=True)
    plt.title("Frequency After Log")

    plt.subplot(1, 3, 3)
    sns.histplot(rfm_log['Monetary'], kde=True)
    plt.title("Monetary After Log")

    plt.tight_layout()
    plt.show()

    # Skewness after Treatment:
    #  Recency     -0.379169
    # Frequency    1.208652 (Log Transform)
    # Monetary     0.396599
    print("----------------------")

    print(rfm_log.describe().round(2))


    def check_outliers_iqr(df):
        outlier_results = {}
        for col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

            outlier_results[col] = {
                'Count': len(outliers),
                'Percentage': (len(outliers) / len(df)) * 100
            }
        return outlier_results

    results = check_outliers_iqr(rfm_log)

    for col, val in results.items():
        logger.info(f"Column {col}: {val['Count']} Outliers ({val['Percentage']:.2f}%)")

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=rfm_log, palette="Set3")
    plt.title("Boxplot for RFM Features (After Log Transformation)")
    plt.ylabel("Log Values")
    plt.show()

    # B. Outlier Audit:
    # After Log Transformation, we checked for outliers using the IQR method.
    # Current Status:
    # - Recency: 0% outliers
    # - Frequency & Monetary: < 1.5% outliers (Acceptable range for behavioral data).
    # The Log transformation successfully mitigated the impact of "Whale" customers
    # without the need to delete valuable data.

    return df, rfm, rfm_log

# هذا السطر للتأكد من عمل الكود عند تشغيله منفرداً
if __name__ == "__main__":
    df, rfm, rfm_log = get_processed_data()
