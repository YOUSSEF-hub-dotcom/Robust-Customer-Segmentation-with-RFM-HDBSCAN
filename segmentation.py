import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("D:\ALL Projects\Segmentation\Online Retail.xlsx")
pd.set_option('display.width', None)
print(df.head(25))

print("=========== Basic Functions ==========")
print("information about data:")
print(df.info())

print("Statistical Operations:")
print(df.describe().round(2))

print("Columns:")
print(df.columns)

print("number of rows & columns:")
print(df.shape)

print("Column types:")
print(df.dtypes)

print("=========== Data Cleaning ==========")
#['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate','UnitPrice', 'CustomerID', 'Country']

# Validation DT
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%Y%m%d', errors='coerce')

# Remove Cancelled Invoice (C) ------->Identifier Feature not numeric
df['InvoiceNo'] = df['InvoiceNo'].astype(str)
df = df[~df['InvoiceNo'].str.startswith('C')]

df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

print("Number of Frequency Rows")
print(df.duplicated().sum()) # 5226
df =df.drop_duplicates()
print("After Removing Duplicates",df.shape)
print('-------------')

print("Missing Values")
print(df.isnull().sum())

# we found miss value in CustomerID : 132186 from 5429090
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
print("Original Skewness:\n", rfm.skew())

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
print("\nSkewness after Treatment:\n", rfm_log.skew())

from scipy import stats

rfm_log['Frequency'], _ = stats.boxcox(rfm['Frequency'])

print("Frequency Skewness (Box-Cox):", rfm_log['Frequency'].skew())

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
# Frequency    1.208652 (Log Transform) -------> Box-Cox : 0.14
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
    print(f"Column {col}: {val['Count']} Outliers ({val['Percentage']:.2f}%)")

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

print("=========== EDA & Visualization ==========")
print("1. Correlation Matrix between RFM Features (Numerical Insight)")
correlation = rfm_log.corr()
print(correlation)

plt.figure(figsize=(8, 5))
sns.heatmap(correlation, annot=True, cmap='RdYlGn', center=0)
plt.title("Correlation Heatmap: Recency vs Frequency vs Monetary")
plt.show()

print("----------------")

print("2. Top 10 Customers by Monetary Value (The Whales VIP)")
top_10_customers = rfm.nlargest(10, 'Monetary')
print(top_10_customers)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_customers.index.astype(str), y=top_10_customers['Monetary'], palette='magma')
plt.title("Revenue Contribution of Top 10 Customers")
plt.xlabel("Customer ID")
plt.ylabel("Total Money Spent (Monetary)")
plt.show()
print("----------------")

print("3. Statistical Summary of Customer Recency (Days since last purchase)")

recency_stats = rfm['Recency'].describe()
print(recency_stats)
print(f"\nMedian Recency: {rfm['Recency'].median()} days")

plt.figure(figsize=(10, 6))
sns.histplot(rfm['Recency'], bins=40, kde=True, color='teal')
plt.axvline(rfm['Recency'].median(), color='red', linestyle='--', label='Median')
plt.title("Distribution of Days Since Last Purchase")
plt.xlabel("Days")
plt.legend()
plt.show()

print("----------------")

print("4. Pairplot: Visualizing Natural Groups in Data (Using Log Values for clarity)")
print("Visualizing the relationship between features to spot natural clusters...")

sns.pairplot(rfm_log, diag_kind='kde', plot_kws={'alpha': 0.4})
plt.suptitle("Pairwise Relationships in RFM Data", y=1.02)
plt.show()

print("----------------")

print(rfm.columns)
print(df.columns)

print("----------------")

df_context = df[['CustomerID', 'Description','Country' ,'Year', 'Month', 'Day', 'DayOfWeek','WeekDay_Name']].copy()

rfm_context = rfm_log.reset_index().merge(df_context, on='CustomerID', how='left')

print("Final Integrated Data Summary:")
print(rfm_context.head())

print("----------------")

print("--- 5.Most Purchased Products by Top Spenders")
top_threshold = rfm_context['Monetary'].quantile(0.9)
top_products = rfm_context[rfm_context['Monetary'] >= top_threshold]['Description'].value_counts().head(10)
print(top_products)

plt.figure(figsize=(10, 5))
top_products.plot(kind='barh', color='gold')
plt.title("Top 10 Products for High-Value Customers")
plt.show()

print("--- 6. Average RFM Metrics per Country ---")
country_analysis = rfm_context.groupby('Country')[['Recency', 'Frequency', 'Monetary']].mean().sort_values(by='Monetary', ascending=False).head(10)
print(country_analysis)

plt.figure(figsize=(12, 6))
sns.boxplot(x='Country', y='Monetary', data=rfm_context[rfm_context['Country'].isin(country_analysis.index)])
plt.xticks(rotation=45)
plt.title("Monetary Distribution (Log) by Top Countries")
plt.show()

print("--- 7. Monthly Activity Pattern ---")
monthly_activity = rfm_context.groupby('Month')['CustomerID'].count()
print(monthly_activity)

plt.figure(figsize=(10, 5))
sns.lineplot(x=monthly_activity.index, y=monthly_activity.values, marker='o', color='red')
plt.title("Customer Activity (Number of Transactions) per Month")
plt.xticks(range(1, 13))
plt.grid()
plt.gray()
plt.show()

print("--- 8. Daily Sales Frequency ---")
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.figure(figsize=(10, 5))
sns.countplot(data=rfm_context, x='DayOfWeek', order=day_order, palette='coolwarm')
plt.title("Traffic Distribution by Day of Week")
plt.show()

print("=========== Building A ML Model ==========")

# Feature selection ----> RFM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import hdbscan

# 1. Scaling
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

pca = PCA(n_components=2,random_state=42)
x_pca = pca.fit_transform(rfm_scaled)
print("PCA Explained Variance:", pca.explained_variance_ratio_.sum())

plt.figure(figsize=(10, 5))
plt.scatter(x_pca[:,0],x_pca[:,1],alpha=0.7)
plt.title("PCA 2D Projection")
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()
# We used PCA to compress 3-dimensional (RFM) data into 2-dimensional (2D) data
# The variance ratio (0.93) is a very high "quality indicator," confirming that the relationships between the variables are strong and interconnected
# gain:To ensure there is no random "noise" preventing clustering in the binary space
# ----
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
x_tsne = tsne.fit_transform(rfm_scaled)
plt.figure(figsize=(10, 5))
plt.scatter(x_tsne[:,0],x_tsne[:,1],alpha=0.7)
plt.title("T-SNE 2D Projection")
plt.ylabel("Dim_2")
plt.xlabel("Dim_1")
plt.show()

#Since PCA showed 93% variance but in a "compressed" form, we used t-SNE to detect "local complexities."
#t-SNE successfully detected gaps and subclusters invisible in PCA.
#Result: t-SNE provided "visual evidence" that HDBSCAN is the most suitable, because the discrete clusters that appeared depended on density, not distance from the center.


# 3. Building HDBSCAN with Metric Loop
metrics = ['euclidean', 'manhattan']
results_list = []

for m in metrics:
    min_c = max(15, int(len(rfm_scaled) * 0.005)) # 0.5%

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_c,
        min_samples=5,  # تقليل min_samples يقلل الـ Noise ويساعد في فصل الـ Clusters المتاربة
        metric=m,
        gen_min_span_tree=True,
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(rfm_scaled)

    total_stability = clusterer.cluster_persistence_.sum()

    if len(set(cluster_labels)) > 1:
        mask = cluster_labels != -1
        if np.any(mask):
            s_score = silhouette_score(rfm_scaled[mask], cluster_labels[mask])
            dbcv = clusterer.relative_validity_

            results_list.append({
                'Metric': m,
                'Clusters': len(set(cluster_labels)) - 1 ,# -1 to remove noise
                'Noise_Pct': (cluster_labels == -1).sum() / len(cluster_labels) * 100,
                'Stability': total_stability,
                'DBCV': dbcv,
                'Silhouette': s_score
            })

results_df = pd.DataFrame(results_list)
print(results_df)


final_clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean',prediction_data=True)
rfm['Cluster'] = final_clusterer.fit_predict(rfm_scaled)

rfm['Cluster_Probability'] = final_clusterer.probabilities_


cluster_profile = rfm.groupby('Cluster').agg(
    Avg_Recency=('Recency', 'mean'),
    Avg_Frequency=('Frequency', 'mean'),
    Avg_Monetary=('Monetary', 'mean'),
    Customer_Count=('Cluster', 'count')
)

cluster_profile = cluster_profile.sort_values('Avg_Monetary', ascending=False)

print(cluster_profile)


rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'], index=rfm.index)
rfm_scaled_df['Cluster'] = rfm['Cluster']

rfm_melted = pd.melt(rfm_scaled_df.reset_index(),
                    id_vars=['Cluster'],
                    value_vars=['Recency', 'Frequency', 'Monetary'],
                    var_name='Attribute',
                    value_name='Value')

plt.figure(figsize=(12, 6))
plt.title('Snake Plot of RFM Segments')
sns.lineplot(x='Attribute', y='Value', hue='Cluster', data=rfm_melted, palette='viridis', marker='o')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

print("========== MLflow LifeCycle =========")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow.pyfunc
import hdbscan

from sklearn.metrics import silhouette_score
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

class RFMClusterWrapper(mlflow.pyfunc.PythonModel):
    """
    Production-ready PyFunc wrapper for RFM Segmentation using HDBSCAN.
    Handles:
    - Scaling
    - Stable out-of-sample inference
    - Noise vs Whale business logic
    """

    def __init__(self, feature_columns, whale_threshold):
        self.feature_columns = feature_columns
        self.whale_threshold = whale_threshold

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model_path"])
        self.scaler = joblib.load(context.artifacts["scaler_path"])

    def predict(self, context, model_input):
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input, columns=self.feature_columns)

        X = model_input[self.feature_columns]
        X_scaled = self.scaler.transform(X)

        # ✅ Correct HDBSCAN inference
        labels, probs = hdbscan.approximate_predict(self.model, X_scaled)

        is_noise = labels == -1
        is_whale = (labels == -1) & (model_input["Monetary"] >= self.whale_threshold)

        return pd.DataFrame({
            "cluster_label": labels,
            "cluster_probability": probs,
            "is_noise": is_noise,
            "is_whale": is_whale
        })

experiment_name = "RFM_Customer_Segmentation"
mlflow.set_experiment(experiment_name)
client = MlflowClient()

with mlflow.start_run(run_name="HDBSCAN_Full_Lifecycle") as run:
    run_id = run.info.run_id

    # -----------------------------
    # Parameters
    # -----------------------------
    min_cluster_size = max(15, int(len(rfm_scaled) * 0.005))
    params = {
        "min_cluster_size": min_cluster_size,
        "min_samples": 5,
        "metric": "euclidean",
        "cluster_selection_method": "eom"
    }
    mlflow.log_params(params)

    # -----------------------------
    # Model Training
    # -----------------------------
    final_model = hdbscan.HDBSCAN(
        **params,
        gen_min_span_tree=True,
        prediction_data=True   # ⚠️ REQUIRED for approximate_predict
    )
    labels = final_model.fit_predict(rfm_scaled)

    # -----------------------------
    # Metrics
    # -----------------------------
    mask = labels != -1
    dbcv_score = final_model.relative_validity_
    s_score = silhouette_score(rfm_scaled[mask], labels[mask])
    noise_pct = (labels == -1).sum() / len(labels) * 100
    total_stability = clusterer.cluster_persistence_.sum()


    mlflow.log_metrics({
        "DBCV": dbcv_score,
        "Silhouette": s_score,
        "Noise_Percentage": noise_pct,
        "Stability": total_stability

    })

    # -----------------------------
    # Visualization Artifact
    # -----------------------------
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Attribute', y='Value', hue='Cluster', data=rfm_melted)
    plt.title("RFM Snake Plot by Segment")
    plt.tight_layout()
    plt.savefig("rfm_snake_plot.png")
    mlflow.log_artifact("rfm_snake_plot.png")
    plt.close()

    # -----------------------------
    # Save Artifacts
    # -----------------------------
    joblib.dump(final_model, "hdbscan_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    artifacts = {
        "model_path": "hdbscan_model.pkl",
        "scaler_path": "scaler.pkl"
    }

    # -----------------------------
    # Signature (Hybrid)
    # -----------------------------
    input_example = rfm[['Recency', 'Frequency', 'Monetary']].iloc[:3]

    output_example = pd.DataFrame({
        "cluster_label": [0, 1, -1],
        "cluster_probability": [0.91, 0.87, 0.05],
        "is_noise": [False, False, True],
        "is_whale": [False, False, True]
    })

    signature = infer_signature(input_example, output_example)

    whale_threshold = rfm["Monetary"].quantile(0.95)

    # -----------------------------
    # Model Packaging
    # -----------------------------

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=RFMClusterWrapper(
            feature_columns=['Recency', 'Frequency', 'Monetary'],
            whale_threshold=whale_threshold
        ),
        artifacts=artifacts,
        signature=signature,
        input_example=input_example
    )

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

model_name = "RFM_Segmentation_Production"

# Register model from THIS run
registered_model = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name=model_name
)

print(
    f"Model registered: name={registered_model.name}, "
    f"version={registered_model.version}"
)

latest_version = registered_model.version

client.transition_model_version_stage(
    name=model_name,
    version=latest_version,
    stage="Staging",
    archive_existing_versions=False
)

print(f"Model v{latest_version} moved to Staging.")

QUALITY_GATE = (
    dbcv_score > 0.01 and
    s_score > 0.1 and
    noise_pct < 40
)

if QUALITY_GATE:
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"✅ Model v{latest_version} promoted to Production.")
else:
    print("❌ Quality Gate failed. Model stays in Staging.")

