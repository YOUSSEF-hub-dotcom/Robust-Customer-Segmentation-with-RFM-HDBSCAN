import streamlit as st
import requests
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="RFM Customer Intelligence Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://127.0.0.1:8000"

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    .whale-card { background-color: #fff3cd; padding: 20px; border-radius: 10px; border-left: 5px solid #ffc107; }
    </style>
    """, unsafe_allow_html=True)

# 3. القائمة الجانبية (Sidebar)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.title("RFM Analytics Admin")
    st.info(
        "RFM (Recency, Frequency, Monetary) is a marketing technique used to quantitatively rank and group customers.")

    st.divider()
    st.subheader("🛠 Data Management")
    record_id = st.number_input("Enter Record ID to Delete", min_value=1, step=1)
    if st.button("🗑 Delete Record", use_container_width=True):
        try:
            del_resp = requests.delete(f"{API_BASE_URL}/history/{record_id}")
            if del_resp.status_code == 200:
                st.success(f"Record {record_id} deleted successfully!")
            else:
                st.error("Record not found or already deleted.")
        except:
            st.error("API Connection Error")

    st.divider()
    st.caption("v2.0 Production | Public API Mode")

# 4. الواجهة الرئيسية
st.title("🎯 RFM Customer Segmentation Intelligence")
st.write("Leverage Machine Learning to categorize your customers into actionable segments.")

tab1, tab2, tab3 = st.tabs(["🚀 Bulk Analysis (CSV)", "✍️ Single Entry", "📈 System Insights"])

# --- Tab 1: Bulk Analysis ---
with tab1:
    st.subheader("Upload Batch Data")
    uploaded_file = st.file_uploader("Upload customer list (CSV)", type="csv")

    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
        st.dataframe(df_input.head(5), use_container_width=True)

        if st.button("🚀 Process Batch Segmentation", type="primary"):
            with st.spinner("Analyzing large dataset..."):
                try:
                    payload = df_input.to_dict(orient="records")
                    resp = requests.post(f"{API_BASE_URL}/predict", json=payload)

                    if resp.status_code == 200:
                        results = resp.json()["predictions"]
                        df_results = pd.concat([df_input, pd.DataFrame(results)], axis=1)

                        st.success(f"Analysis Complete! Grouped {len(df_results)} customers.")


                        # تمييز الـ Whales باللون الذهبي في الجدول
                        def highlight_whales(row):
                            return ['background-color: #fff3cd' if row.is_whale else '' for _ in row]


                        st.dataframe(df_results.style.apply(highlight_whales, axis=1), use_container_width=True)

                        csv = df_results.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Resulting Segments", csv, "rfm_results.csv", "text/csv")
                    else:
                        st.error(f"Error: {resp.text}")
                except Exception as e:
                    st.error(f"API unreachable: {e}")

# --- Tab 2: Single Entry ---
with tab2:
    st.subheader("Manual Customer Profiling")
    with st.form("manual_form"):
        c1, c2, c3 = st.columns(3)
        rec = c1.number_input("Recency (Days since last purchase)", 0, 1000, 30)
        freq = c2.number_input("Frequency (Number of orders)", 1, 500, 5)
        mon = c3.number_input("Monetary (Total value spent $)", 1.0, 100000.0, 500.0)
        btn = st.form_submit_button("Predict Segment", use_container_width=True, type="primary")

    if btn:
        payload = [{"Recency": int(rec), "Frequency": int(freq), "Monetary": float(mon)}]
        try:
            resp = requests.post(f"{API_BASE_URL}/predict", json=payload)
            if resp.status_code == 200:
                res = resp.json()["predictions"][0]
                cluster = res['cluster_label']

                st.divider()
                col_m, col_v = st.columns([1, 1])

                with col_m:
                    st.write(f"### Segment Result: **Cluster {cluster}**")
                    st.metric("Probability Confidence", f"{res['cluster_probability']:.1%}")

                    if res['is_whale']:
                        st.markdown(
                            '<div class="whale-card">🐋 <b>Whale Alert:</b> High-Value Client! Priority Support required.</div>',
                            unsafe_allow_html=True)

                with col_v:
                    strategies = {
                        -1: ("Outliers / Noise",
                             "Data points that don't fit typical patterns. Manual review suggested."),
                        0: ("New & Low Spenders",
                            "Fresh customers but low engagement. Offer a 'First Purchase' reward."),
                        1: ("Loyalists",
                            "Frequent buyers with consistent spending. Keep them engaged with early access."),
                        2: ("Potential Loyalists",
                            "Show high frequency but moderate spending. Cross-sell higher-value items."),
                        3: ("At Risk", "Used to buy often but slowing down. Send a 'We Miss You' discount code."),
                        4: ("About to Sleep", "Low frequency and long recency. Last chance for a win-back campaign."),
                        5: ("Champions / Whales",
                            "High frequency and top-tier spending! Provide VIP support and exclusive gifts.")
                    }

                    title, desc = strategies.get(cluster, ("Active User", "Standard marketing follow-up."))

                    st.subheader(f"Strategy: {title}")
                    st.info(f"💡 **Action:** {desc}")
            else:
                st.error("Prediction failed.")
        except Exception as e:
            st.error(f"Connection failed: {e}")

# --- Tab 3: Insights & Dashboard ---
with tab3:
    st.subheader("🌐 Global Customer Intelligence & 3D Mapping")
    st.markdown("Visualize how customers are clustered based on their buying behavior (RFM).")

    # زر التحديث لجلب البيانات الجديدة
    if st.button("📊 Update Analytics & 3D Map", use_container_width=True, type="secondary"):
        try:
            with st.spinner("Fetching latest data and generating insights..."):
                # 1. جلب البيانات من الـ API
                hist_resp = requests.get(f"{API_BASE_URL}/history?limit=1000", timeout=5)

                if hist_resp.status_code == 200:
                    data = hist_resp.json()
                    if not data:
                        st.warning("No data found in the database. Try running some predictions first!")
                    else:
                        df_viz = pd.DataFrame(data)
                        # توحيد أسامي الأعمدة لتجنب أخطاء الحروف الكبيرة/الصغيرة
                        df_viz.columns = [c.lower() for c in df_viz.columns]

                        # --- القسم الأول: المؤشرات الرئيسية (KPIs) ---
                        st.divider()
                        kpi1, kpi2, kpi3 = st.columns(3)

                        with kpi1:
                            st.metric("Total Customers", f"{len(df_viz)}")
                        with kpi2:
                            # حساب عدد الـ Whales إذا كان العمود موجوداً
                            whales_count = len(
                                df_viz[df_viz['is_whale'] == True]) if 'is_whale' in df_viz.columns else 0
                            st.metric("High-Value Whales 🐋", whales_count)
                        with kpi3:
                            # حساب متوسط الصرف
                            avg_mon = df_viz['monetary'].mean() if 'monetary' in df_viz.columns else 0
                            st.metric("Avg. Monetary Value", f"${avg_mon:,.2f}")

                        # --- القسم الثاني: الرسم ثلاثي الأبعاد (3D RFM Plot) ---
                        st.write("#### 📍 3D Cluster Visualization")
                        st.caption("Rotate and zoom to explore customer positions in the RFM space.")

                        required_cols = ['recency', 'frequency', 'monetary', 'cluster_label']
                        if all(col in df_viz.columns for col in required_cols):
                            fig_3d = px.scatter_3d(
                                df_viz,
                                x='recency',
                                y='frequency',
                                z='monetary',
                                color='cluster_label',
                                symbol='is_whale' if 'is_whale' in df_viz.columns else None,
                                opacity=0.8,
                                color_continuous_scale='Viridis',
                                labels={
                                    'recency': 'Days Since Last Buy',
                                    'frequency': 'Total Orders',
                                    'monetary': 'Total Spent ($)',
                                    'cluster_label': 'Segment ID'
                                },
                                title="Customer Positioning (Recency vs Frequency vs Monetary)"
                            )
                            # تحسين شكل الرسم
                            fig_3d.update_layout(
                                margin=dict(l=0, r=0, b=0, t=30),
                                scene=dict(
                                    xaxis_backgroundcolor="rgb(230, 230,230)",
                                    yaxis_backgroundcolor="rgb(230, 230,230)",
                                    zaxis_backgroundcolor="rgb(230, 230,230)"
                                )
                            )
                            st.plotly_chart(fig_3d, use_container_width=True)
                        else:
                            st.error(
                                f"⚠️ Visualization Error: Required columns not found. Found: {list(df_viz.columns)}")

                        # --- القسم الثالث: التوزيع والجدول التفصيلي ---
                        st.divider()
                        col_pie, col_table = st.columns([1, 1.2])

                        with col_pie:
                            st.write("#### 🍰 Segment Distribution")
                            if 'cluster_label' in df_viz.columns:
                                fig_pie = px.pie(
                                    df_viz,
                                    names='cluster_label',
                                    hole=0.4,
                                    color_discrete_sequence=px.colors.qualitative.Pastel
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)

                        with col_table:
                            st.write("#### 📋 Recent Activity Logs")
                            # عرض آخر 10 سجلات مع تلوين بسيط
                            display_cols = ['id', 'recency', 'frequency', 'monetary', 'cluster_label', 'is_whale']
                            available_display = [c for c in display_cols if c in df_viz.columns]
                            st.dataframe(df_viz[available_display].tail(10), use_container_width=True)

                else:
                    st.error(f"❌ API Error: Received status code {hist_resp.status_code}")

        except Exception as e:
            st.error(f"🚀 Dashboard Connection Error: {str(e)}")
            st.info("Make sure the FastAPI server is running on http://127.0.0.1:8000")
# 5. التذييل
st.divider()
st.caption("RFM Enterprise Deployment v2.0 | Built with Streamlit & FastAPI")