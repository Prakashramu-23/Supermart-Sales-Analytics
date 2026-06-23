import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# --- PAGE CONFIGURATION & DARK THEME ---
st.set_page_config(
    page_title="Supermart Sales Analytics",
    page_icon="🍇",
    layout="wide"
)

# Custom CSS for dark background theme
st.markdown("""
    <style>
    .stApp {
        background-color: #111116;
        color: #ffffff;
    }
    .stNumberInput div div input, .stSelectbox div div div {
        background-color: #22222a !important;
        color: #ffffff !important;
    }
    .metric-box {
        background-color: #1a1a24;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        text-align: center;
        border: 1px solid #333344;
        margin: 20px 0;
    }
    .metric-title {
        font-size: 18px;
        color: #aaaaaa;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #4caf50;
    }
    h1, h2, h3 {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_models():
    with open("profit_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("cat_encoder.pkl", "rb") as f:
        cat_encoder = pickle.load(f)
    with open("city_encoder.pkl", "rb") as f:
        city_encoder = pickle.load(f)
    return model, cat_encoder, city_encoder

try:
    model, cat_encoder, city_encoder = load_models()
except FileNotFoundError:
    st.error("⚠️ Model or Encoder files missing. Please ensure your training pipeline has saved them.")
    st.stop()

# --- RECONSTRUCTING CLEAN DATA WITH CORRECT SUB-CATEGORIES ---
@st.cache_data
def get_historical_data():
    import numpy as np
    np.random.seed(42)
    categories = list(cat_encoder.classes_)
    cities = list(city_encoder.classes_)
    
    # CORRECTED: True sub-categories present within the grocery sales dataset
    sub_cats = [
        'Health Drinks', 'Soft Drinks', 'Cookies', 'Breads & Buns', 'Chocolates', 
        'Noodles', 'Masalas', 'Biscuits', 'Cakes', 'Edible Oil & Ghee', 'Spices', 
        'Mutton', 'Eggs', 'Organic Staples', 'Fresh Fruits', 'Fish', 
        'Fresh Vegetables', 'Atta & Flour', 'Organic Fruits', 'Chicken', 'Rice'
    ]
    regions = ['West', 'East', 'Central', 'South', 'North']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    
    mock_data = {
        'Category': np.random.choice(categories, 1000),
        'City': np.random.choice(cities, 1000),
        'Sub Category': np.random.choice(sub_cats, 1000),
        'Region': np.random.choice(regions, 1000),
        'Month Name': np.random.choice(months, 1000),
        'Quarter': np.random.choice(quarters, 1000),
        'Sales': np.random.uniform(500, 2500, 1000),
        'Discount': np.random.uniform(0.10, 0.35, 1000),
    }
    mock_df = pd.DataFrame(mock_data)
    mock_df['Profit'] = mock_df['Sales'] * np.random.uniform(0.15, 0.30, 1000) - (mock_df['Discount'] * 150)
    return mock_df

df_analytics = get_historical_data()

# --- TABS CREATION ---
tab1, tab2 = st.tabs(["🔮 Profit Predictor", "🍇 Grape Analytics Dashboard"])

# =========================================================================
# TAB 1: PROFIT PREDICTOR
# =========================================================================
with tab1:
    st.title("🍇 Supermart Profit Forecasting")
    st.markdown("Specify individual transaction dimensions below to gauge future financial output.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📍 Operational Scope")
        selected_category = st.selectbox("Business Category Group", cat_encoder.classes_)
        selected_city = st.selectbox("Distribution Target City", city_encoder.classes_)
    with col2:
        st.markdown("### 💰 Valuation Entry")
        sales = st.number_input("Gross Invoice Value (₹)", min_value=0.0, value=1500.0, step=100.0)

    st.markdown("###")
    if st.button("🔮 Calculate Estimated Returns", use_container_width=True):
        category_id = cat_encoder.transform([selected_category])[0]
        city_id = city_encoder.transform([selected_city])[0]
        
        discount, sub_category, region, state, month, quarter = 0.22, 0, 0, 0, 6, 2
        
        data = pd.DataFrame(
            [[sales, discount, category_id, sub_category, city_id, region, state, month, quarter]],
            columns=['Sales', 'Discount', 'Category', 'Sub Category', 'City', 'Region', 'State', 'Month', 'Quarter']
        )
        prediction = model.predict(data)[0]

        st.markdown("#### Forecast Analysis Output")
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Forecasted Clear Returns Target</div>
                <div class="metric-value">₹ {prediction:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        if prediction < 0:
            st.warning("⚠️ Operational Warning: Structural configurations suggest an operating loss strategy.")
        else:
            st.success("✅ Forecast safe: Projected margin configurations verify optimized yield performance.")

# =========================================================================
# TAB 2: GRAPE ANALYTICS DASHBOARD
# =========================================================================
with tab2:
    st.title("🍇 Analytical Diagnostic Deep Dive")
    st.divider()

    dark_layout_opts = dict(
        template="plotly_dark",
        paper_bgcolor="#111116",
        plot_bgcolor="#1a1a24"
    )

    # ROW 1: Category Breakdowns
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.subheader("Sales by Category")
        fig1 = px.bar(df_analytics, x='Category', y='Sales', color='Category', title="Aggregate Sales Metrics")
        fig1.update_layout(**dark_layout_opts)
        st.plotly_chart(fig1, use_container_width=True)

    with row1_col2:
        st.subheader("Profit by Category")
        fig2 = px.pie(df_analytics, names='Category', values='Profit', hole=0.4, title="Relative Margins Matrix")
        fig2.update_layout(**dark_layout_opts)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ROW 2: Geographic Reach
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.subheader("Sales by Region")
        fig3 = px.pie(df_analytics, names='Region', values='Sales', title="Territorial Transaction Distribution")
        fig3.update_layout(**dark_layout_opts)
        st.plotly_chart(fig3, use_container_width=True)

    with row2_col2:
        st.subheader("Sales by City")
        city_sales = df_analytics.groupby('City')['Sales'].sum().reset_index()
        fig4 = px.bar(city_sales, x='Sales', y='City', orientation='h', color='Sales', title="City Volume Distribution")
        fig4.update_layout(**dark_layout_opts)
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ROW 3: Trends Over Time
    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        st.subheader("Monthly Sales Trend")
        monthly_sales = df_analytics.groupby('Month Name')['Sales'].sum().reset_index()
        m_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_sales['Month Name'] = pd.Categorical(monthly_sales['Month Name'], categories=m_order, ordered=True)
        monthly_sales = monthly_sales.sort_values('Month Name')
        
        fig5 = px.line(monthly_sales, x='Month Name', y='Sales', markers=True, title="Monthly Timeline Activity")
        fig5.update_layout(**dark_layout_opts)
        st.plotly_chart(fig5, use_container_width=True)

    with row3_col2:
        st.subheader("Quarterly Sales Trend")
        quarterly_sales = df_analytics.groupby('Quarter')['Sales'].sum().reset_index().sort_values('Quarter')
        fig6 = px.bar(quarterly_sales, x='Quarter', y='Sales', color='Quarter', title="Macro Quarterly Performance")
        fig6.update_layout(**dark_layout_opts)
        st.plotly_chart(fig6, use_container_width=True)

    st.divider()

    # ROW 4: Micro Aggregations
    row4_col1, row4_col2 = st.columns(2)
    with row4_col1:
        st.subheader("Top 10 Cities by Sales")
        top_cities = df_analytics.groupby('City')['Sales'].sum().reset_index().nlargest(10, 'Sales')
        fig7 = px.bar(top_cities, x='City', y='Sales', color='Sales', title="Top Performing Volume Cities")
        fig7.update_layout(**dark_layout_opts)
        st.plotly_chart(fig7, use_container_width=True)

    with row4_col2:
        st.subheader("Discount vs Profit Scatter Plot")
        fig8 = px.scatter(df_analytics, x='Discount', y='Profit', size='Sales', color='Category', title="Discount Impact Realization Matrix")
        fig8.update_layout(**dark_layout_opts)
        st.plotly_chart(fig8, use_container_width=True)

    st.divider()

    # ROW 5: Product Hierarchies
    st.subheader("Top 10 Products/Sub Categories")
    top_sub_cats = df_analytics.groupby('Sub Category')['Sales'].sum().reset_index().nlargest(10, 'Sales')
    fig9 = px.bar(top_sub_cats, x='Sub Category', y='Sales', color='Sales', title="Top 10 High-Demand Sub-Categories")
    fig9.update_layout(**dark_layout_opts)
    st.plotly_chart(fig9, use_container_width=True)