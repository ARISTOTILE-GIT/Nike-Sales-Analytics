import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Set page to wide layout for better dashboard feel
st.set_page_config(layout="wide")

# --- DATA LOADING AND CLEANING (from notebook) ---
@st.cache_data
def load_data(uploaded_file):
    """Loads and cleans the Nike Sales data based on the notebook's logic."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

    # --- Start Cleaning (copied from notebook cell 150) ---
    numeric_cols = ['Discount_Applied', 'MRP', 'Units_Sold']
    for col in numeric_cols:
        if col in df.columns:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
    
    if 'Size' in df.columns:
        mode_value = df['Size'].mode()[0]
        df[col] = df[col].fillna(mode_value)
    
    # Fix 'Order_Date' column
    df["Order_Date_clean"] = df["Order_Date"].astype(str).str.replace(r"[./]", "-", regex=True)
    df["Order_Date"] = pd.to_datetime(df["Order_Date_clean"], dayfirst=True, errors='coerce')
    
    # Handle remaining NaNs in Order_Date (e.g., fill with mode)
    if df["Order_Date"].isnull().any():
        mode_date = df["Order_Date"].mode()[0]
        df["Order_Date"] = df["Order_Date"].fillna(mode_date)
        
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors='coerce')
    
    df["Year_Month"] = df["Order_Date"].dt.to_period("M")
    df["Month"] = df["Order_Date"].dt.month_name().str[:3]

    # replacing region errors
    replace_names = {"hyderbad":"Hyderabad","Hyd":"Hyderabad","bengaluru":"Bangalore","Banglore":"Bangalore"}
    df["Region"] = df["Region"].replace(replace_names)
    
    # replacing size errors
    replace_size = {'11':'L', '9':'L', '6':'L', '12':'L', '7':'L', '10':'L', '8':'L'}
    df["Size"] = df["Size"].replace(replace_size)
    
    # Fix negative 'Units_Sold' and calculate new 'Revenue' (CRITICAL STEP from notebook)
    df['Units_Sold'] = df['Units_Sold'].apply(lambda x: 1 if (x <= 0 or pd.isna(x)) else x)
    df['Revenue'] = (df['Units_Sold'] * df['MRP']) * (1 - df['Discount_Applied'])
    df['Revenue'] = df['Revenue'].astype("int")
    df['Profit'] = df['Profit'].astype("int")
    # --- End Cleaning ---
    
    return df

# --- SIDEBAR ---
st.sidebar.title("Nike Sales Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload Nike_Sales.csv", type=["csv"])

if uploaded_file is None:
    st.info("Awaiting `Nike_Sales.csv` upload...")
    st.stop()
    
# --- MAIN APP ---
try:
    df = load_data(uploaded_file)
    if df is None:
        st.stop()
except Exception as e:
    st.error(f"Error loading or processing file: {e}")
    st.stop()

st.title("Nike Sales EDA Dashboard")

# --- KPIs (from cell 151) ---
st.header("High-Level KPIs")
try:
    total_revenue = df['Revenue'].sum()
    total_units = df['Units_Sold'].sum()
    unique_products = df['Product_Name'].nunique()
    unique_regions = df['Region'].nunique()
    aov = total_revenue / df['Order_ID'].nunique()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Revenue", f"${total_revenue:,.0f}")
    col2.metric("Total Units Sold", f"{total_units:,.0f}")
    col3.metric("Unique Products", f"{unique_products}")
    col4.metric("Unique Regions", f"{unique_regions}")
    col5.metric("Avg. Order Value", f"${aov:,.2f}")
except Exception as e:
    st.error(f"Could not calculate KPIs. Error: {e}")

st.divider()

# --- Visualization Selector ---
plot_list = [
    "1. Total Revenue by Region",
    "2. Total Revenue by Product Line",
    "3. Revenue by Gender",
    "4. Units Sold by Gender",
    "5. Top 10 Products by Units Sold",
    "6. Top 10 Products by Revenue",
    "7. Average MRP vs. Discount",
    "8. Total Profit by Product Line",
    "9. Monthly Revenue Trend",
    "10. Monthly Revenue Heatmap",
    "11. Revenue by Sales Channel",
    "12. Total Profit by Region",
    "13. Sales by Channel & Product Line",
    "14. Impact of Discounts on Revenue (Regression)"
]

st.sidebar.header("Select Visualization")
# --- Use st.sidebar.radio to create a list of options ---
selected_plot = st.sidebar.radio("Choose a chart:", plot_list)
# --- End of change ---

st.header(selected_plot)

# --- PLOTTING LOGIC (Converted to Plotly for interactivity) ---
try:
    if selected_plot == plot_list[0]: # 1. Revenue by Region
        data = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(data, y="Region", x="Revenue", color="Revenue", text_auto=True, title="Total Revenue by Region")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    elif selected_plot == plot_list[1]: # 2. Revenue by Product Line
        data = df.groupby("Product_Line")["Revenue"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(data, x="Product_Line", y="Revenue", color="Product_Line", text_auto=True, title="Total Revenue by Product Line")
        st.plotly_chart(fig, use_container_width=True)

    elif selected_plot == plot_list[2]: # 3. Revenue by Gender
        data = df.groupby("Gender_Category")["Revenue"].sum().reset_index()
        fig = px.pie(data, names="Gender_Category", values="Revenue", title="Revenue by Gender", hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label+value')
        st.plotly_chart(fig, use_container_width=True)

    elif selected_plot == plot_list[3]: # 4. Units Sold by Gender
        data = df.groupby("Gender_Category")["Units_Sold"].sum().sort_values().reset_index()
        fig = px.bar(data, x="Gender_Category", y="Units_Sold", color="Gender_Category", text_auto=True, title="Total Units Sold by Gender")
        st.plotly_chart(fig, use_container_width=True)

    elif selected_plot == plot_list[4]: # 5. Top 10 Products by Units Sold
        data = df.groupby("Product_Name")["Units_Sold"].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(data, y="Product_Name", x="Units_Sold", color="Units_Sold", text_auto=True, title="Top 10 Products by Units Sold")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    elif selected_plot == plot_list[5]: # 6. Top 10 Products by Revenue
        data = df.groupby("Product_Name")["Revenue"].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(data, y="Product_Name", x="Revenue", color="Revenue", text_auto=True, title="Top 10 Products by Revenue")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    elif selected_plot == plot_list[6]: # 7. Avg. MRP vs. Discount
        data = df.groupby("Discount_Applied")["MRP"].mean().reset_index()
        fig = px.line(data, x="Discount_Applied", y="MRP", title="Average MRP vs. Discount Applied", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        
    elif selected_plot == plot_list[7]: # 8. Profit by Product Line
        data = df.groupby("Product_Line")["Profit"].sum().reset_index()
        fig = px.bar(data, x="Product_Line", y="Profit", color="Product_Line", text_auto=True, title="Total Profit by Product Line")
        st.plotly_chart(fig, use_container_width=True)

    elif selected_plot == plot_list[8]: # 9. Monthly Revenue Trend
        data = df.groupby("Year_Month")["Revenue"].sum().reset_index()
        data["Year_Month_str"] = data["Year_Month"].astype(str) # Plotly needs string for categorical x-axis
        fig = px.line(data, x="Year_Month_str", y="Revenue", title="Monthly Revenue Trend", markers=True)
        fig.update_xaxes(type='category')
        st.plotly_chart(fig, use_container_width=True)
        
    elif selected_plot == plot_list[9]: # 10. Monthly Revenue Heatmap
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot_table = df.pivot_table(index="Product_Line", columns="Month", values="Revenue", aggfunc="sum").fillna(0)
        pivot_table = pivot_table.reindex(columns=month_order)
        fig = px.imshow(pivot_table, text_auto=True, aspect="auto", title="Revenue Heatmap: Month vs. Product Line",
                        color_continuous_scale="YlGnBu")
        st.plotly_chart(fig, use_container_width=True)
        
    elif selected_plot == plot_list[10]: # 11. Revenue by Sales Channel
        data = df.groupby("Sales_Channel")["Revenue"].sum().reset_index()
        fig = px.pie(data, names="Sales_Channel", values="Revenue", title="Revenue by Sales Channel", hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label+value')
        st.plotly_chart(fig, use_container_width=True)
        
    elif selected_plot == plot_list[11]: # 12. Profit by Region
        data = df.groupby("Region")["Profit"].sum().reset_index()
        fig = px.bar(data, x="Region", y="Profit", color="Region", text_auto=True, title="Total Profit by Region")
        st.plotly_chart(fig, use_container_width=True)
        
    elif selected_plot == plot_list[12]: # 13. Sales by Channel & Product Line
        data = df.pivot_table(index="Product_Line", columns="Sales_Channel", values="Revenue", aggfunc="sum").fillna(0).reset_index()
        data_melted = pd.melt(data, id_vars=['Product_Line'], value_vars=['Online', 'Retail'], var_name='Sales_Channel', value_name='Revenue')
        fig = px.bar(data_melted, x="Product_Line", y="Revenue", color="Sales_Channel", barmode="group", text_auto=True, title="Online vs Retail Sales by Product Line")
        st.plotly_chart(fig, use_container_width=True)

    elif selected_plot == plot_list[13]: # 14. Impact of Discounts on Revenue (Regression)
        fig = px.scatter(df, x='Discount_Applied', y='Revenue', trendline='ols', 
                         title="Impact of Discounts on Revenue (with Regression Line)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Re-run regression from notebook to show stats
        try:
            X = df[['Discount_Applied']].fillna(0)
            y = df['Revenue'].fillna(0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            coef = lr.coef_[0]
            intercept = lr.intercept_

            st.text("Regression analysis (from notebook logic):")
            st.code(f"""
R² Score: {r2:.3f}
RMSE: {rmse:.2f}
Linear Model: Revenue = {intercept:.2f} + {coef:.2f} × Discount_Applied
            """, language="text")
        except Exception as e:
            st.warning(f"Could not run regression analysis: {e}")

except Exception as e:
    st.error(f"An error occurred while generating the plot: {e}")
    st.exception(e)
