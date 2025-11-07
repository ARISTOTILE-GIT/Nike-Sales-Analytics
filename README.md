# Nike Sales Analytics Dashboard

[![Live Demo](https://img.shields.io/badge/Live%20Demo-brightgreen?style=for-the-badge&logo=streamlit)](https://nike-sales-analytics-totz.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

</div>

This project is an interactive web-based dashboard built with Streamlit to analyze and visualize Nike sales data. It is based on an initial Exploratory Data Analysis (EDA) performed in a Jupyter Notebook.

The dashboard provides high-level KPIs, 14 distinct visualizations, actionable insights, and a final conclusion to help understand sales performance across different segments.

## 🚀 Live Demo

You can access the live, interactive dashboard here:
**[https://nike-sales-analytics-totz.streamlit.app/](https://nike-sales-analytics-totz.streamlit.app/)**

## Features

* **Dynamic KPI Cards**: View high-level metrics for Total Revenue, Total Units Sold, Unique Products, Unique Regions, and Average Order Value.
* **14 Interactive Visualizations**: Access 14 different interactive plots created with Plotly Express.
* **Dynamic Filtering**: Use the sidebar to toggle between viewing "All Visualizations" at once or selecting a single chart to focus on.
* **Automatic Data Cleaning**: The app automatically cleans the uploaded CSV file based on the logic from the EDA notebook (handles missing values, fixes dates, recalculates revenue).
* **Actionable Insights**: The dashboard concludes with the actionable insights and summary from the original analysis.

## Tech Stack

| Technology | Purpose |
| :--- | :--- |
| **Python** | Core programming language |
| **Streamlit** | For the web app and dashboard UI |
| **Pandas** | For data manipulation and cleaning |
| **Plotly Express** | For interactive visualizations |
| **Scikit-learn** | For regression analysis |
| **Statsmodels** | For regression analysis |


## Setup and Running the Project

To run this dashboard on your local machine, follow these steps:

1.  **Clone or Download the Repository**
    Get the project files onto your computer.

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    All required libraries are listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Add the Data File**
    This app is designed to load your `Nike_Sales.csv` file. Place your `Nike_Sales.csv` file in the same root folder as the `streamlit_app.py` file.

5.  **Run the Streamlit App**
    From your terminal, run the following command:
    ```bash
    streamlit run streamlit_app.py
    ```

6.  **Upload and View**
    The app will open in your browser. Use the sidebar to upload the `Nike_Sales.csv` file. The dashboard will automatically clean the data and generate the report.

## Data Cleaning

The application automatically performs the following cleaning steps from the notebook:

* Fills missing `Discount_Applied`, `MRP`, and `Units_Sold` with their median values.
* Fills missing `Size` values with the mode.
* Parses and standardizes various `Order_Date` formats (e.g., dd/mm/yyyy, yyyy-mm-dd) and fills any remaining nulls with the mode.
* Corrects typos and inconsistencies in `Region` names (e.g., 'Hyd' -> 'Hyderabad').
* Standardizes `Size` values (e.g., '9', '10' -> 'L').
* Corrects invalid `Units_Sold` (e.g., 0 or negative) to 1.
* Recalculates the `Revenue` column based on the cleaned `Units_Sold`, `MRP`, and `Discount_Applied`.

## Visualizations Included

You can choose to view any of the following charts from the sidebar, or all of them at once:

* Total Revenue by Region (Bar Chart)
* Total Revenue by Product Line (Bar Chart)
* Revenue by Gender (Pie Chart)
* Units Sold by Gender (Bar Chart)
* Top 10 Products by Units Sold (Bar Chart)
* Top 10 Products by Revenue (Bar Chart)
* Average MRP vs. Discount (Line Chart)
* Total Profit by Product Line (Bar Chart)
* Monthly Revenue Trend (Line Chart)
* Monthly Revenue Heatmap (Product Line vs. Month)
* Revenue by Sales Channel (Pie Chart)
* Total Profit by Region (Bar Chart)
* Sales by Channel & Product Line (Grouped Bar Chart)
* Impact of Discounts on Revenue (Scatter Plot with Regression)

## Insights & Conclusion

The dashboard concludes by presenting the key **Actionable Business Insights** and the **Overall Conclusion** derived from the exploratory data analysis.
