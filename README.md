# NBA Player Performance Prediction

## 📖 Project Overview
The **NBA Player Performance Prediction** app forecasts key player performance metrics (e.g., points per game, rebounds) for upcoming NBA seasons using historical data. This predictive tool empowers NBA team owners and fantasy team managers to make structured, strategic decisions.

---

## 🚀 Key Features

- **Data Preparation:** Organizes and cleans historical NBA player statistics.
- **Trend Analysis:** Identifies patterns and trends in player performances over multiple seasons.
- **Model Implementation:** Uses machine learning models including Linear Regression, Random Forest Regression, and XGBoost.
- **Evaluation:** Model accuracy assessed via Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- **Player Clustering:** Groups players based on statistical similarities to identify meaningful categories.
- **Graph Networks:** Visualizes player relationships and statistical similarities using interactive graph networks.

---

## ⚙️ Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Dependencies
Ensure the following dependencies are installed:

```bash
streamlit
pandas
numpy
plotly
scikit-learn
nba_api
networkx
pillow
python-louvain
scipy
matplotlib
statsmodels
xgboost
seaborn
psutil
pyvis
```

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <your_repository_url>
   cd <repository_folder>
   ```

2. **Create and activate a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

The application opens automatically in your default web browser.
- If it does not open automatically, visit [http://localhost:8501](http://localhost:8501).

---

## 📊 Data

- **Initial Setup:** Data is automatically fetched from the [NBA Stats API](https://pypi.org/project/nba_api/) on first run.
- **Data Refresh:** Use the "Refresh Data" button in the sidebar to update data.
- **Offline Mode:** Cached data in the `data/` folder is used if NBA Stats API is unavailable.

---

## 🗂️ Project Structure

### Features Guide
- **Player Analysis:** Comprehensive statistics for individual players.
- **Team Comparison:** Visual comparison of statistics between teams.
- **Statistical Trends:** League-wide statistical trends over multiple seasons.
- **Performance Prediction:** Forecast player statistics for upcoming seasons.
- **Player Clustering:** Group players based on similar statistical profiles.
- **Player Similarity:** Identify statistically similar players.

### Performance Settings
Adjust visualization quality from the sidebar:
- **High Quality**
- **Balanced**
- **Fast** *(Recommended for low-end devices)*

---

## 📚 Data Source
- **NBA Stats API:** [https://pypi.org/project/nba_api/](https://pypi.org/project/nba_api/)

---

## 🎯 Deliverables

The project deliverables include:

- **GitHub Repository:** Contains all source code, testing data, and documentation.
- **Presentation:** A summary detailing methods, outcomes, and predictions.
- **Final Report:** Comprehensive analysis, visualizations, and detailed insights.
- **Code & Deliverables Submission:** A compressed file with all source code, testing data, and related documentation submitted under the "Code & Deliverables" assignment.

---

This project uses machine learning and data mining techniques to provide valuable insights and reliable predictions for NBA player performance, empowering NBA team owners and fantasy team managers to make structured, strategic decisions.