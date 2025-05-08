import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

def render_predictions(player_df, stats_df, teams_df):
    """Render player performance predictions"""
    st.title("Performance Predictions")
    st.write("Predict player statistics for upcoming seasons based on historical trends")
    
    # Player selection
    selected_player = st.selectbox(
        "Select a player",
        options=player_df['full_name'].sort_values().tolist(),
        index=0
    )
    
    player_id = player_df[player_df['full_name'] == selected_player]['id'].iloc[0]
    
    # Get player's historical stats
    player_stats = stats_df[stats_df['player_id'] == player_id].sort_values('season')
    
    if player_stats.empty:
        st.warning(f"No statistics available for {selected_player}")
        return
        
    # Display available seasons data
    seasons_list = player_stats['season'].unique().tolist()
    seasons_text = ", ".join([str(int(s)) for s in sorted(seasons_list)])
    
    # Get latest available season for validation
    latest_season = player_stats['season'].max()
    
    # Add explanation of data usage
    st.info(f"""
    **Seasons used for prediction:**
    - Available data: {selected_player} has data for {len(seasons_list)} season(s): {seasons_text}
    - Training data: Using seasons prior to {int(latest_season)}
    - Validation against: {int(latest_season)} season (current)
    - Future projection: {int(latest_season)+1} and beyond
    """)
    
    # Allow model selection
    model_type = st.radio(
        "Select prediction model",
        ["Linear Regression", "Random Forest"],
        horizontal=True
    )
    
    # Use data up to the second-latest season for training
    training_seasons = player_stats[player_stats['season'] < latest_season]
    validation_season = player_stats[player_stats['season'] == latest_season]
    
    with st.expander("How our predictions work", expanded=False):
        st.markdown(f"""
        ### Prediction Methodology
        
        **Our model uses the following approach:**
        
        1. We train a {model_type} using {selected_player}'s stats from all seasons **prior to** {int(latest_season)}
        2. We use this model to predict what the player's stats **should have been** in {int(latest_season)}
        3. We compare our prediction with the player's **actual** {int(latest_season)} performance
        4. We then use **all available seasons including {int(latest_season)}** to predict future performance
        
        The accuracy percentage shows how close our prediction was to the actual performance. Higher is better!
        
        **Note:** Predictions become more reliable with more historical seasons of data.
        """)
    
    if len(training_seasons) < 2 or validation_season.empty:
        show_insufficient_data_warning(player_stats, selected_player, latest_season)
        return
        
    # Define stats to predict
    stats_to_predict = ['ppg', 'rpg', 'apg', 'spg', 'bpg', 'fg_pct', 'fg3_pct', 'ft_pct']
    
    # Validate latest season predictions
    predictions, accuracy = validate_predictions(training_seasons, validation_season, stats_to_predict, model_type)
    
    # Display validation results
    st.subheader(f"Prediction vs Actual for {selected_player} ({int(latest_season)})")
    display_validation_results(predictions, validation_season, accuracy, stats_to_predict)
    
    # Calculate overall accuracy
    overall_acc = sum(accuracy.values()) / len(accuracy)
    st.metric("Overall Prediction Accuracy", f"{overall_acc:.1f}%")
    
    # Visualize predictions vs actual values
    visualize_prediction_comparison(predictions, validation_season, stats_to_predict, selected_player, latest_season)
    
    # Future prediction section
    next_season = int(latest_season) + 1
    st.subheader(f"Future Performance Projection ({next_season})")
    
    # Make future predictions
    future_predictions = predict_future_stats(player_stats, stats_to_predict, model_type)
    
    # Display future predictions
    display_future_predictions(future_predictions, validation_season, stats_to_predict)
    
    # Show career prediction chart
    st.subheader(f"Career Trajectory Prediction for {selected_player}")
    
    # Select stat for career projection
    selected_stat = st.selectbox(
        "Select statistic to project",
        options=stats_to_predict,
        index=0
    )
    
    # Create and display career trajectory visualization
    visualize_career_trajectory(
        player_stats, selected_stat, next_season, model_type, 
        selected_player, 3  # Project 3 seasons ahead
    )
    
    # Display model evaluation metrics
    display_model_metrics(player_stats, selected_stat, model_type)
    
    # Warning for small sample sizes
    if len(player_stats) < 5:
        st.warning("⚠️ Limited historical data available. Predictions may be less reliable.")

def show_insufficient_data_warning(player_stats, player_name, latest_season):
    """Display warning for insufficient data"""
    if len(player_stats) == 1:
        st.warning(f"{player_name} only has data for one season ({int(latest_season)}). Cannot make predictions without more historical data.")
    else:
        st.warning(f"Insufficient historical data for {player_name} to make predictions")

def validate_predictions(training_data, validation_data, stats_to_predict, model_type):
    """Train models on training data and validate against validation data"""
    predictions = {}
    accuracy = {}
    
    for stat in stats_to_predict:
        # Prepare training data
        X = np.array(range(len(training_data))).reshape(-1, 1)  # Seasons as feature
        y = training_data[stat].values
        
        # Train model
        model = get_model(model_type)
        model.fit(X, y)
        
        # Make prediction for validation season
        future_X = np.array([len(training_data)]).reshape(-1, 1)
        prediction = model.predict(future_X)[0]
        
        # Round appropriately
        prediction = round_stat(prediction, stat)
        
        # Get actual value
        actual = validation_data[stat].iloc[0]
        
        # Calculate accuracy
        acc = calculate_accuracy(prediction, actual)
        
        # Store results
        predictions[stat] = prediction
        accuracy[stat] = acc
    
    return predictions, accuracy

def predict_future_stats(player_stats, stats_to_predict, model_type):
    """Make predictions for future seasons"""
    future_predictions = {}
    
    for stat in stats_to_predict:
        X = np.array(range(len(player_stats))).reshape(-1, 1)
        y = player_stats[stat].values
        
        # Train model
        model = get_model(model_type)
        model.fit(X, y)
        
        # Predict next season
        future_X = np.array([len(player_stats)]).reshape(-1, 1)
        future_pred = model.predict(future_X)[0]
        
        # Round appropriately
        future_predictions[stat] = round_stat(future_pred, stat)
    
    return future_predictions

def get_model(model_type):
    """Return the appropriate model based on selection"""
    if model_type == "Random Forest":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        return LinearRegression()

def round_stat(value, stat_name):
    """Round stats appropriately based on stat type"""
    if stat_name.endswith('_pct'):
        return round(value, 3)
    else:
        return round(value, 1)

def calculate_accuracy(prediction, actual):
    """Calculate prediction accuracy percentage"""
    if actual != 0:
        acc = 100 - abs((prediction - actual) / actual * 100)
        return max(0, min(100, acc))  # Ensure between 0-100%
    else:
        return 100 if prediction == 0 else 0

def display_validation_results(predictions, validation_data, accuracy, stats_to_predict):
    """Display validation results in a styled table"""
    results = pd.DataFrame({
        'Statistic': stats_to_predict,
        'Predicted': [predictions[stat] for stat in stats_to_predict],
        'Actual': [validation_data[stat].iloc[0] for stat in stats_to_predict],
        'Accuracy': [f"{accuracy[stat]:.1f}%" for stat in stats_to_predict]
    })
    
    # Color code the accuracy column
    def color_accuracy(val):
        val = float(val.replace('%', ''))
        if val >= 90:
            return 'background-color: #c6efce; color: #006100'  # Green
        elif val >= 80:
            return 'background-color: #ffeb9c; color: #9c5700'  # Yellow
        else:
            return 'background-color: #ffc7ce; color: #9c0006'  # Red
    
    # Apply styling
    styled_results = results.style.applymap(color_accuracy, subset=['Accuracy'])
    
    # Display the table
    st.table(styled_results)

def visualize_prediction_comparison(predictions, validation_data, stats_to_predict, player_name, season):
    """Create and display bar chart comparing predictions to actual values"""
    fig = go.Figure()
    
    # Add predicted values
    fig.add_trace(go.Bar(
        x=stats_to_predict,
        y=[predictions[stat] for stat in stats_to_predict],
        name='Predicted',
        marker_color='royalblue'
    ))
    
    # Add actual values
    fig.add_trace(go.Bar(
        x=stats_to_predict,
        y=[validation_data[stat].iloc[0] for stat in stats_to_predict],
        name='Actual',
        marker_color='firebrick'
    ))
    
    fig.update_layout(
        title=f"{player_name}'s Predicted vs Actual Stats ({int(season)})",
        xaxis_title="Statistic",
        yaxis_title="Value",
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_future_predictions(future_predictions, current_data, stats_to_predict):
    """Display future predictions in a styled table"""
    
    # Prepare data with appropriate format for each stat type
    changes = []
    for stat in stats_to_predict:
        diff = future_predictions[stat] - current_data[stat].iloc[0]
        if stat.endswith('_pct'):  # For percentage stats
            changes.append(f"{diff:+.3f}")  # 3 decimal places for percentages
        else:
            changes.append(f"{diff:+.1f}")  # 1 decimal place for counting stats
    
    future_df = pd.DataFrame({
        'Statistic': stats_to_predict,
        'Projected Value': [future_predictions[stat] for stat in stats_to_predict],
        'Current Value': [current_data[stat].iloc[0] for stat in stats_to_predict],
        'Change': changes
    })
    
    # Style the change column
    def color_change(val):
        try:
            val = float(val.replace('+', ''))
            if val > 0:
                return 'color: green'
            elif val < 0:
                return 'color: red'
            return ''
        except:
            return ''
    
    styled_future = future_df.style.applymap(color_change, subset=['Change'])
    
    st.table(styled_future)

def visualize_career_trajectory(player_stats, selected_stat, next_season, model_type, player_name, seasons_ahead=3):
    """Create and display career trajectory visualization"""
    # Get historical data
    historical_seasons = player_stats['season'].astype(int).tolist()
    historical_values = player_stats[selected_stat].tolist()
    
    # Generate future predictions
    future_seasons = list(range(next_season, next_season + seasons_ahead))
    future_values, model, X, y = predict_trajectory(player_stats, selected_stat, model_type, seasons_ahead)
    
    # Create career trajectory chart
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_seasons, 
        y=historical_values,
        mode='lines+markers',
        name=f'Historical ({min(historical_seasons)}-{max(historical_seasons)})',
        line=dict(color='blue')
    ))
    
    # Create a connecting line between historical and predicted data
    fig.add_trace(go.Scatter(
        x=[historical_seasons[-1], future_seasons[0]],
        y=[historical_values[-1], future_values[0]],
        mode='lines',
        line=dict(color='gray', dash='dot'),
        showlegend=False
    ))
    
    # Add predictions with explicit hover information
    fig.add_trace(go.Scatter(
        x=future_seasons, 
        y=future_values,
        mode='lines+markers',
        name=f'Predicted ({min(future_seasons)}-{max(future_seasons)})',
        line=dict(color='red', dash='dash'),
        marker=dict(size=8),  # Larger markers
        text=[f"{selected_stat.upper()}: {value}" for value in future_values],
        hoverinfo='text',
    ))
    
    # Add confidence interval for predictions (only for linear regression)
    if model_type == "Linear Regression":
        # Calculate confidence interval
        conf_interval = calculate_confidence_interval(model, X, y, future_seasons, len(player_stats))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=future_seasons + future_seasons[::-1],
            y=list(conf_interval['upper']) + list(conf_interval['lower'])[::-1],
            fill='toself',
            fillcolor='rgba(231,107,243,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name='95% Confidence Interval',
            hoverinfo='skip'  # Skip hover info for the confidence interval
        ))
    
    fig.update_layout(
        title=f"{player_name}'s Career {selected_stat.upper()} Trajectory",
        xaxis_title="Season",
        yaxis_title=selected_stat.upper(),
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def predict_trajectory(player_stats, selected_stat, model_type, seasons_ahead):
    """Predict trajectory for multiple seasons ahead"""
    X = np.array(range(len(player_stats))).reshape(-1, 1)
    y = player_stats[selected_stat].values
    
    # Train model
    model = get_model(model_type)
    model.fit(X, y)
    
    # Generate predictions
    future_values = []
    for i in range(seasons_ahead):
        future_X = np.array([len(player_stats) + i]).reshape(-1, 1)
        pred = model.predict(future_X)[0]
        future_values.append(round_stat(pred, selected_stat))
    
    return future_values, model, X, y

def display_model_metrics(player_stats, selected_stat, model_type):
    """Display model evaluation metrics"""
    st.subheader("Model Evaluation Metrics")
    
    # Prepare data
    X = np.array(range(len(player_stats))).reshape(-1, 1)
    y = player_stats[selected_stat].values
    
    # Train model
    model = get_model(model_type)
    model.fit(X, y)
    
    # Calculate training metrics
    train_pred = model.predict(X)
    train_mae = mean_absolute_error(y, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y, train_pred))
    train_r2 = r2_score(y, train_pred)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean Absolute Error", f"{train_mae:.2f}")
    with col2:
        st.metric("Root Mean Squared Error", f"{train_rmse:.2f}")
    with col3:
        st.metric("R² Score", f"{train_r2:.2f}")

def calculate_confidence_interval(model, X, y, future_seasons, num_historical_seasons, confidence=0.95):
    """Calculate confidence interval for linear regression predictions"""
    # Get model parameters
    n = len(y)
    p = 2  # Number of parameters (slope and intercept)
    y_pred = model.predict(X)
    
    # Calculate residuals and standard error
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (n - p)
    se = np.sqrt(mse)
    
    # Calculate confidence interval for each future season
    t_value = stats.t.ppf((1 + confidence) / 2, n - p)
    upper = []
    lower = []
    
    for i, season in enumerate(future_seasons):
        # Convert season to the same index format used in training
        x_index = num_historical_seasons + i
        
        # Make prediction
        pred = model.predict(np.array([[x_index]]))[0]
        
        # Calculate prediction interval
        se_pred = se * np.sqrt(1 + 1/n + (x_index - np.mean(X))**2 / np.sum((X - np.mean(X))**2))
        margin = t_value * se_pred
        
        upper.append(pred + margin)
        lower.append(pred - margin)
    
    return {
        'upper': upper,
        'lower': lower
    }