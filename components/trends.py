import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_trends(player_df, stats_df, teams_df):
    """Render league-wide trends and statistical patterns over time"""
    st.title("NBA Statistical Trends")
    st.write("Analyze how NBA statistics and playing styles have evolved over time")
    
    # Verify we have multi-season data
    available_seasons = sorted(stats_df['season'].unique())
    if len(available_seasons) <= 1:
        st.warning("Multiple seasons of data are required to analyze trends. Please load more historical data.")
        return
    
    # Create tabs for different trend analyses
    trend_tabs = st.tabs([
        "Scoring Evolution", 
        "Playing Style Changes", 
        "Positional Evolution", 
        "Statistical Distributions"
    ])
    
    with trend_tabs[0]:
        render_scoring_trends(stats_df, available_seasons)
    with trend_tabs[1]:
        render_playing_style_trends(stats_df, available_seasons)
    with trend_tabs[2]:
        render_positional_trends(player_df, stats_df, available_seasons)
    with trend_tabs[3]:
        render_statistical_distributions(stats_df, available_seasons)

def get_qualified_players(stats_df, season, min_minutes=15):
    """Get qualified players for a given season"""
    season_data = stats_df[stats_df['season'] == season]
    return season_data[season_data['minutes'] >= min_minutes]

def calculate_season_averages(available_seasons, stats_df):
    """Calculate league averages for each season"""
    season_averages = []
    
    for season in available_seasons:
        qualified_players = get_qualified_players(stats_df, season)
        
        if qualified_players.empty:
            continue
            
        # Calculate averages
        ppg_avg = qualified_players['ppg'].mean()
        fg_pct_avg = qualified_players['fg_pct'].mean()
        fg3_pct_avg = qualified_players['fg3_pct'].mean()
        ft_pct_avg = qualified_players['ft_pct'].mean()
        
        # Calculate scoring distribution based on league averages
        if all(col in qualified_players.columns for col in ['ppg', 'fg2m', 'fg3m', 'ftm']):
            # If we have actual made shot data
            points_from_2 = qualified_players['fg2m'].mean() * 2
            points_from_3 = qualified_players['fg3m'].mean() * 3
            points_from_ft = qualified_players['ftm'].mean()
        else:
            # Estimate based on typical NBA distribution
            # NBA averages: ~40% from 2PT, ~35% from 3PT, ~25% from FT
            points_from_2 = ppg_avg * 0.40
            points_from_3 = ppg_avg * 0.35
            points_from_ft = ppg_avg * 0.25
            
            # Verify the total matches PPG
            total = points_from_2 + points_from_3 + points_from_ft
            if abs(total - ppg_avg) > 0.1:
                # Adjust to match PPG
                scale_factor = ppg_avg / total
                points_from_2 *= scale_factor
                points_from_3 *= scale_factor
                points_from_ft *= scale_factor
        
        season_averages.append({
            'Season': int(season),  # FIX 1: Ensure season is an integer
            'PPG': ppg_avg,
            'FG%': fg_pct_avg,
            'FG3%': fg3_pct_avg,
            'FT%': ft_pct_avg,
            'Points from 2PT': points_from_2,
            'Points from 3PT': points_from_3,
            'Points from FT': points_from_ft
        })
    
    return pd.DataFrame(season_averages)

def create_line_chart(df, x_col, y_col, title, y_label=None, add_markers=True, height=450, 
                      show_max_annotation=False, y_format=None):
    """Create a standardized line chart with optional features"""
    # FIX 1: Ensure seasons are displayed as integers
    df = df.copy()
    if x_col == 'Season':
        df[x_col] = df[x_col].astype(int)
    
    fig = px.line(
        df, x=x_col, y=y_col, 
        markers=add_markers,
        title=title,
        labels={x_col: x_col, y_col: y_label or y_col}
    )
    
    # Add annotation for the maximum value
    if show_max_annotation and len(df) > 2:
        max_idx = df[y_col].idxmax()
        max_x = df.iloc[max_idx][x_col]
        max_y = df.iloc[max_idx][y_col]
        
        fig.add_annotation(
            x=max_x, y=max_y,
            text=f"Peak: {max_y:.1f}",
            showarrow=True,
            arrowhead=1
        )
    
    # Apply formatting to the y-axis if specified
    if y_format:
        fig.update_layout(yaxis=dict(tickformat=y_format))
    
    # FIX 1: Ensure x-axis shows integer seasons
    if x_col == 'Season':
        season_values = sorted(df[x_col].unique())
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=season_values,
                ticktext=[str(int(s)) for s in season_values],
                tickformat="d"  # Display as integers
            )
        )
    
    fig.update_layout(height=height)
    return fig

def calculate_percent_change(first_val, last_val):
    """Calculate percentage change between two values"""
    if first_val == 0:
        return 0
    return (last_val - first_val) / first_val * 100

def render_scoring_trends(stats_df, available_seasons):
    """Analyze trends in scoring metrics"""
    st.subheader("Scoring Evolution")
    
    # Calculate league averages per season
    trend_df = calculate_season_averages(available_seasons, stats_df)
    
    if trend_df.empty:
        st.info("Insufficient data to analyze scoring trends.")
        return
    
    # Scoring trends over time
    st.markdown("### Points Per Game Over Time")
    fig = create_line_chart(
        trend_df, 'Season', 'PPG', 
        "Average Points Per Game by Season", 
        "Points Per Game",
        show_max_annotation=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Scoring distribution over time
    st.markdown("### Scoring Distribution by Shot Type")
    
    # Create stacked area chart for scoring distribution
    fig = go.Figure()
    
    # Add each scoring component
    for score_type, color in [
        ('Points from 2PT', 'rgb(73, 160, 213)'),
        ('Points from 3PT', 'rgb(224, 123, 57)'),
        ('Points from FT', 'rgb(126, 172, 109)')
    ]:
        fig.add_trace(go.Scatter(
            x=trend_df['Season'],
            y=trend_df[score_type],
            mode='lines',
            stackgroup='one',
            name=score_type.replace('Points from ', ''),
            line=dict(width=0.5, color=color)
        ))
    
    # FIX 1: Configure x-axis for integer seasons
    season_values = sorted(trend_df['Season'].astype(int).unique())
    
    fig.update_layout(
        title="Scoring Distribution by Shot Type",
        xaxis=dict(
            title="Season",
            tickmode='array',
            tickvals=season_values,
            ticktext=[str(s) for s in season_values]
        ),
        yaxis_title="Points Per Game",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Shooting percentages over time
    st.markdown("### Shooting Percentages Over Time")
    
    # Create multi-line chart for percentages
    shooting_data = trend_df[['Season', 'FG%', 'FG3%', 'FT%']].melt(
        id_vars=['Season'],
        value_vars=['FG%', 'FG3%', 'FT%'],
        var_name='Percentage',
        value_name='Value'
    )
    
    fig = px.line(
        shooting_data, 
        x='Season', y='Value', color='Percentage',
        markers=True,
        title="Shooting Percentages by Season",
        labels={'Value': 'Percentage', 'Season': 'Season'}
    )
    
    # FIX 1: Configure x-axis for integer seasons
    season_values = sorted(shooting_data['Season'].astype(int).unique())
    
    fig.update_layout(
        height=450,
        yaxis=dict(tickformat='.1%'),
        xaxis=dict(
            tickmode='array',
            tickvals=season_values,
            ticktext=[str(s) for s in season_values]
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights about scoring trends
    st.subheader("Key Insights")
    
    # Calculate percentage changes between first and last season
    first_season = trend_df.iloc[0]
    last_season = trend_df.iloc[-1]
    
    # FIX 3: Fix the ambiguity error by safely creating insights
    insights = []
    
    # Add PPG insight
    ppg_change = calculate_percent_change(first_season['PPG'], last_season['PPG'])
    insights.append(
        f"Average scoring has {'increased' if ppg_change > 0 else 'decreased'} by {abs(ppg_change):.1f}% since {int(first_season['Season'])}."
    )
    
    # Add 3PT shooting insight
    fg3_pct_change = calculate_percent_change(first_season['FG3%'], last_season['FG3%'])
    insights.append(
        f"Three-point shooting accuracy has {'improved' if fg3_pct_change > 0 else 'declined'} by {abs(fg3_pct_change):.1f}% over this period."
    )
    
    # Add 3PT contribution insight
    points_from_3_change = calculate_percent_change(first_season['Points from 3PT'], last_season['Points from 3PT'])
    insights.append(
        f"The contribution of 3-point shots to total scoring has {'grown' if points_from_3_change > 0 else 'decreased'} by {abs(points_from_3_change):.1f}%."
    )
    
    # Add current 3PT ratio insight
    total_points = sum(last_season[col] for col in ['Points from 2PT', 'Points from 3PT', 'Points from FT'])
    current_3pt_ratio = (last_season['Points from 3PT'] / total_points * 100) if total_points > 0 else 0
    insights.append(
        f"In {int(last_season['Season'])}, approximately {current_3pt_ratio:.1f}% of points came from 3-pointers."
    )
    
    for insight in insights:
        st.write(f"• {insight}")

def render_playing_style_trends(stats_df, available_seasons):
    """Analyze trends in playing style metrics"""
    st.subheader("Playing Style Evolution")
    
    # Calculate metrics by season
    style_trends = []
    
    for season in available_seasons:
        qualified_players = get_qualified_players(stats_df, season)
        
        if qualified_players.empty:
            continue
            
        # Calculate style metrics
        avg_minutes = qualified_players['minutes'].mean()
        
        # FIX 3: Handle division by zero and Series issues
        ppg_mean = qualified_players['ppg'].mean()
        assist_ratio = (qualified_players['apg'].mean() / ppg_mean * 100) if ppg_mean > 0 else 0
        block_ratio = (qualified_players['bpg'].mean() / ppg_mean * 100) if ppg_mean > 0 else 0
        steal_ratio = (qualified_players['spg'].mean() / ppg_mean * 100) if ppg_mean > 0 else 0
        
        # Estimate pace from available metrics
        player_minutes_mean = qualified_players['minutes'].mean()
        estimated_pace = (qualified_players['ppg'].sum() / (player_minutes_mean * len(qualified_players))) if player_minutes_mean > 0 else 0
        
        style_trends.append({
            'Season': int(season),  # FIX 1: Ensure season is an integer
            'Minutes Per Game': avg_minutes,
            'Assist Ratio': assist_ratio,
            'Block Ratio': block_ratio,
            'Steal Ratio': steal_ratio,
            'Pace Indicator': estimated_pace
        })
    
    style_df = pd.DataFrame(style_trends)
    
    if style_df.empty:
        st.info("Insufficient data to analyze playing style trends.")
        return
    
    # Plot minutes per game trend
    st.markdown("### Minutes Per Game Trend")
    
    fig = create_line_chart(
        style_df, 'Season', 'Minutes Per Game',
        "Average Minutes Per Game by Season", 
        "Minutes",
        height=400
    )
    
    # Add trendline
    if len(style_df) > 2:
        x = np.array(range(len(style_df)))
        y = style_df['Minutes Per Game'].values
        coeffs = np.polyfit(x, y, 1)
        trend_y = coeffs[0] * x + coeffs[1]
        
        fig.add_trace(
            go.Scatter(
                x=style_df['Season'],
                y=trend_y,
                mode='lines',
                name='Trend',
                line=dict(color='rgba(0,0,0,0.3)', dash='dash')
            )
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Plot assist, block, steal ratios
    st.markdown("### Play Style Indicators")
    
    # Create subplots for different metrics
    fig = make_subplots(
        rows=1, cols=3, 
        subplot_titles=("Assist Ratio", "Block Ratio", "Steal Ratio")
    )
    
    # Add traces for each metric
    metrics = ['Assist Ratio', 'Block Ratio', 'Steal Ratio']
    y_titles = ["Assists per 100 Points", "Blocks per 100 Points", "Steals per 100 Points"]
    
    for i, (metric, y_title) in enumerate(zip(metrics, y_titles)):
        fig.add_trace(
            go.Scatter(
                x=style_df['Season'],
                y=style_df[metric],
                mode='lines+markers',
                name=metric
            ),
            row=1, col=i+1
        )
        fig.update_yaxes(title_text=y_title, row=1, col=i+1)
    
    # FIX 1: Configure x-axis for integer seasons
    season_values = sorted(style_df['Season'].astype(int).unique())
    for i in range(1, 4):
        fig.update_xaxes(
            tickmode='array',
            tickvals=season_values,
            ticktext=[str(s) for s in season_values],
            row=1, col=i
        )
    
    # Update layout
    fig.update_layout(
        height=400,
        title_text="Play Style Indicators Over Time",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Game pace over time
    st.markdown("### Game Pace Indicator")
    
    fig = create_line_chart(
        style_df, 'Season', 'Pace Indicator',
        "Game Pace Indicator Over Time", 
        "Pace Indicator",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights about playing style
    st.subheader("Key Insights")
    
    # Calculate percentage changes
    if len(style_df) >= 2:
        first_season = style_df.iloc[0]
        last_season = style_df.iloc[-1]
        
        # FIX 3: Fix the ambiguity error by safely creating insights
        insights = []
        
        # Add minutes insight
        min_change = calculate_percent_change(first_season['Minutes Per Game'], last_season['Minutes Per Game'])
        insights.append(
            f"Player minutes have {'increased' if min_change > 0 else 'decreased'} by {abs(min_change):.1f}% since {int(first_season['Season'])}."
        )
        
        # Add assist ratio insight
        assist_change = calculate_percent_change(first_season['Assist Ratio'], last_season['Assist Ratio'])
        insights.append(
            f"The ratio of assists to points has {'increased' if assist_change > 0 else 'decreased'} by {abs(assist_change):.1f}%, suggesting {'more' if assist_change > 0 else 'less'} team-oriented play."
        )
        
        # Add pace insight
        pace_change = calculate_percent_change(first_season['Pace Indicator'], last_season['Pace Indicator'])
        insights.append(
            f"The game pace indicator has {'increased' if pace_change > 0 else 'decreased'} by {abs(pace_change):.1f}%, indicating {'faster' if pace_change > 0 else 'slower'} gameplay."
        )
        
        for insight in insights:
            st.write(f"• {insight}")
    else:
        st.info("More seasons needed to calculate meaningful insights.")

def render_positional_trends(player_df, stats_df, available_seasons):
    """Analyze trends in positional stats and roles"""
    st.subheader("Positional Evolution")
    
    # Check if we have position data
    if 'position' not in player_df.columns:
        st.info("Position data is not available in the dataset. Cannot analyze positional trends.")
        
        # Alternative analysis based on height groups (if available)
        if 'height' in player_df.columns:
            st.write("However, we can analyze trends based on player height groups instead.")
            # Logic for height-based analysis would go here
        else:
            st.write("Consider adding position or height data to enable this analysis.")
        return
    
    # Calculate stats by position and season
    position_trends = []
    
    for season in available_seasons:
        season_data = stats_df[stats_df['season'] == season]
        
        # Join with player data to get positions
        player_season = season_data.merge(
            player_df[['id', 'position']], 
            left_on='player_id', 
            right_on='id', 
            how='left'
        )
        
        # Aggregate by position
        for position in player_season['position'].unique():
            if pd.notna(position):
                pos_data = player_season[player_season['position'] == position]
                
                # Calculate position averages
                pos_trends = {
                    'Season': int(season),  # FIX 1: Ensure season is an integer
                    'Position': position,
                    'PPG': pos_data['ppg'].mean(),
                    'RPG': pos_data['rpg'].mean(),
                    'APG': pos_data['apg'].mean(),
                    'BPG': pos_data['bpg'].mean(),
                    'SPG': pos_data['spg'].mean(),
                    'FG3%': pos_data['fg3_pct'].mean(),
                    'Count': len(pos_data)
                }
                
                position_trends.append(pos_trends)
    
    if not position_trends:
        st.info("Insufficient data to analyze positional trends.")
        return
        
    pos_df = pd.DataFrame(position_trends)
    
    # Show position scoring trends
    st.markdown("### Scoring Trends by Position")
    
    # Create position trend visualization
    create_positional_trend_chart(pos_df, 'PPG', 'Points Per Game')
    
    # 3-point percentage by position
    st.markdown("### Three-Point Shooting by Position")
    
    create_positional_trend_chart(pos_df, 'FG3%', '3-Point Percentage', is_percentage=True)
    
    # Position versatility index
    st.markdown("### Position Versatility Index")
    
    # Calculate versatility by position
    versatility_data = calculate_position_versatility(pos_df, available_seasons)
    
    if versatility_data:
        vers_df = pd.DataFrame(versatility_data)
        
        # Create position trend chart for versatility
        create_positional_trend_chart(vers_df, 'Versatility Index', 'Versatility Index (0-100)')
        
        # Key insights about positional evolution
        st.subheader("Key Insights")
        
        # Generate positional insights
        insights = generate_positional_insights(pos_df, vers_df, available_seasons)
        
        for insight in insights:
            st.write(f"• {insight}")
    else:
        st.info("Not enough position data available to calculate versatility.")

def create_positional_trend_chart(pos_df, value_column, y_label, is_percentage=False):
    """Create a line chart showing trends by position"""
    # FIX 1: Ensure seasons are displayed as integers
    pos_df = pos_df.copy()
    pos_df['Season'] = pos_df['Season'].astype(int)
    
    # Pivot data for line plot
    pivot_df = pos_df.pivot(index='Season', columns='Position', values=value_column).reset_index()
    
    # Melt back for plotly
    melted_df = pd.melt(
        pivot_df, 
        id_vars=['Season'], 
        value_vars=pivot_df.columns[1:],
        var_name='Position', 
        value_name=value_column
    )
    
    # Create line plot
    fig = px.line(
        melted_df,
        x='Season',
        y=value_column,
        color='Position',
        markers=True,
        title=f"{y_label} by Position Over Time",
        labels={value_column: y_label, 'Season': 'Season'}
    )
    
    # Format for percentages if needed
    if is_percentage:
        fig.update_layout(yaxis=dict(tickformat='.1%'))
    
    # FIX 1: Configure x-axis for integer seasons
    season_values = sorted(pos_df['Season'].unique())
    fig.update_layout(
        height=450,
        xaxis=dict(
            tickmode='array',
            tickvals=season_values,
            ticktext=[str(s) for s in season_values]
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def calculate_position_versatility(pos_df, available_seasons):
    """Calculate position versatility metrics"""
    # Traditional expected stats by position (historical expectations)
    position_expectations = {
        'PG': {'APG': 'high', 'RPG': 'low'},
        'SG': {'PPG': 'high', 'APG': 'medium'},
        'SF': {'PPG': 'high', 'RPG': 'medium'},
        'PF': {'RPG': 'high', 'BPG': 'medium'},
        'C': {'RPG': 'high', 'BPG': 'high', 'APG': 'low'}
    }
    
    versatility_data = []
    
    for season in available_seasons:
        # FIX 1: Convert season to integer for consistency
        season_int = int(season)
        season_pos = pos_df[pos_df['Season'] == season_int]
        
        for position in season_pos['Position'].unique():
            if position in position_expectations and position in season_pos['Position'].values:
                pos_row = season_pos[season_pos['Position'] == position]
                if pos_row.empty:
                    continue
                
                pos_row = pos_row.iloc[0]
                
                # Calculate versatility index
                versatility = 0
                
                # Check for non-traditional stats for this position
                if position in ['PG', 'SG'] and pos_row['RPG'] > 5:
                    versatility += 1  # Guards rebounding well
                
                if position in ['PF', 'C'] and pos_row['APG'] > 3:
                    versatility += 1  # Big men passing well
                
                if position in ['PF', 'C'] and pos_row['FG3%'] > 0.33:
                    versatility += 1  # Big men shooting 3s well
                
                if position in ['PG'] and pos_row['BPG'] > 0.5:
                    versatility += 1  # Point guards blocking shots
                
                # Normalize to 0-100 scale
                versatility = min(100, versatility * 25)
                
                versatility_data.append({
                    'Season': season_int,  # FIX 1: Store season as integer
                    'Position': position,
                    'Versatility Index': versatility
                })
    
    return versatility_data

def generate_positional_insights(pos_df, vers_df, available_seasons):
    """Generate insights about positional evolution"""
    insights = []
    
    # Find position with most change in 3PT
    if len(available_seasons) >= 2:
        # FIX 1: Convert seasons to integers
        first_season = int(available_seasons[0])
        last_season = int(available_seasons[-1])
        
        pos_first_season = pos_df[pos_df['Season'] == first_season]
        pos_last_season = pos_df[pos_df['Season'] == last_season]
        
        if not pos_first_season.empty and not pos_last_season.empty:
            # FIX 3: Safely handle position data by creating position data frames
            pos_first_dict = {row['Position']: row for _, row in pos_first_season.iterrows()}
            pos_last_dict = {row['Position']: row for _, row in pos_last_season.iterrows()}
            
            # Traditional positions to check
            positions = ['PG', 'SG', 'SF', 'PF', 'C']
            
            max_3pt_change_pos = None
            max_3pt_change_val = 0
            
            for position in positions:
                if position in pos_first_dict and position in pos_last_dict:
                    first_val = pos_first_dict[position]['FG3%']
                    last_val = pos_last_dict[position]['FG3%']
                    change = last_val - first_val
                    if abs(change) > abs(max_3pt_change_val):
                        max_3pt_change_pos = position
                        max_3pt_change_val = change
            
            if max_3pt_change_pos:
                insights.append(f"{max_3pt_change_pos} position has seen the largest change in 3-point shooting: {max_3pt_change_val*100:.1f}% {'increase' if max_3pt_change_val > 0 else 'decrease'}.")
    
    # Check versatility changes
    if len(vers_df) > 0:
        # FIX 1: Convert seasons to integers
        first_season = int(available_seasons[0])
        last_season = int(available_seasons[-1])
        
        # FIX 3: Safely handle Series by calculating mean values directly
        vers_first = vers_df[vers_df['Season'] == first_season]
        vers_last = vers_df[vers_df['Season'] == last_season]
        
        if not vers_first.empty and not vers_last.empty:
            vers_first_avg = vers_first.groupby('Position')['Versatility Index'].mean()
            vers_last_avg = vers_last.groupby('Position')['Versatility Index'].mean()
            
            overall_vers_first = vers_first['Versatility Index'].mean()
            overall_vers_last = vers_last['Versatility Index'].mean()
            overall_vers_change = overall_vers_last - overall_vers_first
            
            insights.append(f"Overall position versatility has {'increased' if overall_vers_change > 0 else 'decreased'} by {abs(overall_vers_change):.1f} points.")
            
            # Most versatile position
            if len(vers_last) > 0:
                most_versatile_row = vers_last.loc[vers_last['Versatility Index'].idxmax()]
                most_versatile = most_versatile_row['Position']
                highest_versatility = most_versatile_row['Versatility Index']
                
                insights.append(f"The {most_versatile} position currently shows the highest versatility index at {highest_versatility:.1f}.")
    
    return insights

def render_statistical_distributions(stats_df, available_seasons):
    """Analyze how statistical distributions have changed over time"""
    st.subheader("Statistical Distributions")
    
    # Select seasons for comparison
    col1, col2 = st.columns(2)
    
    # FIX 1: Ensure seasons are integers
    available_seasons = [int(season) for season in available_seasons]
    
    early_seasons = available_seasons[:int(len(available_seasons)/2)]
    recent_seasons = available_seasons[int(len(available_seasons)/2):]
    
    with col1:
        early_season = st.selectbox(
            "Select early season",
            options=early_seasons,
            index=0,
            key="early_season_select"
        )
    
    with col2:
        recent_season = st.selectbox(
            "Select recent season",
            options=recent_seasons,
            index=len(recent_seasons) - 1 if recent_seasons else 0,
            key="recent_season_select"
        )
    
    # Get data for selected seasons
    early_qualified = get_qualified_players(stats_df, early_season)
    recent_qualified = get_qualified_players(stats_df, recent_season)
    
    if early_qualified.empty or recent_qualified.empty:
        st.warning("Not enough qualified players in the selected seasons for comparison.")
        return
        
    # Distribution comparison
    st.markdown(f"### Statistical Distribution: {early_season} vs {recent_season}")
    
    # Let user select statistic to compare
    stat_options = {
        'ppg': 'Points Per Game',
        'rpg': 'Rebounds Per Game',
        'apg': 'Assists Per Game',
        'spg': 'Steals Per Game',
        'bpg': 'Blocks Per Game',
        'fg_pct': 'Field Goal %',
        'fg3_pct': '3-Point %',
        'ft_pct': 'Free Throw %'
    }
    
    selected_stat = st.selectbox(
        "Select statistic to compare",
        options=list(stat_options.keys()),
        format_func=lambda x: stat_options[x],
        key="dist_stat_select"
    )
    
    # Create distribution comparison visualization
    create_distribution_comparison(early_qualified, recent_qualified, selected_stat, 
                                  stat_options[selected_stat], early_season, recent_season)
    
    # Statistical summary table
    create_summary_table(early_qualified, recent_qualified, stat_options, early_season, recent_season)
    
    # Gini coefficient analysis
    st.markdown("### Statistical Inequality Analysis")
    st.write("The Gini coefficient measures statistical inequality among players (0 = perfect equality, 1 = perfect inequality)")
    
    # Create Gini visualization
    create_gini_comparison(early_qualified, recent_qualified, stat_options, early_season, recent_season)
    
    # Generate and display insights
    display_distribution_insights(early_qualified, recent_qualified, stat_options, early_season, recent_season)

def create_distribution_comparison(early_data, recent_data, selected_stat, stat_label, early_season, recent_season):
    """Create a distribution comparison visualization"""
    # Create DataFrames with data from both seasons
    early_season_data = early_data[['player_id', selected_stat]].copy()
    early_season_data['Season'] = str(early_season)
    
    recent_season_data = recent_data[['player_id', selected_stat]].copy()
    recent_season_data['Season'] = str(recent_season)
    
    # Combine the data
    combined_data = pd.concat([early_season_data, recent_season_data])
    
    # Create histogram
    fig = px.histogram(
        combined_data,
        x=selected_stat,
        color='Season',
        barmode='overlay',
        opacity=0.7,
        marginal="box",
        histnorm='percent',
        labels={selected_stat: stat_label},
        title=f"Distribution of {stat_label}: {early_season} vs {recent_season}"
    )
    
    # Format x-axis for percentages
    if selected_stat in ['fg_pct', 'fg3_pct', 'ft_pct']:
        fig.update_xaxes(tickformat='.0%')
    
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

def create_summary_table(early_data, recent_data, stat_options, early_season, recent_season):
    """Create a statistical summary comparison table"""
    st.markdown("### Statistical Summary Comparison")
    
    # Create summary table
    summary_data = []
    
    for stat, label in stat_options.items():
        early_mean = early_data[stat].mean()
        recent_mean = recent_data[stat].mean()
        
        early_median = early_data[stat].median()
        recent_median = recent_data[stat].median()
        
        early_std = early_data[stat].std()
        recent_std = recent_data[stat].std()
        
        pct_change = calculate_percent_change(early_mean, recent_mean)
        
        summary_data.append({
            'Statistic': label,
            f'Mean ({early_season})': early_mean,
            f'Mean ({recent_season})': recent_mean,
            f'Median ({early_season})': early_median,
            f'Median ({recent_season})': recent_median,
            f'Std Dev ({early_season})': early_std,
            f'Std Dev ({recent_season})': recent_std,
            'Change %': pct_change
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Format the summary table
    formatted_df = format_summary_table(summary_df)
    
    # Display the formatted table
    st.dataframe(formatted_df, use_container_width=True)

def format_summary_table(summary_df):
    """Format the summary table for display"""
    formatted_df = summary_df.copy()
    
    # Apply formatting based on column type
    for col in formatted_df.columns:
        if 'Mean' in col or 'Median' in col:
            if 'FG %' in formatted_df['Statistic'].values or 'FT %' in formatted_df['Statistic'].values or '3-Point %' in formatted_df['Statistic'].values:
                # Format as percentage
                formatted_df[col] = formatted_df.apply(
                    lambda row: f"{row[col]:.3f}" if "%" in row['Statistic'] else f"{row[col]:.1f}", 
                    axis=1
                )
            else:
                # Format as number with 1 decimal place
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.1f}")
        elif 'Std Dev' in col:
            # Format standard deviation
            formatted_df[col] = formatted_df.apply(
                lambda row: f"{row[col]:.3f}" if "%" in row['Statistic'] else f"{row[col]:.1f}", 
                axis=1
            )
        elif 'Change' in col:
            # Format percentage change with sign
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:+.1f}%")
    
    return formatted_df

def create_gini_comparison(early_data, recent_data, stat_options, early_season, recent_season):
    """Create a Gini coefficient comparison visualization"""
    # Calculate Gini coefficients
    gini_data = []
    
    for stat, label in stat_options.items():
        early_gini = calculate_gini(early_data[stat].values)
        recent_gini = calculate_gini(recent_data[stat].values)
        
        gini_change = recent_gini - early_gini
        
        gini_data.append({
            'Statistic': label,
            f'Gini ({early_season})': early_gini,
            f'Gini ({recent_season})': recent_gini,
            'Change': gini_change
        })
    
    gini_df = pd.DataFrame(gini_data)
    
    # Create bar chart to compare Gini coefficients
    fig = go.Figure()
    
    # Add bars for early season
    fig.add_trace(go.Bar(
        y=gini_df['Statistic'],
        x=gini_df[f'Gini ({early_season})'],
        name=f'{early_season}',
        orientation='h'
    ))
    
    # Add bars for recent season
    fig.add_trace(go.Bar(
        y=gini_df['Statistic'],
        x=gini_df[f'Gini ({recent_season})'],
        name=f'{recent_season}',
        orientation='h'
    ))
    
    # Update layout
    fig.update_layout(
        title="Gini Coefficient by Statistic",
        xaxis_title="Gini Coefficient",
        yaxis_title="Statistic",
        barmode='group',
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_distribution_insights(early_data, recent_data, stat_options, early_season, recent_season):
    """Generate and display insights from the distribution analysis"""
    st.subheader("Key Insights")
    
    # Create summary data for calculating insights
    summary_data = []
    for stat, label in stat_options.items():
        early_mean = early_data[stat].mean()
        recent_mean = recent_data[stat].mean()
        early_std = early_data[stat].std()
        recent_std = recent_data[stat].std()
        pct_change = calculate_percent_change(early_mean, recent_mean)
        std_change = calculate_percent_change(early_std, recent_std)
        
        # Calculate Gini coefficients
        early_gini = calculate_gini(early_data[stat].values)
        recent_gini = calculate_gini(recent_data[stat].values)
        gini_change = recent_gini - early_gini
        
        summary_data.append({
            'Statistic': label,
            'Change %': pct_change,
            'Std Change %': std_change,
            'Gini Change': gini_change
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Generate insights - FIX 3: Safely handle Series and comparisons
    insights = []
    
    # Find statistic with biggest change
    if not summary_df.empty:
        max_change_idx = summary_df['Change %'].abs().idxmax()
        max_change_stat = summary_df.loc[max_change_idx, 'Statistic']
        max_change_val = summary_df.loc[max_change_idx, 'Change %']
        
        insights.append(f"{max_change_stat} has shown the largest change: {max_change_val:+.1f}% from {early_season} to {recent_season}.")
        
        # Find stat with most change in inequality
        max_gini_change_idx = summary_df['Gini Change'].abs().idxmax()
        max_gini_change_stat = summary_df.loc[max_gini_change_idx, 'Statistic']
        max_gini_change_val = summary_df.loc[max_gini_change_idx, 'Gini Change']
        
        inequality_direction = "more unequal" if max_gini_change_val > 0 else "more equal"
        insights.append(f"The distribution of {max_gini_change_stat} has become {inequality_direction} (Gini change: {max_gini_change_val:+.3f}).")
        
        # Find stat with biggest change in spread
        max_std_change_idx = summary_df['Std Change %'].abs().idxmax()
        max_std_change_stat = summary_df.loc[max_std_change_idx, 'Statistic']
        max_std_change_val = summary_df.loc[max_std_change_idx, 'Std Change %']
        
        spread_direction = "wider" if max_std_change_val > 0 else "narrower"
        insights.append(f"The spread of {max_std_change_stat} has grown {spread_direction} by {abs(max_std_change_val):.1f}%.")
    
    for insight in insights:
        st.write(f"• {insight}")

def calculate_gini(array):
    """Calculate the Gini coefficient of a numpy array"""
    # Based on https://github.com/oliviaguest/gini
    # All values are treated equally, arrays must be 1d
    array = array.flatten()
    if np.amin(array) < 0:
        # Shift values to eliminate negative values
        array -= np.amin(array)
    
    # Values must be sorted
    array = np.sort(array)
    # Index for array
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements
    n = array.shape[0]
    
    if n <= 1 or np.sum(array) == 0:
        return 0
        
    # Gini coefficient
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))