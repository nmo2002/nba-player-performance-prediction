import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import subprocess
import warnings
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# IMPORTANT: Page config must be the first Streamlit command
st.set_page_config(
    page_title="NBA Player Performance Analysis",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')

# Import utilities
from utils import check_and_install_dependencies, render_sidebar, apply_custom_styling, create_directory_if_not_exists
from data_loader import (
    load_player_data, load_stats_data, load_team_data, load_team_stats, 
    fetch_and_refresh_nba_data, NBA_API_AVAILABLE
)

# Set up components path
components_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'components')
create_directory_if_not_exists(components_dir)
sys.path.append(components_dir)

# Import components - keep original imports to avoid errors
from components import (
    render_player_analysis, render_team_comparison, render_predictions,
    render_trends, render_player_clustering, render_player_similarity
)

# Initialize performance settings in session state
if 'performance_settings' not in st.session_state:
    st.session_state.performance_settings = {
        'max_chart_points': 1000,
        'use_downsampling': True,
        'enable_animations': True
    }

# Optimized data loading with caching
@st.cache_data(ttl=3600, show_spinner=False)
def get_player_data():
    """Load player data with performance tracking"""
    start_time = time.time()
    try:
        player_df = load_player_data()
        # Optimize memory usage
        player_df = optimize_dataframe_memory(player_df)
        logger.info(f"Loaded player data in {time.time() - start_time:.2f} seconds")
        return player_df
    except Exception as e:
        logger.error(f"Error loading player data: {e}")
        st.error(f"Error loading player data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_stats_data(min_season=None):
    """Load stats data with filtering options"""
    start_time = time.time()
    try:
        stats_df = load_stats_data()
        
        # Apply filters if provided
        if min_season is not None and 'season' in stats_df.columns:
            stats_df = stats_df[stats_df['season'] >= min_season]
        
        # Optimize memory usage
        stats_df = optimize_dataframe_memory(stats_df)
        
        # Preprocess data
        stats_df = preprocess_stats(stats_df)
        
        logger.info(f"Loaded stats data in {time.time() - start_time:.2f} seconds, {len(stats_df)} rows")
        return stats_df
    except Exception as e:
        logger.error(f"Error loading stats data: {e}")
        st.error(f"Error loading stats data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_team_data():
    """Load team data with performance tracking"""
    start_time = time.time()
    try:
        teams_df = load_team_data()
        # Optimize memory usage
        teams_df = optimize_dataframe_memory(teams_df)
        logger.info(f"Loaded team data in {time.time() - start_time:.2f} seconds")
        return teams_df
    except Exception as e:
        logger.error(f"Error loading team data: {e}")
        st.error(f"Error loading team data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_team_stats_data():
    """Load team stats data with performance tracking"""
    start_time = time.time()
    try:
        team_stats_df = load_team_stats()
        # Optimize memory usage
        team_stats_df = optimize_dataframe_memory(team_stats_df)
        logger.info(f"Loaded team stats data in {time.time() - start_time:.2f} seconds")
        return team_stats_df
    except Exception as e:
        logger.error(f"Error loading team stats data: {e}")
        st.error(f"Error loading team stats data: {str(e)}")
        return pd.DataFrame(columns=[
            'team_id', 'team_name', 'wins', 'losses', 'win_pct', 
            'ppg', 'oppg', 'rpg', 'apg', 'spg', 'bpg', 
            'tpg', 'fg_pct', 'fg3_pct', 'ft_pct'
        ])

def optimize_dataframe_memory(df):
    """Reduce memory usage of dataframe by downcasting types"""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
        
    df_optimized = df.copy()
    
    # Downcast numeric columns
    for col in df.select_dtypes(include=['int']).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
        
    for col in df.select_dtypes(include=['float']).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
    
    # Convert object columns with few unique values to categories
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < len(df) * 0.5:  # Less than 50% unique values
            df_optimized[col] = df_optimized[col].astype('category')
    
    return df_optimized

def preprocess_stats(stats_df):
    """Preprocess stats data to avoid repeated calculations"""
    if stats_df is None or not isinstance(stats_df, pd.DataFrame) or stats_df.empty:
        return stats_df
        
    # Calculate commonly used metrics once
    if all(col in stats_df.columns for col in ['ppg', 'rpg', 'apg']):
        stats_df['combined_stats'] = stats_df['ppg'] + stats_df['rpg'] + stats_df['apg']
    
    # Add efficiency rating if not present and needed
    if 'efficiency_rating' not in stats_df.columns and all(col in stats_df.columns for col in ['ppg', 'rpg', 'apg', 'spg', 'bpg', 'tpg']):
        stats_df['efficiency_rating'] = (
            stats_df['ppg'] + stats_df['rpg'] + stats_df['apg'] + 
            stats_df['spg'] + stats_df['bpg'] - stats_df['tpg']
        )
    
    return stats_df

def create_fallback_team_stats(teams_df):
    """Create minimal team stats DataFrame from teams data"""
    if not isinstance(teams_df, pd.DataFrame) or teams_df.empty:
        return pd.DataFrame()
        
    # Handle team ID
    if 'team_id' not in teams_df.columns and 'id' in teams_df.columns:
        teams_df['team_id'] = teams_df['id']
    if 'team_id' not in teams_df.columns:
        return pd.DataFrame()
        
    # Get team name column
    name_col = next((col for col in ['full_name', 'nickname'] if col in teams_df.columns), None)
    
    # Create minimal team stats
    return pd.DataFrame({
        'team_id': teams_df['team_id'],
        'team_name': teams_df[name_col] if name_col else "Team " + teams_df['team_id'].astype(str),
        'wins': 0, 'losses': 0, 'win_pct': 0.0, 'ppg': 0.0, 'oppg': 0.0,
        'rpg': 0.0, 'apg': 0.0, 'spg': 0.0, 'bpg': 0.0, 'tpg': 0.0,
        'fg_pct': 0.0, 'fg3_pct': 0.0, 'ft_pct': 0.0
    })

def debug_data(player_df, stats_df, teams_df, team_stats_df=None):
    """Display debug information for troubleshooting"""
    with st.expander("Debug Data Issues", expanded=False):
        # DataFrame validation
        for name, df in [("player_df", player_df), ("stats_df", stats_df), 
                         ("teams_df", teams_df), ("team_stats_df", team_stats_df)]:
            if isinstance(df, pd.DataFrame):
                st.write(f"{name} columns:", df.columns.tolist() if not df.empty else [])
                if not df.empty:
                    st.dataframe(df.head(3))
            else:
                st.write(f"{name} is not a valid DataFrame")
        
        # Check for critical columns
        st.write("### Critical Column Check")
        for df_name, df, col_name in [
            ("teams_df", teams_df, "team_id"),
            ("stats_df", stats_df, "team_id"),
            ("stats_df", stats_df, "player_id")
        ]:
            if isinstance(df, pd.DataFrame) and not df.empty:
                st.write(f"'{col_name}' in {df_name}:", col_name in df.columns)
                if col_name in df.columns:
                    st.write(f"{df_name}['{col_name}'] dtype:", df[col_name].dtype)

def fix_dataframe_columns(df, original_name, target_name, potential_names=None):
    """Fix column names in dataframes for consistency"""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
        
    # Already has the target column
    if target_name in df.columns:
        return df
        
    # Direct rename if original name exists
    if original_name in df.columns:
        df = df.rename(columns={original_name: target_name})
        return df
        
    # Try potential alternative names
    if potential_names:
        for col in potential_names:
            if col in df.columns:
                df = df.rename(columns={col: target_name})
                return df
    
    return df

def ensure_numeric_column(df, column):
    """Ensure a column is numeric"""
    if not isinstance(df, pd.DataFrame) or df.empty or column not in df.columns:
        return df
        
    try:
        # Convert to string then numeric to handle mixed types
        df[column] = df[column].astype(str)
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column] = df[column].fillna(0).astype(int)
    except Exception as e:
        logger.error(f"Error converting {column}: {e}")
    
    return df

def render_performance_settings():
    """Render performance tuning settings in the sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Performance Settings")
        
        # Performance mode selection
        performance_mode = st.radio(
            "Performance Mode",
            options=["Balanced", "High Performance", "High Quality"],
            index=0,
            key="performance_mode",
            help="Controls the trade-off between performance and visual quality"
        )
        
        # Set performance variables based on selection
        if performance_mode == "High Performance":
            st.session_state.performance_settings = {
                'max_chart_points': 500,
                'use_downsampling': True,
                'enable_animations': False
            }
        elif performance_mode == "High Quality":
            st.session_state.performance_settings = {
                'max_chart_points': 5000,
                'use_downsampling': False,
                'enable_animations': True
            }
        else:  # Balanced
            st.session_state.performance_settings = {
                'max_chart_points': 1000,
                'use_downsampling': True,
                'enable_animations': True
            }
        
        # Advanced performance options
        with st.expander("Advanced Performance Options", expanded=False):
            # Allow manual override of settings
            st.slider(
                "Max Chart Points", 
                min_value=100, 
                max_value=10000, 
                value=st.session_state.performance_settings['max_chart_points'],
                step=100,
                key="custom_max_chart_points",
                help="Maximum number of data points to display in charts"
            )
            st.session_state.performance_settings['max_chart_points'] = st.session_state.custom_max_chart_points
            
            st.checkbox(
                "Enable Downsampling", 
                value=st.session_state.performance_settings['use_downsampling'],
                key="custom_downsampling",
                help="Reduces data points in large datasets for faster rendering"
            )
            st.session_state.performance_settings['use_downsampling'] = st.session_state.custom_downsampling
            
            st.checkbox(
                "Enable Animations", 
                value=st.session_state.performance_settings['enable_animations'],
                key="custom_animations",
                help="Enables chart animations at the cost of performance"
            )
            st.session_state.performance_settings['enable_animations'] = st.session_state.custom_animations

def main():
    """Main application function with progressive loading"""
    # Initialize performance monitoring
    start_time = time.time()
    
    # Handle NBA API availability
    if NBA_API_AVAILABLE:
        # Check if data files exist
        data_files_exist = all(os.path.exists(f) for f in [
            'data/players.csv', 'data/player_stats.csv', 'data/teams.csv'
        ])
        
        if not data_files_exist:
            st.warning("No NBA data files found. Fetching data automatically...")
            try:
                with st.spinner("Fetching NBA data... This may take a minute."):
                    player_df, stats_df, teams_df, team_stats_df = fetch_and_refresh_nba_data()
                    if player_df is not None and not player_df.empty:
                        st.success("✅ NBA data successfully loaded!")
                        st.rerun()
                    else:
                        st.error("Failed to fetch NBA data automatically.")
                        st.info("Please click 'Fetch Latest NBA Data' in the sidebar.")
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
    else:
        st.error("NBA API is not installed correctly. Try installing with: pip install nba_api")
        if st.button("Install NBA API"):
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "nba_api==1.9.0"])
                st.success("NBA API installed! Please restart the app.")
                st.rerun()
            except Exception as e:
                st.error(f"Installation failed: {str(e)}")
    
    # Render sidebar and get navigation (moved earlier to show while data loads)
    nav_selection = render_sidebar()
    
    # Add performance settings to sidebar
    render_performance_settings()
    
    # Apply styling
    apply_custom_styling()
    
    # Progressive and selective data loading based on selected component
    with st.spinner('Loading data...'):
        # Always load player data (small and used by all components)
        player_df = get_player_data()
        
        # Selective loading for other data based on component
        if nav_selection == "Player Analysis":
            teams_df = get_team_data()
            # For player analysis, we need all stats
            stats_df = get_stats_data()
            team_stats_df = pd.DataFrame()  # Not needed for this component
        
        elif nav_selection == "Team Comparison":
            teams_df = get_team_data()
            team_stats_df = get_team_stats_data()
            # For team comparison, load recent seasons
            stats_df = get_stats_data(min_season=2018)
        
        elif nav_selection == "Performance Prediction":
            teams_df = get_team_data()
            team_stats_df = pd.DataFrame()  # Not needed
            # For prediction, we need all stats for training models
            stats_df = get_stats_data()
        
        elif nav_selection == "Statistical Trends":
            teams_df = get_team_data()
            team_stats_df = pd.DataFrame()  # Not needed
            # For trends, we need all historical data
            stats_df = get_stats_data()
        
        elif nav_selection == "Player Clustering":
            teams_df = get_team_data()
            team_stats_df = pd.DataFrame()  # Not needed
            # For clustering, focus on recent data
            stats_df = get_stats_data(min_season=2019)
        
        elif nav_selection == "Player Similarity":
            teams_df = get_team_data()
            team_stats_df = pd.DataFrame()  # Not needed
            # For similarity, we need recent seasons
            stats_df = get_stats_data(min_season=2018)
        
        else:
            # Default: load all data
            teams_df = get_team_data()
            stats_df = get_stats_data()
            team_stats_df = pd.DataFrame()
    
    # Create fallback team stats if needed
    if nav_selection == "Team Comparison" and isinstance(team_stats_df, pd.DataFrame) and team_stats_df.empty and isinstance(teams_df, pd.DataFrame) and not teams_df.empty:
        team_stats_df = create_fallback_team_stats(teams_df)
        st.warning("⚠️ Using placeholder team statistics as data couldn't be fetched from NBA API.")
    
    # Check if all dataframes are empty
    if all(isinstance(df, pd.DataFrame) and df.empty for df in [player_df, stats_df, teams_df]):
        st.error("⚠️ No NBA data available. Please click 'Fetch Latest NBA Data' in the sidebar.")
        with st.expander("Data Source Information"):
            st.write("""
            This application uses data from the official NBA API.
            Please click 'Fetch Latest NBA Data' in the sidebar to download current data.
            """)
        return
    
    # Fix dataframe columns for consistency
    # Teams dataframe fixes
    teams_df = fix_dataframe_columns(teams_df, 'id', 'team_id')
    teams_df = ensure_numeric_column(teams_df, 'team_id')
    
    # Stats dataframe fixes
    stats_df = fix_dataframe_columns(stats_df, 'id', 'player_id', 
                                    ['player', 'player_name', 'player_number', 'nba_id'])
    stats_df = fix_dataframe_columns(stats_df, 'team', 'team_id', 
                                    ['team_name', 'team_abbr', 'team_code', 'franchise'])
    stats_df = ensure_numeric_column(stats_df, 'team_id')
    stats_df = ensure_numeric_column(stats_df, 'player_id')  # Ensure player_id is numeric
    
    # Team stats dataframe fixes
    if not team_stats_df.empty:
        team_stats_df = fix_dataframe_columns(team_stats_df, 'id', 'team_id')
        team_stats_df = ensure_numeric_column(team_stats_df, 'team_id')
    
    # Ensure all player_id and team_id are consistently typed to avoid merge issues
    if not player_df.empty and 'id' in player_df.columns:
        player_df['id'] = player_df['id'].astype('int64')
    
    # Store recent season in session state for future use
    if not stats_df.empty and 'season' in stats_df.columns:
        st.session_state.recent_season = int(stats_df['season'].max())
    
    # Navigation components mapping
    components = {
        "Player Analysis": (render_player_analysis, [player_df, stats_df, teams_df]),
        "Performance Prediction": (render_predictions, [player_df, stats_df, teams_df]),
        "Statistical Trends": (render_trends, [player_df, stats_df, teams_df]),
        "Team Comparison": (render_team_comparison, [player_df, stats_df, teams_df, team_stats_df]),
        "Player Clustering": (render_player_clustering, [player_df, stats_df, teams_df]),
        "Player Similarity": (render_player_similarity, [player_df, stats_df, teams_df])
    }
    
    # Handle component rendering
    if nav_selection in components:
        component_func, args = components[nav_selection]
        
        # Data validation for the component
        required_df = args[0] if len(args) > 0 else None
        if isinstance(required_df, pd.DataFrame) and not required_df.empty:
            component_func(*args)
        else:
            st.error(f"Data required for {nav_selection} is not available. Please fetch NBA data first.")
    
    # Footer
    st.markdown("---")
    st.markdown("### NBA Player Performance Analysis Dashboard")
    st.markdown("Data Mining and Machine Learning Project")
    
    # Add performance metrics at the bottom
    elapsed_time = time.time() - start_time
    st.markdown(f"<small>Page rendered in {elapsed_time:.2f} seconds</small>", unsafe_allow_html=True)

    # Data attribution
    with st.expander("Data Source Information"):
        st.write("""
        This application uses data from the official NBA API when available.
        If the NBA API is not accessible, previously saved data is used as a fallback.
        
        To enable real NBA data, install the NBA API package:
        ```
        pip install nba_api==1.9.0
        ```
        
        **Note**: NBA API 1.9+ works best with Python 3.12 or earlier.
        """)

# Run the application
if __name__ == "__main__":
    try:
        check_and_install_dependencies(verbose=False)  # Set to False to hide output
        main()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down NBA dashboard...")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Application error: {e}")