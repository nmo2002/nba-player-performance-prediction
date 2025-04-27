import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess
import sys
import plotly.express as px
import plotly.graph_objects as go
import requests
from io import BytesIO
from PIL import Image
import re

def check_and_install_dependencies(verbose=False):
    """Check and install required dependencies if needed"""
    # Dictionary mapping install names (keys) to import names (values)
    package_mapping = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'plotly': 'plotly',
        'scikit-learn': 'sklearn',
        'nba_api': 'nba_api',
        'requests': 'requests',
        'networkx': 'networkx',
        'pillow': 'PIL',
        'python-louvain': 'community',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'statsmodels': 'statsmodels',
        'xgboost': 'xgboost',
        'seaborn': 'seaborn',
        'psutil': 'psutil',
        'pyvis': 'pyvis'  # Add this new dependency
    }
    
    if verbose:
        print("⚙️ Checking dependencies...")
    
    for install_name, import_name in package_mapping.items():
        try:
            __import__(import_name)
            if verbose:
                print(f"✅ {install_name} is installed")
        except ImportError:
            if verbose:
                print(f"📦 Installing {install_name}...")
            
            try:
                # Hide the pip output
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", install_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                # Verify installation worked
                __import__(import_name)
                if verbose:
                    print(f"✅ {install_name} installed successfully")
            except Exception as e:
                if verbose:
                    print(f"❌ Failed to install {install_name}: {str(e)}")
    
    if verbose:
        print("✅ All dependencies checked!")

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Player search and data functions
def search_player(player_df, search_term):
    """Search for players by name with improved fuzzy matching"""
    if not search_term:
        return player_df.head(10)
    
    search_lower = search_term.lower()
    
    # Calculate relevance score
    def calc_relevance(name):
        name_lower = name.lower()
        
        # Exact match
        if name_lower == search_lower:
            return 100
        
        # Name part exact match
        if search_lower in name_lower.split():
            return 90
        
        # Contains full search
        if search_lower in name_lower:
            return 80
        
        # Starts with search
        if name_lower.startswith(search_lower):
            return 70
        
        # Contains all search words
        search_words = search_lower.split()
        if all(word in name_lower for word in search_words):
            return 60
        
        # Contains any search word
        if any(word in name_lower for word in search_words):
            return 50
        
        return 0
    
    player_df['relevance'] = player_df['full_name'].apply(calc_relevance)
    results = player_df[player_df['relevance'] > 0].sort_values('relevance', ascending=False)
    
    return results.drop(columns=['relevance'])

def normalize_stats(stats_df, features, method='minmax'):
    """Normalize player statistics for fair comparisons"""
    normalized_df = stats_df.copy()
    
    for feature in features:
        if feature not in normalized_df.columns:
            continue
            
        values = normalized_df[feature].values
        
        if method == 'minmax':
            # Min-max scaling
            min_val, max_val = np.min(values), np.max(values)
            if max_val > min_val:
                normalized_df[f'{feature}_norm'] = (values - min_val) / (max_val - min_val)
            else:
                normalized_df[f'{feature}_norm'] = values
        
        elif method == 'zscore':
            # Z-score normalization
            mean_val, std_val = np.mean(values), np.std(values)
            if std_val > 0:
                normalized_df[f'{feature}_norm'] = (values - mean_val) / std_val
            else:
                normalized_df[f'{feature}_norm'] = values
        
        elif method == 'robust':
            # Robust scaling
            median_val = np.median(values)
            q1, q3 = np.percentile(values, 25), np.percentile(values, 75)
            iqr = q3 - q1
            if iqr > 0:
                normalized_df[f'{feature}_norm'] = (values - median_val) / iqr
            else:
                normalized_df[f'{feature}_norm'] = values
    
    return normalized_df

def calculate_similarity(player1_stats, player2_stats, features):
    """Calculate similarity between two players based on selected features"""
    # Extract feature values
    p1_values = player1_stats[features].values[0] if isinstance(player1_stats, pd.DataFrame) else player1_stats[features]
    p2_values = player2_stats[features].values[0] if isinstance(player2_stats, pd.DataFrame) else player2_stats[features]
    
    # Normalize using min-max scaling
    min_vals = np.min([p1_values, p2_values], axis=0)
    max_vals = np.max([p1_values, p2_values], axis=0)
    range_vals = max_vals - min_vals + 1e-10  # Avoid division by zero
    
    p1_norm = (p1_values - min_vals) / range_vals
    p2_norm = (p2_values - min_vals) / range_vals
    
    # Calculate Euclidean distance and convert to similarity score
    distance = np.sqrt(np.sum((p1_norm - p2_norm) ** 2))
    similarity = 100 * (1 / (1 + distance))
    
    return similarity

def format_stats_dataframe(df, precision=1):
    """Format a dataframe of player statistics with proper precision"""
    # Identify column types
    pct_columns = [col for col in df.columns if 'pct' in col.lower() or '%' in col]
    count_columns = [col for col in df.columns if col not in pct_columns and df[col].dtype != 'object']
    
    # Create format dictionary
    format_dict = {
        **{col: '{:.3f}' for col in pct_columns},
        **{col: f'{{:.{precision}f}}' for col in count_columns}
    }
    
    return df.style.format(format_dict)

# Visualization functions
def create_radar_chart(values, categories, title, player_name=None):
    """Create a radar chart for player statistics"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=player_name if player_name else 'Player'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title=title,
        height=400
    )
    
    return fig

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_player_image(player_name):
    """Get player image URL or return placeholder image"""
    # Format player name for URL: lowercase, replace spaces with hyphens
    try:
        formatted_name = re.sub(r'[^a-zA-Z0-9\s]', '', player_name).lower().replace(' ', '-')
        
        # Try NBA.com image format
        potential_urls = [
            f"https://cdn.nba.com/headshots/nba/latest/1040x760/{formatted_name}.png",
            f"https://cdn.nba.com/headshots/nba/latest/260x190/{formatted_name}.png"
        ]
        
        for url in potential_urls:
            response = requests.get(url, timeout=1.5)
            if response.status_code == 200 and len(response.content) > 1000:
                try:
                    Image.open(BytesIO(response.content))
                    return url
                except:
                    continue
    except Exception:
        pass
        
    # Return NBA logo as fallback
    return "https://cdn.nba.com/headshots/nba/latest/260x190/logoman.png"

# UI components
def render_sidebar():
    """Render the sidebar with navigation options"""
    st.sidebar.title("NBA Analytics")
    
    # Display logo
    st.sidebar.image("https://cdn.nba.com/logos/nba/nba-logoman-75-word_white.svg", width=200)
    
    # Navigation menu
    nav_options = [
        "Player Analysis",
        "Team Comparison",
        "Statistical Trends",
        "Performance Prediction",
        "Player Clustering",
        "Player Similarity"
    ]
    
    nav_selection = st.sidebar.radio("Navigation", nav_options)
    
    # Data refresh button
    st.sidebar.markdown("---")
    if st.sidebar.button("🏀 Fetch Latest NBA Data", 
                       use_container_width=True, 
                       key="fetch_nba_data_button",
                       help="Get the most current NBA stats directly from the official NBA API"):
        try:
            with st.sidebar.status("Fetching fresh NBA data from official API..."):
                # Import here to avoid circular imports
                from data_loader import fetch_and_refresh_nba_data
                
                # Clear any existing cache
                st.cache_data.clear()
                
                # Fetch fresh data
                player_df, stats_df, teams_df, _ = fetch_and_refresh_nba_data()
                
                if player_df is not None:
                    st.sidebar.success("✅ Latest NBA data successfully loaded!")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Failed to fetch NBA data. Check your internet connection.")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    This dashboard analyzes NBA player performance using data mining and machine learning techniques.
    
    Data is sourced from the official NBA API.
    
    Version: 1.0.0 (April 2025)
    """)
    
    return nav_selection

def apply_custom_styling():
    """Apply custom styling to the app - dark mode only"""
    # Dark theme values
    primary_color = "#4da6ff"       # Light blue
    secondary_color = "#ff9f45"     # Light orange (unused)
    background_color = "#121212"    # Very dark gray
    text_color = "#f0f0f0"          # Off-white
    card_background = "#1e1e1e"     # Dark gray
    sidebar_color = "#262626"       # Slightly lighter dark gray
    
    # Generate CSS
    st.markdown(f"""
    <style>
        /* Main layout */
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            background-color: {background_color};
        }}
        
        /* Force background color */
        .stApp {{
            background-color: {background_color} !important;
        }}
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Roboto', sans-serif !important;
            font-weight: 700 !important;
            color: {primary_color} !important;
        }}
        
        p, span, div, li {{
            color: {text_color} !important;
        }}
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background-color: {sidebar_color} !important;
        }}
        
        /* Cards */
        .player-card {{
            background-color: {card_background};
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            display: flex;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        
        .player-card-image {{
            width: 80px;
            height: 80px;
            object-fit: cover;
            border-radius: 50%;
        }}
        
        .player-card-content {{
            margin-left: 1rem;
            flex-grow: 1;
        }}
        
        .player-card-name {{
            font-weight: bold;
            font-size: 1.2rem;
            margin-bottom: 0.3rem;
            color: {primary_color} !important;
        }}
        
        .player-card-info {{
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
            opacity: 0.8;
        }}
        
        .player-card-stats {{
            font-size: 0.9rem;
        }}
        
        .player-card-stats span {{
            margin-right: 0.8rem;
            display: inline-block;
        }}
    </style>
    """, unsafe_allow_html=True)

def render_player_card(player_info, key_stats=None):
    """Render a styled player card with image and key stats"""
    # Extract player information
    if isinstance(player_info, pd.Series):
        player_name = player_info.get('full_name', 'Unknown Player')
        player_position = player_info.get('position', '')
        player_team = player_info.get('team', '')
    else:
        player_name = player_info.get('full_name', 'Unknown Player')
        player_position = player_info.get('position', '')
        player_team = player_info.get('team', '')
    
    img_url = get_player_image(player_name)
    
    # Create card HTML
    card_html = f"""
    <div class="player-card">
        <img src="{img_url}" class="player-card-image" alt="{player_name}">
        <div class="player-card-content">
            <div class="player-card-name">{player_name}</div>
            <div class="player-card-info">
                {player_position} | {player_team}
            </div>
    """
    
    # Add stats if provided
    if key_stats:
        card_html += '<div class="player-card-stats">'
        for stat_name, stat_value in key_stats.items():
            card_html += f'<span>{stat_name}: {stat_value}</span> '
        card_html += '</div>'
    
    # Close card
    card_html += """
        </div>
    </div>
    """
    
    # Render card
    st.markdown(card_html, unsafe_allow_html=True)