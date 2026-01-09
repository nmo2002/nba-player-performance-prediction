import pandas as pd
import numpy as np
import os
import time
import streamlit as st
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def get_data_path(filename):
    """Return an absolute path inside the data directory."""
    return os.path.join(DATA_DIR, filename)

# Try to import NBA API - without problematic timeout settings
try:
    from nba_api.stats.static import players, teams
    from nba_api.stats.endpoints import (
        playercareerstats, 
        commonplayerinfo, 
        leaguegamefinder,
        leaguedashplayerstats,
        teamyearbyyearstats,
        leaguedashteamstats
    )
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
except Exception:
    NBA_API_AVAILABLE = False

# Season handling functions
def get_current_season():
    """Get the current NBA season year"""
    now = datetime.now()
    # NBA season spans two years, it starts in October and ends in June of the next year
    return now.year + 1 if now.month >= 10 else now.year

def format_season(year):
    """Format a year into NBA season format (e.g., 2023 -> '2022-23')"""
    return f"{year-1}-{str(year)[-2:]}"

# Data loading functions
def load_player_data():
    """Load NBA player data, trying API first with CSV as backup"""
    # Try NBA API first if available
    if NBA_API_AVAILABLE:
        try:
            player_df = fetch_players_from_api()
            return player_df
        except Exception as e:
            players_path = get_data_path('players.csv')
            if os.path.exists(players_path):
                player_df = pd.read_csv(players_path)
                return player_df
            raise e
    
    # Use CSV if API not available
    try:
        player_df = pd.read_csv(get_data_path('players.csv'))
        return player_df
    except FileNotFoundError:
        return pd.DataFrame()

def load_stats_data():
    """Load multi-season player statistics"""
    # Check if API is available
    if NBA_API_AVAILABLE:
        try:
            # Check if we've already fetched stats today
            multiseason_path = get_data_path('player_stats_multiseason.csv')
            if os.path.exists(multiseason_path):
                stats_age = time.time() - os.path.getmtime(multiseason_path)
                if stats_age < 86400:  # Less than a day old
                    return pd.read_csv(multiseason_path)
            
            # Fetch fresh multi-season data
            stats_df = fetch_multiseason_stats()
            if not stats_df.empty:
                return stats_df
        except Exception:
            pass
    
    # Try to load saved data
    for file_path in [get_data_path('player_stats_multiseason.csv'), get_data_path('player_stats.csv')]:
        if os.path.exists(file_path):
            try:
                return pd.read_csv(file_path)
            except Exception:
                pass
    
    # No data available
    return pd.DataFrame()

def load_team_data():
    """Load NBA team data, trying API first with CSV as backup"""
    # Try NBA API first if available
    if NBA_API_AVAILABLE:
        try:
            # Get teams from NBA API
            all_teams = teams.get_teams()
            team_df = pd.DataFrame(all_teams)
            
            # Add team_id column for consistency
            team_df['team_id'] = team_df['id']
            
            # Save for future fallback
            team_df.to_csv(get_data_path('teams.csv'), index=False)
            return team_df
        except Exception:
            pass
    
    # Try to load saved data
    teams_path = get_data_path('teams.csv')
    if os.path.exists(teams_path):
        try:
            team_df = pd.read_csv(teams_path)
            # Ensure team_id exists for consistency
            if 'team_id' not in team_df.columns and 'id' in team_df.columns:
                team_df['team_id'] = team_df['id']
            return team_df
        except Exception:
            pass
    
    # No data available
    return pd.DataFrame()

def load_team_stats():
    """Load multi-season team stats"""
    # Check if API is available
    if NBA_API_AVAILABLE:
        try:
            # Check if we already have recent multi-season team stats
            multiseason_path = get_data_path('team_stats_multiseason.csv')
            if os.path.exists(multiseason_path):
                stats_age = time.time() - os.path.getmtime(multiseason_path)
                if stats_age < 86400:  # Less than a day old
                    team_stats_df = pd.read_csv(multiseason_path)
                    if 'team_id' in team_stats_df.columns:
                        return team_stats_df
            
            # Fetch fresh team stats
            team_stats_df = fetch_multiseason_team_stats()
            if not team_stats_df.empty:
                return team_stats_df
        except Exception:
            pass
    
    # Try to load saved multi-season stats first
    for file_path in [get_data_path('team_stats_multiseason.csv'), get_data_path('team_stats.csv')]:
        if os.path.exists(file_path):
            try:
                team_stats_df = pd.read_csv(file_path)
                # Fix team_id if missing
                if 'team_id' not in team_stats_df.columns and 'id' in team_stats_df.columns:
                    team_stats_df['team_id'] = team_stats_df['id']
                return team_stats_df
            except Exception:
                pass
    
    # No data available - return empty DataFrame with expected columns
    return pd.DataFrame(columns=[
        'team_id', 'team_name', 'season', 'wins', 'losses', 'win_pct', 
        'ppg', 'oppg', 'rpg', 'apg', 'spg', 'bpg', 
        'tpg', 'fg_pct', 'fg3_pct', 'ft_pct'
    ])

# Stats calculation functions
def calculate_efficiency(df):
    """Calculate a player's efficiency rating from season stats"""
    # Handle DataFrame
    if isinstance(df, pd.DataFrame):
        try:
            efficiency = (df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK'] - df['TOV']) / df['GP']
            return efficiency.round(1)
        except Exception:
            try:
                efficiency = (df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK']) / df['GP']
                return efficiency.round(1)
            except Exception:
                return pd.Series([0.0] * len(df), index=df.index)
    
    # Handle Series/row
    if 'GP' not in df or df['GP'] <= 0:
        return 0.0
    
    # Get stats with safe fallbacks
    pts = df.get('PTS', df.get('ppg', 0))
    reb = df.get('REB', df.get('rpg', 0))
    ast = df.get('AST', df.get('apg', 0))
    stl = df.get('STL', df.get('spg', 0))
    blk = df.get('BLK', df.get('bpg', 0))
    tov = df.get('TOV', 0)
    games = df.get('GP', df.get('games_played', 1))
    
    efficiency = (pts + reb + ast + stl + blk - tov) / max(1, games)
    return round(efficiency, 1)

# API fetching functions
def fetch_players_from_api():
    """Fetch current player data for reference"""
    # Get current player list
    active_players = players.get_active_players()
    player_df = pd.DataFrame(active_players)
    
    # Filter and restructure
    player_df = player_df[['id', 'full_name', 'first_name', 'last_name']]
    player_df['is_active'] = True
    
    # Save to CSV
    player_df.to_csv(get_data_path('players.csv'), index=False)
    return player_df

def fetch_multiseason_stats(seasons_to_fetch=5):
    """Fetch player statistics for multiple seasons using league dashboard"""
    if not NBA_API_AVAILABLE:
        return pd.DataFrame()
    
    # Get current season for reference
    current_season = get_current_season()
    
    # Calculate seasons to fetch
    season_years = list(range(current_season - seasons_to_fetch + 1, current_season + 1))
    formatted_seasons = [format_season(year) for year in season_years]
    
    all_season_stats = []
    
    # Use streamlit progress bar if available
    try:
        progress_bar = st.progress(0.0)
        use_streamlit = True
    except:
        use_streamlit = False
    
    # Fetch one season at a time
    for i, season in enumerate(formatted_seasons):
        try:
            # Use league dashboard to get all players for this season at once
            league_dash = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed='PerGame',
                measure_type_detailed_defense='Base',
                season_type_all_star='Regular Season'
            )
            
            season_data = league_dash.get_data_frames()[0]
            if season_data.empty:
                continue
            
            # Convert season format to year
            season_year = int(season[:4]) + 1
            # Build standardized DataFrame for this season
            cols = {
                'PLAYER_ID': 'player_id',
                'TEAM_ID': 'team_id',
                'GP': 'games_played',
                'PTS': 'ppg',
                'REB': 'rpg',
                'AST': 'apg',
                'STL': 'spg',
                'BLK': 'bpg',
                'FG_PCT': 'fg_pct',
                'FG3_PCT': 'fg3_pct',
                'FT_PCT': 'ft_pct',
                'MIN': 'minutes',
                'TOV': 'tov'
            }
            season_subset = season_data[list(cols.keys())].rename(columns=cols)
            season_subset['season'] = season_year
            efficiency_inputs = season_data[['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'GP']]
            season_subset['player_efficiency'] = calculate_efficiency(efficiency_inputs)
            all_season_stats.append(season_subset)
            
        except Exception:
            continue
        
        # Update progress
        if use_streamlit:
            progress_bar.progress((i + 1) / len(formatted_seasons))
    
    # Convert to DataFrame
    if all_season_stats:
        stats_df = pd.concat(all_season_stats, ignore_index=True)
        
        # Sort by player_id and season
        stats_df.sort_values(['player_id', 'season'], inplace=True)
        
        # Save to multi-season file
        stats_df.to_csv(get_data_path('player_stats_multiseason.csv'), index=False)
        
        # Also save latest season for backwards compatibility
        latest_season = stats_df['season'].max()
        stats_df[stats_df['season'] == latest_season].to_csv(get_data_path('player_stats.csv'), index=False)
        
        return stats_df
    
    return pd.DataFrame()

def fetch_multiseason_team_stats(seasons_to_fetch=5):
    """Fetch team statistics for multiple seasons using league dashboard"""
    if not NBA_API_AVAILABLE:
        return pd.DataFrame()
    
    # Get current season for reference
    current_season = get_current_season()
    
    # Calculate seasons to fetch
    season_years = list(range(current_season - seasons_to_fetch + 1, current_season + 1))
    formatted_seasons = [format_season(year) for year in season_years]
    
    all_team_seasons = []
    
    # Use streamlit progress bar if available
    try:
        progress_bar = st.progress(0.0)
        use_streamlit = True
    except:
        use_streamlit = False
    
    # Fetch one season at a time
    for i, season in enumerate(formatted_seasons):
        try:
            # Use league dashboard to get all teams for this season at once
            league_dash = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                per_mode_detailed='PerGame',
                measure_type_detailed_defense='Base',
                season_type_all_star='Regular Season'
            )
            
            season_data = league_dash.get_data_frames()[0]
            if season_data.empty:
                continue
            
            # Convert season format to year
            season_year = int(season[:4]) + 1
            cols = {
                'TEAM_ID': 'team_id',
                'TEAM_NAME': 'team_name',
                'W': 'wins',
                'L': 'losses',
                'W_PCT': 'win_pct',
                'PTS': 'ppg',
                'REB': 'rpg',
                'AST': 'apg',
                'STL': 'spg',
                'BLK': 'bpg',
                'TOV': 'tpg',
                'FG_PCT': 'fg_pct',
                'FG3_PCT': 'fg3_pct',
                'FT_PCT': 'ft_pct',
                'PLUS_MINUS': 'plus_minus'
            }
            season_subset = season_data[list(cols.keys())].rename(columns=cols)
            season_subset['season'] = season_year
            season_subset['oppg'] = season_subset['ppg'] - season_subset['plus_minus']
            season_subset.drop(columns=['plus_minus'], inplace=True)
            season_subset = season_subset[season_subset['team_id'] != 0]
            all_team_seasons.append(season_subset)
            
        except Exception:
            continue
        
        # Update progress
        if use_streamlit:
            progress_bar.progress((i + 1) / len(formatted_seasons))
    
    # Convert to DataFrame
    if all_team_seasons:
        team_stats_df = pd.concat(all_team_seasons, ignore_index=True)
        
        # Make sure team_id is the correct type
        team_stats_df['team_id'] = team_stats_df['team_id'].astype(int)
        
        # Sort by team_id and season
        team_stats_df.sort_values(['team_id', 'season'], inplace=True)
        
        # Save to multi-season file
        team_stats_df.to_csv(get_data_path('team_stats_multiseason.csv'), index=False)
        
        # Also save latest season for backwards compatibility
        latest_season = team_stats_df['season'].max()
        team_stats_df[team_stats_df['season'] == latest_season].to_csv(get_data_path('team_stats.csv'), index=False)
        
        return team_stats_df
    
    # Return empty DataFrame with expected columns
    return pd.DataFrame(columns=[
        'team_id', 'team_name', 'season', 'wins', 'losses', 'win_pct', 
        'ppg', 'oppg', 'rpg', 'apg', 'spg', 'bpg', 
        'tpg', 'fg_pct', 'fg3_pct', 'ft_pct'
    ])

def fetch_and_refresh_nba_data(seasons_to_fetch=5):
    """
    Force refresh all NBA data directly from the official NBA API.
    
    Args:
        seasons_to_fetch: Number of seasons to fetch (default 5)
    
    Returns:
        tuple: (player_df, stats_df, teams_df, team_stats_df) or (None, None, None, None) if failed
    """
    if not NBA_API_AVAILABLE:
        st.error("NBA API not installed. Please run: pip install nba_api==1.9.0")
        return None, None, None, None
    
    try:
        # 1. Get teams first
        all_teams = teams.get_teams()
        teams_df = pd.DataFrame(all_teams)
        teams_df['team_id'] = teams_df['id']  # Add team_id for consistency
        teams_df.to_csv(get_data_path('teams.csv'), index=False)
        
        # 2. Get multi-season player stats
        stats_df = fetch_multiseason_stats(seasons_to_fetch)
        if stats_df.empty:
            return None, None, None, None
        
        # Get player IDs from stats
        player_ids = stats_df['player_id'].unique()
        
        # 3. Get player details to match stats
        all_players = players.get_active_players()
        player_df = pd.DataFrame([p for p in all_players if p['id'] in player_ids])
        player_df['is_active'] = True
        player_df.to_csv(get_data_path('players.csv'), index=False)
        
        # 4. Get multi-season team stats
        team_stats_df = fetch_multiseason_team_stats(seasons_to_fetch)
        
        # Create fallback if needed
        if team_stats_df.empty or 'team_id' not in team_stats_df.columns:
            team_stats_df = pd.DataFrame({
                'team_id': teams_df['id'],
                'team_name': teams_df['full_name'],
                'season': get_current_season(),
                'wins': 0, 'losses': 0, 'win_pct': 0.0,
                'ppg': 0.0, 'oppg': 0.0, 'rpg': 0.0, 'apg': 0.0,
                'spg': 0.0, 'bpg': 0.0, 'tpg': 0.0,
                'fg_pct': 0.0, 'fg3_pct': 0.0, 'ft_pct': 0.0
            })
            team_stats_df.to_csv(get_data_path('team_stats.csv'), index=False)
        
        return player_df, stats_df, teams_df, team_stats_df
    
    except Exception as e:
        st.error(f"Failed to refresh NBA data: {str(e)}")
        return None, None, None, None

# Initialize the NBA API connection
if NBA_API_AVAILABLE:
    print("✅ NBA API is available. Data will be fetched from the official NBA stats.")
else:
    print("❌ NBA API is not available. Install with: pip install nba_api==1.9.0")
