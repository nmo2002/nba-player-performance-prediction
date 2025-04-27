import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import search_player, create_radar_chart, format_stats_dataframe, get_player_image

# ======================= Helper Functions =======================
def safe_team_lookup(player_stats, teams_df):
    """Safely look up team name from team_id with improved matching"""
    try:
        if not player_stats.empty and not teams_df.empty:
            latest_stats = player_stats.sort_values('season', ascending=False).iloc[0]
            
            if 'team_id' in latest_stats and pd.notna(latest_stats['team_id']):
                # Get the team ID and try various formats for matching
                team_id_raw = latest_stats['team_id']
                team_id_formats = [
                    str(team_id_raw).strip(),                  # Basic string
                    str(int(float(team_id_raw))),              # As integer (no decimal)
                    str(team_id_raw).split('.')[0],            # Remove decimal part
                    f"{int(float(team_id_raw)):08d}"           # Format as 8-digit number
                ]
                
                # Try matching with team dataframe
                for team_id_fmt in team_id_formats:
                    # Check ID column
                    if 'id' in teams_df.columns:
                        team_matches = teams_df[teams_df['id'].astype(str).str.strip() == team_id_fmt]
                        if not team_matches.empty:
                            return team_matches.iloc[0]['full_name']
                    
                    # Check team_id column
                    if 'team_id' in teams_df.columns:
                        team_matches = teams_df[teams_df['team_id'].astype(str).str.strip() == team_id_fmt]
                        if not team_matches.empty:
                            return team_matches.iloc[0]['full_name']
                
                # Fallback to NBA API team mapping
                nba_teams = {
                    1610612737: "Atlanta Hawks", 1610612738: "Boston Celtics",
                    1610612739: "Cleveland Cavaliers", 1610612740: "New Orleans Pelicans",
                    1610612741: "Chicago Bulls", 1610612742: "Dallas Mavericks",
                    1610612743: "Denver Nuggets", 1610612744: "Golden State Warriors",
                    1610612745: "Houston Rockets", 1610612746: "LA Clippers",
                    1610612747: "Los Angeles Lakers", 1610612748: "Miami Heat",
                    1610612749: "Milwaukee Bucks", 1610612750: "Minnesota Timberwolves",
                    1610612751: "Brooklyn Nets", 1610612752: "New York Knicks",
                    1610612753: "Orlando Magic", 1610612754: "Indiana Pacers",
                    1610612755: "Philadelphia 76ers", 1610612756: "Phoenix Suns",
                    1610612757: "Portland Trail Blazers", 1610612758: "Sacramento Kings",
                    1610612759: "San Antonio Spurs", 1610612760: "Oklahoma City Thunder",
                    1610612761: "Toronto Raptors", 1610612762: "Utah Jazz",
                    1610612763: "Memphis Grizzlies", 1610612764: "Washington Wizards",
                    1610612765: "Detroit Pistons", 1610612766: "Charlotte Hornets"
                }
                
                team_id_int = int(float(team_id_raw))
                if team_id_int in nba_teams:
                    return nba_teams[team_id_int]
                
                return f"Team ID: {team_id_raw}"
    except Exception as e:
        print(f"Error getting team name: {e}")
    return "No team"

def create_stat_chart(stats_df, x_col, value_cols, value_names, title, y_label=None, percentage=False):
    """Create time series chart for player stats"""
    # Prepare data and ensure seasons are integers
    stats_df = stats_df.copy()
    if x_col == 'season':
        stats_df['season'] = stats_df['season'].astype(int)
    
    data = stats_df[[x_col] + value_cols].melt(
        id_vars=[x_col], value_vars=value_cols, var_name='Metric', value_name='Value'
    )
    
    # Map stat names
    if value_names:
        name_map = {value_cols[i]: name for i, name in enumerate(value_names)}
        data['Metric'] = data['Metric'].map(name_map)
    
    # Create chart
    fig = px.line(
        data, x=x_col, y='Value', color='Metric', markers=True,
        title=title, labels={x_col: x_col.capitalize(), 'Value': y_label or 'Value'}
    )
    
    # Format x-axis for seasons
    if x_col == 'season':
        season_values = sorted(stats_df[x_col].unique())
        fig.update_layout(
            xaxis=dict(
                tickmode='array', 
                tickvals=season_values,
                ticktext=[str(s) for s in season_values],
                tickformat="d"
            )
        )
    
    # Format
    fig.update_layout(
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    if percentage:
        fig.update_layout(yaxis=dict(tickformat='.0%'))
        
    return fig

def get_valid_columns(df, required_cols):
    """Get only columns that exist in dataframe"""
    return [col for col in required_cols if col in df.columns]

def calculate_true_shooting(row):
    """Safely calculate true shooting percentage"""
    try:
        if all(k in row for k in ['ppg', 'fg_pct', 'ft_pct']):
            ppg, fg_pct, ft_pct = row['ppg'], row['fg_pct'], row['ft_pct']
            if fg_pct > 0 and ft_pct > 0:
                return ppg / (2 * (ppg / fg_pct * 0.44 + ppg / ft_pct * 0.44))
        return 0
    except:
        return 0

# ======================= Main Render Function =======================
def render_player_analysis(player_df, stats_df, teams_df):
    """Render the player analysis section"""
    try:
        st.title("NBA Player Analysis")
        st.write("Analyze individual player statistics and performance trends")
        
        # Player search UI
        col1, col2 = st.columns([3, 1])
        with col1:
            player_search = st.text_input("Search for a player", placeholder="Enter player name (e.g., LeBron, Curry)...")
        with col2:
            position_filter = st.multiselect("Position Filter", ["PG", "SG", "SF", "PF", "C"], key="pos_filter_player") if 'position' in player_df.columns else []
        
        # Apply filters
        filtered_df = player_df[player_df['position'].isin(position_filter)] if position_filter and 'position' in player_df.columns else player_df
        
        # Search with error handling
        try:
            filtered_players = search_player(filtered_df, player_search)
        except Exception as e:
            print(f"Error in search_player: {e}")
            filtered_players = filtered_df[filtered_df['full_name'].str.contains(player_search, case=False, na=False)] if player_search else filtered_df
        
        if filtered_players.empty:
            st.info("No players found. Try different search terms or position filters.")
            return
            
        # Display search results
        st.subheader(f"Player Search Results ({len(filtered_players)} players)")
        recent_season = stats_df['season'].max() if not stats_df.empty else None
        
        if recent_season:
            # Get recent stats and merge with players
            recent_stats = stats_df[stats_df['season'] == recent_season]
            player_stats = filtered_players[['id', 'full_name']].merge(
                recent_stats[['player_id', 'ppg', 'rpg', 'apg', 'fg_pct', 'team_id']], 
                left_on='id', right_on='player_id', how='left'
            )
            
            # Add team info
            if 'team_id' in player_stats.columns and not teams_df.empty:
                try:
                    player_stats['team_id_str'] = player_stats['team_id'].astype(str)
                    teams_df_copy = teams_df.copy()
                    teams_df_copy['id_str'] = teams_df_copy['id'].astype(str)
                    
                    player_stats = player_stats.merge(
                        teams_df_copy[['id_str', 'nickname']], 
                        left_on='team_id_str', right_on='id_str', how='left', suffixes=('', '_team')
                    )
                except Exception as e:
                    print(f"Error merging team data: {e}")
            
            # Prepare display columns
            display_cols = ['full_name']
            if 'position' in filtered_players.columns:
                display_cols.append('position')
            if 'nickname' in player_stats.columns:
                display_cols.append('nickname')
            display_cols.extend(['ppg', 'rpg', 'apg', 'fg_pct'])
            
            # Filter valid columns and create display df
            display_cols = get_valid_columns(player_stats, display_cols)
            display_df = player_stats[display_cols].copy()
            
            # Rename and format
            rename_map = {
                'full_name': 'Player', 'position': 'Position', 'nickname': 'Team',
                'ppg': 'PPG', 'rpg': 'RPG', 'apg': 'APG', 'fg_pct': 'FG%'
            }
            display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns}, inplace=True)
            
            # Sort by PPG if available
            if 'PPG' in display_df.columns:
                display_df = display_df.sort_values('PPG', ascending=False)
            
            # Format for display
            format_dict = {
                'PPG': '{:.1f}', 'RPG': '{:.1f}', 'APG': '{:.1f}',
                'FG%': '{:.3f}'
            }
            
            st.dataframe(
                display_df.style.format({k: v for k, v in format_dict.items() if k in display_df.columns}),
                height=400, use_container_width=True
            )
            
            # Player selection for detailed analysis
            selected_player = st.selectbox("Select a player for detailed analysis", filtered_players['full_name'])
            if not selected_player:
                return
                
            # Get player data
            selected_id = filtered_players[filtered_players['full_name'] == selected_player]['id'].iloc[0]
            player = filtered_players[filtered_players['id'] == selected_id].iloc[0]
            player_stats = stats_df[stats_df['player_id'] == selected_id].sort_values('season')
            
            # Display player header with image
            col1, col2 = st.columns([1, 3])
            with col1:
                player_img = get_player_image(player['full_name'])
                st.image(player_img, width=150)
            
            with col2:
                team_name = safe_team_lookup(player_stats, teams_df)
                st.markdown(f"## {player['full_name']}")
                position_text = f"**Position:** {player['position']} | " if 'position' in player else ""
                st.markdown(f"{position_text}**Team:** {team_name}")
                
                if all(k in player for k in ['height', 'weight', 'country']):
                    st.markdown(f"**Height:** {player['height']} | **Weight:** {player['weight']} lbs | **Country:** {player['country']}")
                
                if 'draft_year' in player:
                    st.markdown(f"**Draft Year:** {int(player['draft_year']) if pd.notna(player['draft_year']) else 'N/A'}")
            
            # Player tabs
            st.markdown("---")
            player_tabs = st.tabs(["Career Stats", "Season Comparison", "Performance Analysis", "Advanced Metrics"])
            
            with player_tabs[0]: render_career_progression(player, player_stats)
            with player_tabs[1]: render_season_comparison(player, player_stats)
            with player_tabs[2]: render_performance_analysis(player, player_stats)
            with player_tabs[3]: render_advanced_metrics(player, player_stats)
        else:
            st.dataframe(filtered_players[['full_name']], height=400, use_container_width=True)
            st.info("Stats data is not available for these players. Try refreshing NBA data.")

    except Exception as e:
        import traceback
        print(f"❌ Error in render_player_analysis: {e}")
        print(traceback.format_exc())
        st.error(f"An error occurred: {str(e)}")

# ======================= Tab-Specific Render Functions =======================
def render_career_progression(player, player_stats):
    """Render player career progression charts"""
    try:
        st.subheader(f"{player['full_name']} - Career Progression")
        
        if player_stats.empty:
            st.info("No career data available for this player.")
            return
        
        # Ensure seasons are displayed as integers
        player_stats = player_stats.copy()
        if 'season' in player_stats.columns:
            player_stats['season'] = player_stats['season'].astype(int)
        
        # Key stats chart
        st.markdown("### Key Statistics by Season")
        stats_fig = create_stat_chart(
            player_stats, 'season', ['ppg', 'rpg', 'apg'], 
            ['Points', 'Rebounds', 'Assists'],
            "Points, Rebounds, and Assists Per Game by Season",
            "Per Game Average"
        )
        st.plotly_chart(stats_fig, use_container_width=True)
        
        # Shooting percentages chart
        st.markdown("### Shooting Percentages by Season")
        pct_fig = create_stat_chart(
            player_stats, 'season', ['fg_pct', 'fg3_pct', 'ft_pct'],
            ['Field Goal %', '3-Point %', 'Free Throw %'],
            "Shooting Percentages by Season",
            "Percentage", True
        )
        st.plotly_chart(pct_fig, use_container_width=True)
        
        # Season stats table
        st.subheader("Season-by-Season Statistics")
        
        # Get available columns and prepare display dataframe
        cols = get_valid_columns(player_stats, ['season', 'games_played', 'ppg', 'rpg', 'apg', 'spg', 'bpg', 
                                               'fg_pct', 'fg3_pct', 'ft_pct', 'minutes'])
        
        display_stats = player_stats[cols].copy()
        
        # Rename columns
        rename_map = {
            'season': 'Season', 'games_played': 'GP', 'ppg': 'PTS', 'rpg': 'REB', 'apg': 'AST',
            'spg': 'STL', 'bpg': 'BLK', 'fg_pct': 'FG%', 'fg3_pct': '3P%', 'ft_pct': 'FT%',
            'minutes': 'MIN'
        }
        display_stats.rename(columns={k: v for k, v in rename_map.items() if k in cols}, inplace=True)
        
        # Sort by most recent season first
        if 'Season' in display_stats.columns:
            display_stats = display_stats.sort_values('Season', ascending=False)
        
        # Format values
        format_dict = {
            'FG%': '{:.3f}', '3P%': '{:.3f}', 'FT%': '{:.3f}',
            'PTS': '{:.1f}', 'REB': '{:.1f}', 'AST': '{:.1f}', 
            'STL': '{:.1f}', 'BLK': '{:.1f}', 'MIN': '{:.1f}'
        }
        
        st.dataframe(
            display_stats.style.format({k: v for k, v in format_dict.items() if k in display_stats.columns}),
            height=300, use_container_width=True
        )
    except Exception as e:
        print(f"Error in render_career_progression: {e}")
        st.error("An error occurred displaying career progression.")

def render_season_comparison(player, player_stats):
    """Render comparison between two seasons for a player"""
    try:
        st.subheader(f"{player['full_name']} - Season Comparison")
        
        if player_stats.empty or len(player_stats) < 2:
            st.info("Need at least two seasons of data for comparison.")
            return
        
        # Ensure seasons are displayed as integers
        player_stats = player_stats.copy()
        if 'season' in player_stats.columns:
            player_stats['season'] = player_stats['season'].astype(int)
        
        # Season selection
        available_seasons = sorted(player_stats['season'].unique(), reverse=True)
        col1, col2 = st.columns(2)
        with col1:
            season1 = st.selectbox("Select First Season", available_seasons, index=0)
        with col2:
            season2 = st.selectbox("Select Second Season", available_seasons, index=min(1, len(available_seasons)-1))
        
        # Get season data
        season1_data = player_stats[player_stats['season'] == season1]
        season2_data = player_stats[player_stats['season'] == season2]
        
        if season1_data.empty or season2_data.empty:
            st.info(f"Could not find data for one of the selected seasons.")
            return
            
        season1_data, season2_data = season1_data.iloc[0], season2_data.iloc[0]
        
        # Categories for radar chart
        categories = ['ppg', 'rpg', 'apg', 'spg', 'bpg']
        category_names = ['Points', 'Rebounds', 'Assists', 'Steals', 'Blocks']
        
        # Get common categories present in both seasons
        valid_cats, valid_names = [], []
        for i, cat in enumerate(categories):
            if cat in season1_data and cat in season2_data:
                valid_cats.append(cat)
                valid_names.append(category_names[i])
        
        if not valid_cats:
            st.info("Missing required statistics for comparison.")
            return
        
        # Create radar chart
        season1_values = [season1_data[cat] for cat in valid_cats]
        season2_values = [season2_data[cat] for cat in valid_cats]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=season1_values, theta=valid_names, fill='toself', name=f"Season {int(season1)}"
        ))
        fig.add_trace(go.Scatterpolar(
            r=season2_values, theta=valid_names, fill='toself', name=f"Season {int(season2)}"
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title=f"Statistical Comparison: {int(season1)} vs {int(season2)}",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison table
        st.markdown("### Detailed Season Comparison")
        
        # Get common stats
        compare_stats = ['games_played', 'ppg', 'rpg', 'apg', 'spg', 'bpg', 
                         'fg_pct', 'fg3_pct', 'ft_pct', 'minutes']
        compare_labels = ['Games Played', 'Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 
                         'FG%', '3P%', 'FT%', 'Minutes']
        
        valid_stats, valid_labels = [], []
        for i, stat in enumerate(compare_stats):
            if stat in season1_data and stat in season2_data:
                valid_stats.append(stat)
                valid_labels.append(compare_labels[i])
        
        # Calculate differences
        diffs = [season2_data[stat] - season1_data[stat] for stat in valid_stats]
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Statistic': valid_labels,
            f'Season {int(season1)}': [season1_data[stat] for stat in valid_stats],
            f'Season {int(season2)}': [season2_data[stat] for stat in valid_stats],
            'Difference': diffs
        })
        
        # Format and display
        def color_diff(val):
            if isinstance(val, str): return ''
            threshold = 0.01 if isinstance(val, float) and abs(val) < 1 else 0.5
            return 'color: green' if val > threshold else 'color: red' if val < -threshold else ''
        
        st.dataframe(
            comparison_df.style.format({
                f'Season {int(season1)}': lambda x: f"{x:.1f}" if isinstance(x, (int, float)) and x >= 1 else f"{x:.3f}" if isinstance(x, float) else x,
                f'Season {int(season2)}': lambda x: f"{x:.1f}" if isinstance(x, (int, float)) and x >= 1 else f"{x:.3f}" if isinstance(x, float) else x,
                'Difference': lambda x: f"{x:+.1f}" if isinstance(x, (int, float)) and abs(x) >= 1 else f"{x:+.3f}" if isinstance(x, float) else x
            }).applymap(color_diff, subset=['Difference']),
            height=400, use_container_width=True
        )
        
        # Key insights
        st.subheader("Key Insights")
        
        # Calculate significant changes
        key_stats = ['ppg', 'rpg', 'apg']
        key_labels = ['scoring', 'rebounding', 'assists']
        
        insights = []
        for i, stat in enumerate(key_stats):
            if stat in season1_data and stat in season2_data and season1_data[stat] > 0:
                pct_change = 100 * (season2_data[stat] - season1_data[stat]) / season1_data[stat]
                if abs(pct_change) >= 10:
                    direction = "increased" if pct_change > 0 else "decreased"
                    insights.append(f"{player['full_name']}'s {key_labels[i]} {direction} by {abs(pct_change):.1f}% from {int(season1)} to {int(season2)}.")
        
        if insights:
            for insight in insights:
                st.write(insight)
        else:
            st.write(f"No significant statistical changes between {int(season1)} and {int(season2)} seasons.")
    except Exception as e:
        print(f"Error in render_season_comparison: {e}")
        st.error("An error occurred displaying season comparison.")

def render_performance_analysis(player, player_stats):
    """Render deep performance analysis for a player"""
    try:
        st.subheader(f"{player['full_name']} - Performance Analysis")
        
        if player_stats.empty:
            st.info("No performance data available for this player.")
            return
        
        # Ensure seasons are displayed as integers
        player_stats = player_stats.copy()
        if 'season' in player_stats.columns:
            player_stats['season'] = player_stats['season'].astype(int)
        
        # Most recent season metrics
        recent_season = player_stats['season'].max()
        recent_data = player_stats[player_stats['season'] == recent_season].iloc[0]
        
        # Calculate deltas vs career averages
        col1, col2, col3 = st.columns(3)
        metrics = [
            ('ppg', 'Points', col1),
            ('rpg', 'Rebounds', col2),
            ('apg', 'Assists', col3)
        ]
        
        for stat, label, col in metrics:
            if stat in recent_data:
                val = recent_data[stat]
                delta = val - player_stats[stat].mean()
                with col:
                    st.metric(label, f"{val:.1f}", delta=f"{delta:.1f} vs career")
            else:
                with col:
                    st.metric(label, "N/A")
        
        # Performance vs Minutes analysis
        st.subheader("Performance vs. Minutes")
        
        stat_option = st.selectbox(
            "Select Statistic",
            options=["Points", "Rebounds", "Assists"],
            key="perf_vs_minutes_stat"
        )
        
        stat_map = {
            "Points": "ppg", "Rebounds": "rpg",
            "Assists": "apg"
        }
        selected_stat = stat_map[stat_option]
        
        if 'minutes' in player_stats.columns and selected_stat in player_stats.columns:
            # Create scatter plot
            fig = px.scatter(
                player_stats,
                x='minutes', y=selected_stat,
                size='games_played' if 'games_played' in player_stats.columns else None,
                color='season', hover_name='season',
                hover_data=get_valid_columns(player_stats, ['ppg', 'rpg', 'apg', 'games_played']),
                title=f"{stat_option} vs. Minutes by Season",
                labels={'minutes': 'Minutes Per Game', selected_stat: stat_option, 'games_played': 'Games Played'}
            )
            
            # Ensure seasons display as integers
            seasons = sorted(player_stats['season'].unique())
            fig.update_layout(
                coloraxis=dict(
                    colorbar=dict(
                        title=dict(
                            text="Season",
                            side="right"
                        ),
                        tickvals=seasons,
                        ticktext=[str(int(s)) for s in seasons]
                    )
                ),
                height=450
            )
            
            # Update hover template to show integer seasons
            fig.update_traces(
                hovertemplate=fig.data[0].hovertemplate.replace(
                    "%{hovertext}", "Season %{hovertext:.0f}"
                )
            )
            
            # Add trendline if enough data
            x, y = player_stats['minutes'], player_stats[selected_stat]
            if len(x) >= 3:
                coeffs = np.polyfit(x, y, 1)
                line_x = np.array([min(x), max(x)])
                line_y = coeffs[0] * line_x + coeffs[1]
                
                fig.add_trace(go.Scatter(
                    x=line_x, y=line_y, mode='lines', name='Trend',
                    line=dict(color='rgba(0,0,0,0.7)', dash='dash')
                ))
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Missing required data for the {stat_option} vs. Minutes chart.")
        
        # Scoring breakdown
        st.subheader("Scoring Breakdown")
        st.markdown("#### Points Distribution Over Career")
        
        required_cols = ['season', 'ppg', 'fg_pct', 'fg3_pct', 'ft_pct']
        if all(col in player_stats.columns for col in required_cols):
            # Estimate scoring breakdown
            seasons = player_stats['season'].tolist()
            two_points, three_points, ft_points = [], [], []
            
            for _, season in player_stats.iterrows():
                ppg, fg_pct, fg3_pct, ft_pct = season['ppg'], season['fg_pct'], season['fg3_pct'], season['ft_pct']
                
                # Approximate breakdown
                three_est = min(ppg * 0.3, ppg * fg3_pct)
                ft_est = min(ppg * 0.2, ppg * ft_pct)
                two_est = max(0, ppg - three_est - ft_est)
                
                two_points.append(two_est)
                three_points.append(three_est)
                ft_points.append(ft_est)
            
            # Create stacked bar chart
            scoring_df = pd.DataFrame({
                'Season': seasons,
                '2-Point': two_points,
                '3-Point': three_points,
                'Free Throws': ft_points
            })
            
            # Ensure seasons are integers
            scoring_df['Season'] = scoring_df['Season'].astype(int)
            
            scoring_melt = scoring_df.melt(
                id_vars=['Season'],
                value_vars=['2-Point', '3-Point', 'Free Throws'],
                var_name='Shot Type', value_name='Points'
            )
            
            fig = px.bar(
                scoring_melt, x='Season', y='Points', color='Shot Type',
                title="Estimated Scoring Breakdown by Season",
                labels={'Season': 'Season', 'Points': 'Points Per Game'}
            )
            
            # Format x-axis for clean integer seasons
            season_values = sorted(scoring_df['Season'].unique())
            fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=season_values,
                    ticktext=[str(s) for s in season_values],
                    tickformat="d"  # Display as integers
                ),
                height=400, 
                barmode='stack'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Missing required data for scoring breakdown.")
    except Exception as e:
        print(f"Error in render_performance_analysis: {e}")
        st.error("An error occurred displaying performance analysis.")

def render_advanced_metrics(player, player_stats):
    """Render advanced statistical metrics for a player"""
    try:
        st.subheader(f"{player['full_name']} - Advanced Metrics")
        
        if player_stats.empty:
            st.info("No advanced metrics available for this player.")
            return
        
        # Ensure seasons are integers
        stats_enhanced = player_stats.copy()
        if 'season' in stats_enhanced.columns:
            stats_enhanced['season'] = stats_enhanced['season'].astype(int)
        
        # ========== EFFICIENCY METRICS ==========
        st.markdown("### Shooting Efficiency")
        
        # TS% - TRUE SHOOTING PERCENTAGE
        if all(col in stats_enhanced.columns for col in ['PTS', 'FGA', 'FTA']):
            stats_enhanced['ts_pct'] = stats_enhanced.apply(
                lambda row: row['PTS'] / (2 * (row['FGA'] + 0.44 * row['FTA'])) 
                if row['FGA'] > 0 else None, 
                axis=1
            )
        elif all(col in stats_enhanced.columns for col in ['ppg', 'fga', 'fta']):
            stats_enhanced['ts_pct'] = stats_enhanced.apply(
                lambda row: row['ppg'] / (2 * (row['fga'] + 0.44 * row['fta'])) 
                if row['fga'] > 0 else None, 
                axis=1
            )
        # Fallback to FG% and FT% if raw attempts are not available
        elif all(col in stats_enhanced.columns for col in ['fg_pct', 'ft_pct']):
            stats_enhanced['ts_pct'] = stats_enhanced.apply(
                lambda row: (row['fg_pct'] * 0.8) + (row['ft_pct'] * 0.2),
                axis=1
            )
        # Last resort: Just use FG% as a base if nothing else is available
        elif 'fg_pct' in stats_enhanced.columns:
            stats_enhanced['ts_pct'] = stats_enhanced['fg_pct'] * 1.1  # TS% is typically ~10% higher than FG%
            
        # eFG% - EFFECTIVE FIELD GOAL PERCENTAGE
        if all(col in stats_enhanced.columns for col in ['FGM', 'FGA', 'FG3M']):
            stats_enhanced['efg_pct'] = stats_enhanced.apply(
                lambda row: (row['FGM'] + 0.5 * row['FG3M']) / row['FGA'] 
                if row['FGA'] > 0 else None, 
                axis=1
            )
        elif all(col in stats_enhanced.columns for col in ['fgm', 'fga', 'fg3m']):
            stats_enhanced['efg_pct'] = stats_enhanced.apply(
                lambda row: (row['fgm'] + 0.5 * row['fg3m']) / row['fga'] 
                if row['fga'] > 0 else None, 
                axis=1
            )
        # Fallback: Use FG% and FG3% to estimate
        elif all(col in stats_enhanced.columns for col in ['fg_pct', 'fg3_pct']):
            stats_enhanced['efg_pct'] = stats_enhanced.apply(
                lambda row: row['fg_pct'] * (1 + 0.5 * row['fg3_pct'] / max(0.01, row['fg_pct'])) 
                if pd.notna(row['fg3_pct']) else row['fg_pct'] * 1.05,
                axis=1
            )
        # Last resort
        elif 'fg_pct' in stats_enhanced.columns:
            stats_enhanced['efg_pct'] = stats_enhanced['fg_pct'] * 1.05
            
        # 3PAr - THREE POINT ATTEMPT RATE
        if all(col in stats_enhanced.columns for col in ['FG3A', 'FGA']):
            stats_enhanced['fg3a_rate'] = stats_enhanced.apply(
                lambda row: row['FG3A'] / row['FGA'] if row['FGA'] > 0 else 0, 
                axis=1
            )
        elif all(col in stats_enhanced.columns for col in ['fg3a', 'fga']):
            stats_enhanced['fg3a_rate'] = stats_enhanced.apply(
                lambda row: row['fg3a'] / row['fga'] if row['fga'] > 0 else 0, 
                axis=1
            )
        elif 'fg3_pct' in stats_enhanced.columns and 'fg_pct' in stats_enhanced.columns:
            # Rough estimate based on percentages
            stats_enhanced['fg3a_rate'] = stats_enhanced.apply(
                lambda row: 0.33 if pd.isna(row['fg3_pct']) else min(0.8, max(0.1, row['fg3_pct'] / max(0.01, row['fg_pct']) * 0.4)), 
                axis=1
            )
            
        # Prepare metrics for display - ONLY INCLUDE SHOOTING EFFICIENCY METRICS (NO PER)
        efficiency_metrics = []
        
        if 'ts_pct' in stats_enhanced.columns and not stats_enhanced['ts_pct'].isna().all():
            efficiency_metrics.append(('True Shooting %', 'ts_pct'))
            
        if 'efg_pct' in stats_enhanced.columns and not stats_enhanced['efg_pct'].isna().all():
            efficiency_metrics.append(('Effective Field Goal %', 'efg_pct'))
            
        if 'fg3a_rate' in stats_enhanced.columns and not stats_enhanced['fg3a_rate'].isna().all():
            efficiency_metrics.append(('3-Point Attempt Rate', 'fg3a_rate'))
            
        # Create separate percentage chart with proper formatting
        if efficiency_metrics:
            # Create dataframe for chart
            pct_df = pd.DataFrame()
            pct_df['Season'] = stats_enhanced['season']
            
            for display_name, col_name in efficiency_metrics:
                pct_df[display_name] = stats_enhanced[col_name]
            
            # DO NOT USE make_subplot_figure - Create simple line chart directly with px.line
            pct_cols = [name for name, _ in efficiency_metrics]
            pct_melt = pct_df.melt(
                id_vars=['Season'], value_vars=pct_cols,
                var_name='Metric', value_name='Value'
            )
            
            # Sort by season
            pct_melt = pct_melt.sort_values('Season')
            
            # Create simple line chart for percentages with correct formatting
            fig = px.line(
                pct_melt, x='Season', y='Value', color='Metric',
                markers=True, 
                title="Shooting Efficiency Metrics by Season"
            )
            
            # Format specifically for percentage values
            fig.update_layout(
                yaxis=dict(tickformat=".3f", title="Value"),
                xaxis=dict(
                    tickmode='array',
                    tickvals=sorted(pct_df['Season'].unique()),
                    ticktext=[str(s) for s in sorted(pct_df['Season'].unique())],
                    title="Season"
                ),
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create metrics table with all metrics
            st.markdown("#### Shooting Efficiency by Season")
            metrics_display = pd.DataFrame()
            metrics_display['Season'] = stats_enhanced['season']
            
            for display_name, col_name in efficiency_metrics:
                short_name = {
                    'True Shooting %': 'TS%',
                    'Effective Field Goal %': 'eFG%',
                    '3-Point Attempt Rate': '3PAr'
                }.get(display_name, display_name)
                
                metrics_display[short_name] = stats_enhanced[col_name]
            
            # Sort by recent season first    
            metrics_display = metrics_display.sort_values('Season', ascending=False)
            
            # Format with proper decimal places
            format_dict = {
                'TS%': '{:.3f}', 'eFG%': '{:.3f}', '3PAr': '{:.3f}'
            }
            
            format_dict = {k: v for k, v in format_dict.items() if k in metrics_display.columns}
                
            st.dataframe(
                metrics_display.style.format(format_dict),
                height=300, use_container_width=True
            )
            
            # Add simple explanation of metrics
            st.info("""
### What These Metrics Mean:

**TS% (True Shooting Percentage)**: Measures shooting efficiency accounting for 2-pointers, 3-pointers, and free throws. League average is typically around 0.550.

**eFG% (Effective Field Goal Percentage)**: Adjusts FG% to account for the fact that 3-pointers are worth more. League average is typically around 0.520.

**3PAr (Three-Point Attempt Rate)**: Percentage of field goal attempts that are 3-pointers. Shows how much a player relies on 3-point shooting.
            """)
            
        else:
            st.info("Insufficient data to calculate advanced shooting metrics.")
            
        # ========== SCORING DISTRIBUTION ========== 
        st.markdown("### Scoring Distribution")
        
        # Calculate points breakdown if we have the necessary data
        if all(k in stats_enhanced.columns for k in ['season', 'FG3M', 'FTM', 'PTS']):
            # Calculate exact breakdown from NBA API data
            scoring_df = pd.DataFrame()
            scoring_df['Season'] = stats_enhanced['season']
            
            scoring_df['3-Point'] = stats_enhanced.apply(lambda row: 3 * row['FG3M'], axis=1)
            scoring_df['Free Throws'] = stats_enhanced.apply(lambda row: row['FTM'], axis=1)
            scoring_df['2-Point'] = stats_enhanced.apply(
                lambda row: row['PTS'] - scoring_df.loc[row.name, '3-Point'] - scoring_df.loc[row.name, 'Free Throws'], 
                axis=1
            )
            
            # Create stacked bar chart
            scoring_melt = scoring_df.melt(
                id_vars=['Season'], 
                value_vars=['2-Point', '3-Point', 'Free Throws'],
                var_name='Shot Type', value_name='Points'
            )
            
            fig = px.bar(
                scoring_melt, x='Season', y='Points', color='Shot Type',
                title="Scoring Distribution by Season",
                labels={'Season': 'Season', 'Points': 'Points Per Game'}
            )
            
            # Format for integers
            season_values = sorted(scoring_df['Season'].unique())
            fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=season_values,
                    ticktext=[str(s) for s in season_values],
                    tickformat="d"
                ),
                height=400, 
                barmode='stack'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        # Fallback: estimate from percentages
        elif all(k in stats_enhanced.columns for k in ['season', 'ppg', 'fg3_pct', 'ft_pct']):
            # Create estimates
            scoring_df = pd.DataFrame()
            scoring_df['Season'] = stats_enhanced['season']
            
            scoring_df['3-Point'] = stats_enhanced.apply(
                lambda row: min(row['ppg'] * 0.3, row['ppg'] * row['fg3_pct']), 
                axis=1
            )
            scoring_df['Free Throws'] = stats_enhanced.apply(
                lambda row: min(row['ppg'] * 0.2, row['ppg'] * row['ft_pct']), 
                axis=1
            )
            scoring_df['2-Point'] = stats_enhanced.apply(
                lambda row: max(0, row['ppg'] - scoring_df.loc[row.name, '3-Point'] - scoring_df.loc[row.name, 'Free Throws']),
                axis=1
            )
            
            # Create stacked bar chart
            scoring_melt = scoring_df.melt(
                id_vars=['Season'], 
                value_vars=['2-Point', '3-Point', 'Free Throws'],
                var_name='Shot Type', value_name='Points'
            )
            
            fig = px.bar(
                scoring_melt, x='Season', y='Points', color='Shot Type',
                title="Estimated Scoring Distribution by Season",
                labels={'Season': 'Season', 'Points': 'Points Per Game'}
            )
            
            # Format for integers
            season_values = sorted(scoring_df['Season'].unique())
            fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=season_values,
                    ticktext=[str(s) for s in season_values],
                    tickformat="d"
                ),
                height=400, 
                barmode='stack'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Note: Scoring distribution is estimated based on available percentages.")
            
    except Exception as e:
        print(f"Error in render_advanced_metrics: {e}")
        import traceback
        print(traceback.format_exc())
        st.error("An error occurred displaying advanced metrics.")