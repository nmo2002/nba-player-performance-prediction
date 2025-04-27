import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import community as community_louvain
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform

def render_team_comparison(player_df, stats_df, teams_df, team_stats_df=None):
    """Main function to render team comparison dashboard"""
    st.title("Team Comparison & Network Analysis")
    
    # Season selection and team setup
    available_seasons = sorted([int(s) for s in stats_df['season'].unique()], reverse=True)
    if not available_seasons:
        st.error("No season data available.")
        return
        
    selected_season = st.selectbox("Select season", options=available_seasons, index=0)
    season_stats = stats_df[stats_df['season'] == selected_season]
    team_mapping = create_team_mapping(teams_df)
    
    # Get available teams and let user select
    available_teams = get_available_teams(season_stats, team_mapping)
    if not available_teams:
        st.warning(f"No team data available for season {selected_season}")
        return
    
    selected_teams = st.multiselect(
        "Select teams", options=sorted(available_teams),
        default=sorted(available_teams)[:2] if len(available_teams) >= 2 else available_teams
    )
    
    if not selected_teams:
        st.info("Select at least one team to view comparison.")
        return
        
    team_ids = [get_team_id(team_name, team_mapping) for team_name in selected_teams]
    
    # Create tabs for different views
    team_tabs = st.tabs(["Team Stats", "Roster Comparison", "Player Distribution", "Network Analysis"])
    
    with team_tabs[0]:
        render_team_stats_comparison(selected_teams, team_ids, season_stats, teams_df, team_stats_df, selected_season)
    
    with team_tabs[1]:
        render_roster_comparison(selected_teams, team_ids, player_df, season_stats)
        
    with team_tabs[2]:
        render_player_distribution(selected_teams, team_ids, player_df, season_stats)
        
    with team_tabs[3]:
        render_network_analysis(selected_teams, team_ids, player_df, stats_df, teams_df, selected_season)

def create_team_mapping(teams_df):
    """Create mapping between team IDs and names"""
    team_mapping = {}
    for _, team in teams_df.iterrows():
        if 'team_id' in team and 'nickname' in team:
            team_mapping[int(team['team_id'])] = team['nickname']
        elif 'id' in team and 'nickname' in team:
            team_mapping[int(team['id'])] = team['nickname']
    return team_mapping

def get_team_id(team_name, team_mapping):
    """Get team ID from team name"""
    for team_id, name in team_mapping.items():
        if name == team_name:
            return int(team_id)
    return None

def get_available_teams(stats_df, team_mapping):
    """Get list of available teams"""
    teams = []
    for team_id in stats_df['team_id'].unique():
        try:
            if int(team_id) in team_mapping:
                teams.append(team_mapping[int(team_id)])
        except (ValueError, TypeError):
            continue
    return teams

def extract_team_stats(team_stats_df, team_mapping, team_id_col):
    """Extract and format team statistics"""
    team_stats = []
    column_mapping = {
        'TEAM_ID': 'team_id', 'TEAM_NAME': 'team_name', 'W': 'wins', 
        'L': 'losses', 'W_PCT': 'win_pct', 'PTS': 'ppg', 'REB': 'rpg', 
        'AST': 'apg', 'STL': 'spg', 'BLK': 'bpg', 'TOV': 'tpg', 
        'FG_PCT': 'fg_pct', 'FG3_PCT': 'fg3_pct', 'FT_PCT': 'ft_pct',
        'SEASON': 'season', 'SEASON_ID': 'season_id', 'SEASON_YEAR': 'season_year'
    }
    
    for _, row in team_stats_df.iterrows():
        team_id = int(row[team_id_col])
        team_name = team_mapping.get(team_id, f"Team {team_id}")
        
        # Create team stats dictionary
        team_dict = {'Team': team_name, 'Team ID': team_id}
        
        # Extract season information
        season_value = None
        for season_col in ['SEASON_YEAR', 'season_year', 'SEASON', 'season']:
            if season_col in row and pd.notna(row[season_col]):
                season_value = row[season_col]
                break
        
        if season_value is None and 'SEASON_ID' in row:
            season_id = str(row['SEASON_ID'])
            if len(season_id) >= 5:
                season_value = int(season_id[1:5])
        
        if season_value is not None:
            team_dict['Season'] = season_value
        
        # Add stats
        for api_col, our_col in column_mapping.items():
            if api_col in row and api_col not in ['SEASON', 'SEASON_ID', 'SEASON_YEAR']:
                team_dict[our_col.title() if our_col != 'win_pct' else 'Win %'] = row[api_col]
            elif our_col in row and our_col not in ['season', 'season_id', 'season_year']:
                team_dict[our_col.title() if our_col != 'win_pct' else 'Win %'] = row[our_col]
        
        # Format special columns
        for pct_col in ['Fg_pct', 'Fg3_pct', 'Ft_pct']:
            if pct_col in team_dict:
                team_dict[f"Avg {pct_col.split('_')[0].upper()}%"] = team_dict.pop(pct_col)
        
        for stat in ['Ppg', 'Rpg', 'Apg', 'Spg', 'Bpg', 'Tpg']:
            if stat in team_dict:
                team_dict[f"Total {stat.upper()}"] = team_dict.pop(stat)
        
        team_stats.append(team_dict)
    
    return team_stats

def get_team_stats(selected_teams, team_ids, stats_df, teams_df, team_stats_df=None):
    """Get team statistics from API or by aggregation"""
    # Try to use official team stats first
    if team_stats_df is not None and not team_stats_df.empty:
        team_id_col = next((col for col in team_stats_df.columns 
                        if col.upper() == 'TEAM_ID' or col == 'team_id'), None)
        
        if team_id_col:
            team_ids_int = [int(tid) for tid in team_ids]
            team_stats_df[team_id_col] = team_stats_df[team_id_col].astype(int)
            selected_team_stats = team_stats_df[team_stats_df[team_id_col].isin(team_ids_int)]
            
            if not selected_team_stats.empty:
                return extract_team_stats(selected_team_stats, create_team_mapping(teams_df), team_id_col)
    
    # Calculate from player stats
    team_stats = []
    stats_df_copy = stats_df.copy()
    stats_df_copy['team_id_int'] = stats_df_copy['team_id'].astype(int)
    team_ids_int = [int(tid) for tid in team_ids]
    
    for team_name, team_id in zip(selected_teams, team_ids_int):
        team_players = stats_df_copy[stats_df_copy['team_id_int'] == team_id]
        
        if not team_players.empty:
            season_value = team_players['season'].iloc[0] if 'season' in team_players.columns else None
            
            team_stats.append({
                'Team': team_name,
                'Team ID': team_id,
                'Season': season_value,
                'Players': len(team_players),
                'Total PPG': team_players['ppg'].sum(),
                'Total RPG': team_players['rpg'].sum(),
                'Total APG': team_players['apg'].sum(),
                'Avg FG%': team_players['fg_pct'].mean(),
                'Avg 3P%': team_players['fg3_pct'].mean() if 'fg3_pct' in team_players else 0,
                'Avg FT%': team_players['ft_pct'].mean() if 'ft_pct' in team_players else 0
            })
    
    return team_stats

def render_team_stats_comparison(selected_teams, team_ids, stats_df, teams_df, team_stats_df=None, selected_season=None):
    """Render team statistics comparison"""
    st.subheader("Team Statistics Comparison")
    
    team_stats = get_team_stats(selected_teams, team_ids, stats_df, teams_df, team_stats_df)
    if not team_stats:
        st.warning("No team statistics available for the selected teams.")
        return
        
    team_stats_df = pd.DataFrame(team_stats)
    
    # Add season if needed and create Team_Season column
    if 'Season' not in team_stats_df.columns and selected_season:
        team_stats_df['Season'] = selected_season
    if 'Season' in team_stats_df.columns:
        team_stats_df['Team_Season'] = team_stats_df['Team'] + ' ' + team_stats_df['Season'].astype(str)
    
    # Define columns to exclude from charts - ensure we exclude all variations of team ID
    non_stat_cols = ['Team', 'Team_Season', 'Season', 'team_id', 'Team ID', 'Team_Id', 'Team_id', 
                    'TEAM_ID', 'Team Name', 'Players', 'Team_ID', 'team_id_int']
    
    # Get numeric columns for charts
    categories = [col for col in team_stats_df.columns 
                if col not in non_stat_cols and 
                team_stats_df[col].dtype in ['float64', 'int64']]
    
    # Create radar chart
    fig = create_team_radar_chart(team_stats_df, categories)
    st.plotly_chart(fig, use_container_width=True)
    
    # Let user select stat for bar chart
    stat_options = [col for col in categories if col not in non_stat_cols]
    selected_stat = st.selectbox("Select statistic to compare", options=stat_options)
    
    # Create bar chart
    x_axis = 'Team_Season' if 'Team_Season' in team_stats_df.columns else 'Team'
    fig = px.bar(
        team_stats_df, x=x_axis, y=selected_stat, color=x_axis,
        title=f"Team Comparison: {selected_stat}", text=selected_stat
    )
    
    fig.update_traces(
        texttemplate='%{text:.3f}' if selected_stat.startswith('Avg') else '%{text:.1f}', 
        textposition='outside'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Format and display data table - clean up ID columns
    display_df = team_stats_df.copy()
    
    # Keep only one Team ID column and rename it
    if 'Team ID' in display_df.columns:
        # Remove any other team ID columns
        for col in ['team_id', 'Team_Id', 'Team_id', 'TEAM_ID', 'Team_ID', 'team_id_int']:
            if col in display_df.columns and col != 'Team ID':
                display_df = display_df.drop(columns=[col])
                
    # Format numeric columns
    for col in display_df.columns:
        if col not in ['Team', 'Season', 'Team_Season'] and display_df[col].dtype != 'object':
            if col in ['Team ID']:
                display_df[col] = display_df[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "")
            elif col.startswith('Avg') or col.endswith('%'):
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")
            else:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
    
    st.dataframe(display_df, use_container_width=True)

def create_team_radar_chart(team_stats_df, categories):
    """Create radar chart for team comparison"""
    max_vals = team_stats_df[categories].max()
    min_vals = team_stats_df[categories].min()
    
    fig = go.Figure()
    
    for _, team in team_stats_df.iterrows():
        team_values = []
        for cat in categories:
            norm_val = 0.5 if max_vals[cat] == min_vals[cat] else (team[cat] - min_vals[cat]) / (max_vals[cat] - min_vals[cat])
            team_values.append(norm_val)
        
        # Use Team-Season format if Season is available
        team_label = f"{team['Team']} {int(team['Season'])}" if 'Season' in team and pd.notna(team['Season']) else team['Team']
            
        fig.add_trace(go.Scatterpolar(
            r=team_values, theta=categories, fill='toself', name=team_label
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Team Statistical Comparison",
        height=500
    )
    
    return fig

def render_roster_comparison(selected_teams, team_ids, player_df, stats_df):
    """Render team roster comparison"""
    st.subheader("Roster Comparison")
    
    team_rosters = []
    stats_df_copy = stats_df.copy()
    stats_df_copy['team_id_int'] = stats_df_copy['team_id'].astype(int)
    team_ids_int = [int(tid) for tid in team_ids]
    
    for team_name, team_id in zip(selected_teams, team_ids_int):
        team_players = stats_df_copy[stats_df_copy['team_id_int'] == team_id]
        if not team_players.empty:
            roster = team_players.merge(
                player_df[['id', 'full_name']], 
                left_on='player_id', right_on='id', how='left'
            )
            
            team_rosters.append({'team': team_name, 'roster': roster})
    
    if not team_rosters:
        st.info("No roster data available for the selected teams.")
        return
        
    # Display team rosters in columns
    cols = st.columns(min(len(team_rosters), 3))
    
    for i, roster_data in enumerate(team_rosters):
        team_name = roster_data['team']
        roster = roster_data['roster']
        
        # Show top players for each team
        top_players = roster.sort_values('ppg', ascending=False)[['full_name', 'ppg', 'rpg', 'apg', 'fg_pct']].head(10)
        
        with cols[i % len(cols)]:
            st.markdown(f"### {team_name}")
            
            # Format display table
            display_df = top_players.copy()
            display_df.columns = ['Player', 'PPG', 'RPG', 'APG', 'FG%']
            for col in ['PPG', 'RPG', 'APG']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}")
            display_df['FG%'] = display_df['FG%'].apply(lambda x: f"{x:.3f}")
            
            st.table(display_df)
    
    # Create top scorers comparison
    st.subheader("Top Scorers Comparison")
    
    # Get top 5 scorers from each team
    top_scorers_data = []
    for team_data in team_rosters:
        team_name = team_data['team']
        top_5 = team_data['roster'].nlargest(5, 'ppg')
        
        for _, player in top_5.iterrows():
            top_scorers_data.append({
                'Team': team_name,
                'Player': player['full_name'],
                'PPG': player['ppg']
            })
    
    if top_scorers_data:
        fig = px.bar(
            pd.DataFrame(top_scorers_data), x='Player', y='PPG', color='Team',
            title="Top 5 Scorers by Team", text='PPG'
        )
        
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        
        st.plotly_chart(fig, use_container_width=True)

def render_player_distribution(selected_teams, team_ids, player_df, stats_df):
    """Render player statistics distribution by team"""
    st.subheader("Player Distribution Analysis")
    
    # Get player data by team
    team_data = []
    stats_df_copy = stats_df.copy()
    stats_df_copy['team_id_int'] = stats_df_copy['team_id'].astype(int)
    team_ids_int = [int(tid) for tid in team_ids]
    
    for team_name, team_id in zip(selected_teams, team_ids_int):
        team_players = stats_df_copy[stats_df_copy['team_id_int'] == team_id]
        if not team_players.empty:
            team_roster = team_players.merge(
                player_df[['id', 'full_name']], 
                left_on='player_id', right_on='id', how='left'
            )
            
            for _, player in team_roster.iterrows():
                player_name = player['full_name'] if pd.notna(player['full_name']) else f"Player {player['player_id']}"
                
                team_data.append({
                    'Team': team_name,
                    'Player': player_name,
                    'PPG': player['ppg'],
                    'RPG': player['rpg'],
                    'APG': player['apg'],
                    'FG%': player['fg_pct'],
                    'MPG': player['minutes'],
                    'Games': player['games_played']
                })
    
    if not team_data:
        st.warning("No player data available for the selected teams.")
        return
        
    team_df = pd.DataFrame(team_data)
    
    # Let user select statistic to analyze
    stat_options = ['PPG', 'RPG', 'APG', 'FG%', 'MPG']
    selected_stat = st.selectbox("Select statistic to analyze", options=stat_options, key="dist_stat")
    
    # Create box plot
    fig = px.box(
        team_df, x='Team', y=selected_stat, color='Team',
        title=f"{selected_stat} Distribution by Team",
        points="all", hover_name="Player"
    )
    
    fig.update_traces(boxmean=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show summary statistics
    cols = st.columns(len(selected_teams))
    for i, team in enumerate(selected_teams):
        team_stats = team_df[team_df['Team'] == team][selected_stat]
        if not team_stats.empty:
            with cols[i]:
                st.markdown(f"**{team}**")
                st.write(f"Mean: {team_stats.mean():.2f}")
                st.write(f"Median: {team_stats.median():.2f}")
                st.write(f"Max: {team_stats.max():.2f}")
                st.write(f"Min: {team_stats.min():.2f}")
                st.write(f"Std Dev: {team_stats.std():.2f}")
    
    # Create minutes scatter plot
    fig = px.scatter(
        team_df, x='MPG', y=selected_stat, color='Team', size='Games',
        hover_name='Player', title=f"Minutes vs {selected_stat} by Team"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create contribution treemap
    st.markdown("### Team Contribution Analysis")
    
    contribution_data = []
    for team in selected_teams:
        team_subset = team_df[team_df['Team'] == team]
        team_total = team_subset[selected_stat].sum()
        
        if team_total > 0:
            for _, player in team_subset.iterrows():
                player_name = player['Player']
                if pd.isna(player_name) or player_name == '':
                    player_name = f"Player {_}"
                    
                contribution_data.append({
                    'Team': team,
                    'Player': player_name,
                    'Contribution': (player[selected_stat] / team_total) * 100,
                    'Stat': player[selected_stat]
                })
    
    if contribution_data:
        fig = px.treemap(
            pd.DataFrame(contribution_data), path=['Team', 'Player'], values='Contribution',
            color='Stat', hover_data=['Stat'],
            title=f"Team Contribution Analysis ({selected_stat})"
        )
        
        fig.update_traces(textinfo="label+percent parent")
        st.plotly_chart(fig, use_container_width=True)

def render_network_analysis(selected_teams, team_ids, player_df, stats_df, teams_df, selected_season):
    """Render network analysis visualizations"""
    st.subheader(f"NBA Team-Player Network Analysis - {selected_season}")
    
    # Let user select analysis type
    analysis_type = st.radio(
        "Select network analysis type",
        ["Team-Player Network", "Network Mining"],
        horizontal=True
    )
    
    if analysis_type == "Team-Player Network":
        create_team_player_network(selected_teams, team_ids, player_df, stats_df, teams_df, selected_season)
    else:  # Network Mining
        render_network_mining(selected_teams, stats_df, teams_df, selected_season=selected_season)

def create_team_player_network(selected_teams, team_ids, player_df, stats_df, teams_df, selected_season=None):
    """Create team-player network visualization"""
    st.subheader(f"Team-Player Network Visualization")
    st.markdown("This network shows connections between teams and their players.")
    
    # Get player data
    stats_df_copy = stats_df.copy()
    stats_df_copy['team_id_int'] = stats_df_copy['team_id'].astype(int)
    team_ids_int = [int(tid) for tid in team_ids]
    team_players = stats_df_copy[stats_df_copy['team_id_int'].isin(team_ids_int)]
    
    if team_players.empty:
        st.info("No player data available.")
        return
        
    # Join with player info
    player_info = team_players.merge(
        player_df[['id', 'full_name']], 
        left_on='player_id', right_on='id', how='left'
    )
    
    # Let user select stat to visualize
    stat_options = {
        'Points Per Game': 'ppg',
        'Rebounds Per Game': 'rpg',
        'Assists Per Game': 'apg',
        'Field Goal %': 'fg_pct',
        'Minutes Per Game': 'minutes'
    }
    
    selected_stat = st.selectbox(
        "Select statistic to visualize:",
        options=list(stat_options.keys()),
        index=0
    )
    
    stat_column = stat_options[selected_stat]
    
    # Create network graph
    G = nx.Graph()
    
    # Add team nodes
    for team, team_id in zip(selected_teams, team_ids):
        G.add_node(team, type='team', id=team_id)
    
    # Add player nodes and connections
    for _, player in player_info.iterrows():
        player_name = player['full_name'] if pd.notna(player['full_name']) else f"Player {player['player_id']}"
        team_id = int(player['team_id'])
        
        # Find corresponding team name
        team_name = next((team for team, tid in zip(selected_teams, team_ids) if int(tid) == team_id), None)
                
        if team_name:
            # Add player node with stats
            G.add_node(
                player_name, 
                type='player', 
                ppg=float(player['ppg']),
                rpg=float(player['rpg']),
                apg=float(player['apg']),
                fg_pct=float(player['fg_pct']),
                minutes=float(player['minutes']) if 'minutes' in player else 0,
                selected_stat_value=float(player[stat_column])
            )
            
            G.add_edge(team_name, player_name)
    
    # Visualization with Plotly
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Create node traces for teams and players
    team_nodes_x, team_nodes_y, team_nodes_text = [], [], []
    player_nodes_x, player_nodes_y, player_nodes_text, player_nodes_size, player_nodes_color = [], [], [], [], []
    
    for node in G.nodes():
        x, y = pos[node]
        node_type = G.nodes[node].get('type', '')
        
        if node_type == 'team':
            team_nodes_x.append(x)
            team_nodes_y.append(y)
            team_nodes_text.append(f"Team: {node}")
        else:
            player_nodes_x.append(x)
            player_nodes_y.append(y)
            
            # Get player stats for hover text
            ppg = G.nodes[node].get('ppg', 0)
            rpg = G.nodes[node].get('rpg', 0)
            apg = G.nodes[node].get('apg', 0)
            fg_pct = G.nodes[node].get('fg_pct', 0)
            minutes = G.nodes[node].get('minutes', 0)
            stat_value = G.nodes[node].get('selected_stat_value', 0)
            
            hover_text = (f"Player: {node}<br>"
                         f"PPG: {ppg:.1f}<br>"
                         f"RPG: {rpg:.1f}<br>"
                         f"APG: {apg:.1f}<br>"
                         f"FG%: {fg_pct:.3f}<br>"
                         f"MPG: {minutes:.1f}<br>"
                         f"{selected_stat}: {stat_value:.2f}")
            
            player_nodes_text.append(hover_text)
            
            # Size and color based on selected stat
            size_value = 5 + (stat_value * 2)  # Adjust multiplier based on the stat
            if stat_column == 'fg_pct':
                size_value = 5 + (stat_value * 20)  # Larger multiplier for percentage values
                
            player_nodes_size.append(size_value)
            player_nodes_color.append(stat_value)
    
    # Create edge trace
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create team node trace
    team_node_trace = go.Scatter(
        x=team_nodes_x, y=team_nodes_y,
        mode='markers+text',
        hoverinfo='text',
        text=team_nodes_text,
        textposition="top center",
        marker=dict(
            color='rgba(255, 0, 0, 0.8)',
            size=25,
            line=dict(width=2, color='darkred')
        )
    )
    
    # Create player node trace
    player_node_trace = go.Scatter(
        x=player_nodes_x, y=player_nodes_y,
        mode='markers',
        hoverinfo='text',
        text=player_nodes_text,
        marker=dict(
            colorscale='Blues',
            color=player_nodes_color,
            size=player_nodes_size,
            line=dict(width=1, color='darkblue'),
            showscale=True,
            colorbar=dict(title=selected_stat)
        )
    )
    
    # Create figure
    season_text = f" - {selected_season}" if selected_season else ""
    title_text = f'Team-Player Network - {selected_stat}{season_text}'
    fig = go.Figure(
        data=[edge_trace, team_node_trace, player_node_trace],
        layout=go.Layout(
            title=title_text,
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def get_team_performance_data(stats_df, teams_df, team_stats_df=None):
    """Get aggregated performance data for all teams"""
    team_mapping = create_team_mapping(teams_df)
    
    # Calculate from player stats
    team_data = []
    stats_df['team_id_int'] = stats_df['team_id'].astype(int)
    
    # Find season column
    season_col = next((col for col in ['season', 'SEASON', 'Season', 'SEASON_YEAR', 'season_year'] 
                     if col in stats_df.columns), None)
    
    # Group by team and season if available
    if season_col:
        grouped = stats_df.groupby(['team_id_int', season_col])
    else:
        grouped = stats_df.groupby('team_id_int')
    
    # Process each group
    for group_key, group_data in grouped:
        if season_col:
            team_id, season = group_key
        else:
            team_id, season = group_key, None
            
        if team_id in team_mapping:
            team_name = team_mapping[team_id]
            
            if not group_data.empty:
                stats_dict = {
                    'Team': team_name,
                    'Team ID': int(team_id),
                    'Players': len(group_data),
                    'Total PPG': group_data['ppg'].sum(),
                    'Total RPG': group_data['rpg'].sum(),
                    'Total APG': group_data['apg'].sum(),
                    'Avg FG%': group_data['fg_pct'].mean(),
                    'Avg 3P%': group_data['fg3_pct'].mean() if 'fg3_pct' in group_data.columns else 0,
                    'Avg FT%': group_data['ft_pct'].mean() if 'ft_pct' in group_data.columns else 0
                }
                
                if season is not None:
                    stats_dict['Season'] = season
                    
                team_data.append(stats_dict)
    
    return team_data

def render_network_mining(selected_teams, stats_df, teams_df, selected_season=None):
    """Render network mining analysis for team statistics"""
    st.subheader("Team Network Mining Analysis")
    st.markdown("This analysis uses graph theory to discover relationships between teams based on statistical similarity.")
    
    # Get team performance data
    all_teams_data = get_team_performance_data(stats_df, teams_df)
    
    if not all_teams_data:
        st.warning("Insufficient data for network analysis.")
        return
        
    # Convert to DataFrame for analysis
    all_teams_df = pd.DataFrame(all_teams_data)
    
    # Show available data
    st.markdown("### Available Team Statistics")
    with st.expander("View available team data"):
        # Ensure we only have one Team ID column before displaying
        display_df = all_teams_df.copy()
        id_columns = [col for col in display_df.columns if 'id' in col.lower() and 'team' in col.lower()]
        if len(id_columns) > 1:
            # Keep only the first one
            for col in id_columns[1:]:
                display_df = display_df.drop(columns=[col])
        st.dataframe(display_df.head())
    
    # Add season identifier to team name if available
    if 'Season' in all_teams_df.columns:
        all_teams_df['Team_Season'] = all_teams_df.apply(
            lambda row: f"{row['Team']} {int(row['Season'])}" if pd.notna(row['Season']) else row['Team'], 
            axis=1
        )
    else:
        all_teams_df['Team_Season'] = all_teams_df['Team']
    
    # Get available metrics
    excluded_columns = ['Team', 'Team_Season', 'Season', 'team_id', 'Team ID', 'Team_Id', 'Team_id', 
                      'TEAM_ID', 'Players', 'Team_ID', 'team_id_int']
    
    # Convert numeric columns
    for col in all_teams_df.columns:
        if col not in excluded_columns:
            try:
                all_teams_df[col] = pd.to_numeric(all_teams_df[col], errors='coerce')
            except:
                pass
    
    # Get available numeric columns
    available_metrics = []
    for col in all_teams_df.columns:
        if col not in excluded_columns and pd.api.types.is_numeric_dtype(all_teams_df[col]):
            available_metrics.append(col)
    
    if not available_metrics:
        st.error("No numerical statistics found for network analysis.")
        column_types = pd.DataFrame({
            'Column': all_teams_df.columns,
            'Type': [str(all_teams_df[col].dtype) for col in all_teams_df.columns]
        })
        st.write("Column types in the dataset:")
        st.dataframe(column_types)
        return
        
    # Display available metrics
    st.write(f"Found {len(available_metrics)} available statistics for analysis:")
    st.write(", ".join(available_metrics))
    
    # Select default metrics
    basketball_metrics = ['Total PPG', 'Total RPG', 'Total APG', 'Avg FG%', 'Avg 3P%']
    default_metrics = [m for m in basketball_metrics if m in available_metrics]
    
    if len(default_metrics) < 2:
        default_metrics = available_metrics[:5] if len(available_metrics) >= 5 else available_metrics
    
    # Select metrics for analysis
    st.markdown("### Select Statistics for Similarity Comparison")
    
    selected_metrics = st.multiselect(
        "Select statistics:",
        options=available_metrics,
        default=default_metrics[:5] if len(default_metrics) >= 5 else default_metrics
    )
    
    if len(selected_metrics) < 2:
        st.warning("Please select at least 2 statistics for comparison.")
        return
    
    # Set similarity threshold
    similarity_threshold = st.slider(
        "Similarity threshold (higher = fewer connections)",
        min_value=0.5, max_value=0.95, value=0.75, step=0.05
    )
    
    # Generate network
    if st.button("Generate Team Network"):
        with st.spinner("Building team network graph..."):
            try:
                # Create team network
                fig, G, metrics = create_team_network(
                    all_teams_df, 
                    stats_to_include=selected_metrics,
                    similarity_threshold=similarity_threshold,
                    team_label_col='Team_Season',
                    selected_season=selected_season
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display network metrics
                    st.markdown("### Network Metrics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Network Density", f"{metrics['density']:.3f}")
                        st.metric("Average Clustering", f"{metrics['avg_clustering']:.3f}")
                    
                    with col2:
                        st.metric("Connected Components", metrics['components'])
                        st.metric("Average Path Length", f"{metrics['avg_path_length']:.2f}" if 'avg_path_length' in metrics else "N/A")
                    
                    # Display centrality analysis
                    display_network_centrality(G, all_teams_df, selected_teams, selected_metrics)
                    
            except Exception as e:
                st.error(f"Error creating network: {e}")
                import traceback
                st.error(traceback.format_exc())

def create_team_network(team_df, stats_to_include, similarity_threshold=0.75, 
                       team_label_col='Team', selected_season=None):
    """Create team network based on statistical similarity"""
    # Normalize data for similarity calculation
    team_stats = team_df.copy()
    
    # Prepare data columns
    analysis_cols = stats_to_include if stats_to_include else [
        col for col in team_stats.columns 
        if col not in ['Team', 'Team_Season', 'Season', 'team_id', 'Team ID', 'Players']
    ]
    
    # Check for minimum columns
    if len(analysis_cols) < 2:
        st.error("Not enough statistical columns for analysis.")
        return None, None, {}
    
    # Normalize data
    scaler = MinMaxScaler()
    team_stats[analysis_cols] = scaler.fit_transform(team_stats[analysis_cols])
    
    # Calculate similarity matrix
    similarity_matrix = 1 - squareform(pdist(team_stats[analysis_cols], 'cosine'))
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes (teams)
    for i, team in enumerate(team_stats[team_label_col]):
        G.add_node(
            team, 
            season=team_stats['Season'].iloc[i] if 'Season' in team_stats else selected_season,
            team_stats=team_stats.iloc[i][analysis_cols].to_dict()  # Add team stats as a node attribute
        )
        
    # Add edges (connections between similar teams)
    for i in range(len(team_stats)):
        for j in range(i+1, len(team_stats)):
            similarity = similarity_matrix[i, j]
            if similarity > similarity_threshold:
                G.add_edge(
                    team_stats[team_label_col].iloc[i], 
                    team_stats[team_label_col].iloc[j], 
                    weight=similarity
                )
    
    # Calculate network metrics using our local function (not from data_mining.py)
    metrics = calculate_network_metrics(G)
    
    # Create visualization
    fig = create_network_visualization(G)
    
    return fig, G, metrics

def calculate_network_metrics(G):
    """Calculate network metrics manually without using select_dtypes"""
    metrics = {}
    
    # Calculate basic metrics
    metrics['density'] = nx.density(G)
    
    # Average clustering coefficient
    try:
        metrics['avg_clustering'] = nx.average_clustering(G)
    except:
        metrics['avg_clustering'] = 0
    
    # Count connected components
    metrics['components'] = nx.number_connected_components(G)
    
    # Calculate average path length if graph is connected
    if nx.is_connected(G) and len(G.nodes()) > 1:
        metrics['avg_path_length'] = nx.average_shortest_path_length(G)
    else:
        metrics['avg_path_length'] = 0
        
    return metrics

def create_network_visualization(G):
    """Create network visualization with Plotly"""
    # Calculate layout
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    # Calculate node colors based on community detection
    try:
        communities = community_louvain.best_partition(G)
        color_values = [communities[node] for node in G.nodes()]
    except:
        color_values = [0] * len(G.nodes())
    
    # Node sizes based on degree
    degree_dict = dict(nx.degree(G))
    
    # Calculate centrality metrics
    try:
        betweenness = nx.betweenness_centrality(G)
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06, weight='weight')
    except:
        betweenness = {node: 0 for node in G.nodes()}
        eigenvector = {node: 0 for node in G.nodes()}
    
    # Create node trace
    node_x, node_y, node_text = [], [], []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        
        # Create hover text
        similar_teams = list(G.neighbors(node))
        similar_text = "<br>Similar teams:<br>" + "<br>".join(similar_teams[:3]) if similar_teams else ""
        
        hover_text = (
            f"Team: {node}<br>"
            f"Connections: {degree_dict[node]}<br>"
            f"Betweenness: {betweenness[node]:.3f}<br>"
            f"Eigenvector: {eigenvector[node]:.3f}"
            f"{similar_text}"
        )
        
        node_x.append(x)
        node_y.append(y)
        node_text.append(hover_text)
        node_size.append(10 + degree_dict[node] * 3)
    
    # Create edge traces
    edge_traces = []
    for edge in G.edges():
        edge_x = [pos[edge[0]][0], pos[edge[1]][0], None]
        edge_y = [pos[edge[0]][1], pos[edge[1]][1], None]
        weight = G[edge[0]][edge[1]]['weight'] if 'weight' in G[edge[0]][edge[1]] else 0.5
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=weight*2, color=f'rgba(150, 150, 150, {weight})'),
            hoverinfo='none',
            mode='lines'
        )
        edge_traces.append(edge_trace)
    
    # Create node trace - FIX: replace titleside with proper title structure
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=node_size,
            color=color_values,
            colorbar=dict(
                thickness=15,
                title='Community'  # Fixed: removed titleside, using only title
            ),
            line=dict(width=2, color='black')
        )
    )
    
    # Create figure
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title='Team Similarity Network',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            annotations=[
                dict(
                    text="Nodes = Teams<br>Edges = Statistical Similarity<br>Size = Connectivity",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0, y=-0.07
                )
            ]
        )
    )
    
    return fig

def display_network_centrality(G, all_teams_df, selected_teams, selected_metrics):
    """Display centrality analysis for network"""
    st.markdown("### Team Centrality Analysis")
    st.markdown("Teams with high centrality are statistically similar to many other teams.")
    
    # Calculate centrality metrics
    degree_dict = dict(nx.degree(G))
    try:
        betweenness = nx.betweenness_centrality(G)
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06, weight='weight')
    except:
        betweenness = {node: 0 for node in G.nodes()}
        eigenvector = {node: 0 for node in G.nodes()}
    
    # Create dataframe for display
    centrality_df = pd.DataFrame({
        'Team': list(G.nodes()),
        'Season': [G.nodes[node].get('season', 'N/A') for node in G.nodes()],
        'Degree': [degree_dict[node] for node in G.nodes()],
        'Betweenness': [betweenness[node] for node in G.nodes()],
        'Eigenvector': [eigenvector[node] for node in G.nodes()]
    }).sort_values('Degree', ascending=False)
    
    # Format for display
    for col in ['Betweenness', 'Eigenvector']:
        centrality_df[col] = centrality_df[col].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(centrality_df, use_container_width=True)
    
    # Find similar teams
    st.markdown("### Team Similarity Analysis")
    
    for team in selected_teams:
        # Find matching teams
        team_matches = all_teams_df[all_teams_df['Team'] == team]
        
        if not team_matches.empty:
            for _, team_row in team_matches.iterrows():
                team_season_label = team_row['Team_Season'] if 'Team_Season' in team_row else team_row['Team']
                
                if team_season_label in G:
                    similar_teams = []
                    for neighbor in G.neighbors(team_season_label):
                        weight = G[team_season_label][neighbor]['weight'] if 'weight' in G[team_season_label][neighbor] else 0
                        similar_teams.append({
                            'Team': neighbor,
                            'Similarity Score': weight
                        })
                    
                    if similar_teams:
                        similar_df = pd.DataFrame(similar_teams).sort_values('Similarity Score', ascending=False)
                        similar_df['Similarity Score'] = similar_df['Similarity Score'].apply(lambda x: f"{x:.3f}")
                        
                        st.markdown(f"#### Teams Most Similar to {team_season_label}")
                        st.dataframe(similar_df.head(5), use_container_width=True)