import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import community.community_louvain as community_louvain
from community import best_partition
import tempfile
import os

# ================ HELPER FUNCTIONS ================

def preprocess_data(data, features, scaler=None):
    """Preprocess data with standardization"""
    X = data[features].fillna(0).values
    
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler
    else:
        X_scaled = scaler.transform(X)
        return X_scaled, scaler

def create_pca_projection(X_scaled, n_components=2):
    """Apply PCA dimensionality reduction"""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca

def calculate_similarity(data_df, features, entity_col='player_id'):
    """Calculate similarity matrix for entities (players or teams)"""
    # Prepare data for similarity calculation
    stats_with_ids = data_df[[entity_col] + features].fillna(0)
    
    # Normalize the features
    X_scaled, _ = preprocess_data(stats_with_ids, features)
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(X_scaled)
    
    # Create mapping from ID to index
    id_to_idx = {pid: i for i, pid in enumerate(stats_with_ids[entity_col])}
    
    return similarity_matrix, id_to_idx

def filter_numeric_columns(df, exclude_terms=None):
    """Filter numeric columns excluding specified terms"""
    if exclude_terms is None:
        exclude_terms = ['team_id', 'id', 'season', 'Team ID', 'player_id']
        
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    return [col for col in numeric_cols if not any(term in col for term in exclude_terms)]

# ================ PLAYER CLUSTERING ================

def perform_clustering(player_stats, features, num_clusters):
    """Perform K-means clustering on player stats"""
    # Preprocess data
    X_scaled, _ = preprocess_data(player_stats, features)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster assignment to the dataframe
    player_stats_with_clusters = player_stats.copy()
    player_stats_with_clusters['cluster'] = clusters
    
    return player_stats_with_clusters

def render_player_clustering(player_df, stats_df, teams_df):
    """Render player clustering analysis"""
    st.title("Player Clustering")
    st.write("Group similar players using machine learning clustering techniques")
    
    # Season selection
    selected_season = st.selectbox(
        "Select season for clustering",
        options=sorted(stats_df['season'].unique(), reverse=True),
        key="cluster_season"
    )
    
    # Get stats for selected season
    season_stats = stats_df[stats_df['season'] == selected_season]
    
    if season_stats.empty:
        st.warning(f"No data available for the {selected_season} season.")
        return
        
    # Join with player info
    player_stats = season_stats.merge(
        player_df[['id', 'full_name']],
        left_on='player_id',
        right_on='id',
        how='inner'
    )
    
    # Add team information if available
    if 'team_id' in season_stats.columns and not teams_df.empty:
        player_stats['team_id_str'] = player_stats['team_id'].astype(str)
        teams_df_copy = teams_df.copy()
        teams_df_copy['id_str'] = teams_df_copy['id'].astype(str)
        
        player_stats = player_stats.merge(
            teams_df_copy[['id_str', 'nickname']],
            left_on='team_id_str',
            right_on='id_str',
            how='left'
        )
        player_stats['team'] = player_stats['nickname'].fillna('Unknown')
    else:
        player_stats['team'] = 'Unknown'
    
    # Feature selection
    st.subheader("Select Features for Clustering")
    
    feature_options = {
        'ppg': ('Points Per Game', True),
        'rpg': ('Rebounds Per Game', True),
        'apg': ('Assists Per Game', True),
        'spg': ('Steals Per Game', False),
        'bpg': ('Blocks Per Game', False),
        'fg_pct': ('Field Goal %', False)
    }
    
    col1, col2 = st.columns(2)
    selected_features = []
    
    # Create checkboxes and collect selected features
    for i, (feature, (label, default)) in enumerate(feature_options.items()):
        with col1 if i < 3 else col2:
            if st.checkbox(label, value=default, key=f"cluster_{feature}"):
                selected_features.append(feature)
    
    # Number of clusters
    num_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=5, key="num_clusters")
    
    if len(selected_features) >= 2 and st.button("Run Clustering"):
        with st.spinner("Clustering players..."):
            # Perform clustering
            clusters = perform_clustering(player_stats, selected_features, num_clusters)
            
            # Display results
            st.subheader("Clustering Results")
            
            # Display number of players in each cluster
            cluster_counts = clusters['cluster'].value_counts().sort_index()
            
            col1, col2 = st.columns([3, 1])
            with col1:
                # Create bar chart of cluster sizes
                fig = px.bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    labels={'x': 'Cluster', 'y': 'Number of Players'},
                    title="Players per Cluster"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Display counts
                for cluster, count in cluster_counts.items():
                    st.metric(f"Cluster {cluster}", count)
            
            # Create PCA visualization
            X_scaled, _ = preprocess_data(clusters, selected_features)
            X_pca, pca = create_pca_projection(X_scaled)
            
            pca_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Cluster': clusters['cluster'],
                'Player': clusters['full_name'],
                'Team': clusters['team']
            })
            
            # Add original features to hover data
            for feature in selected_features:
                pca_df[feature] = clusters[feature]
                
            # Create scatter plot
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='Cluster',
                hover_name='Player',
                hover_data=['Team'] + selected_features,
                title="Player Clusters - PCA Visualization",
                color_continuous_scale=px.colors.qualitative.G10
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display explained variance
            explained_variance = pca.explained_variance_ratio_
            st.info(f"PC1 explains {explained_variance[0]*100:.1f}% of the variance, PC2 explains {explained_variance[1]*100:.1f}%")
            
            # Show cluster centers
            cluster_centers = clusters.groupby('cluster')[selected_features].mean().reset_index()
            st.write("### Cluster Centers (Average Statistics)")
            
            # Format for display
            for feature in selected_features:
                if feature.endswith('_pct'):
                    cluster_centers[feature] = cluster_centers[feature].round(3)
                else:
                    cluster_centers[feature] = cluster_centers[feature].round(1)
            
            st.dataframe(cluster_centers, use_container_width=True)
            
            # Show players in each cluster
            st.write("### Players by Cluster")
            selected_cluster = st.selectbox(
                "Select cluster to view players",
                options=sorted(clusters['cluster'].unique()),
                key="cluster_select"
            )
            
            # Filter to selected cluster and display
            cluster_players = clusters[clusters['cluster'] == selected_cluster]
            st.dataframe(
                cluster_players[['full_name', 'team'] + selected_features].sort_values('full_name'),
                use_container_width=True
            )

# ================ SIMILARITY ANALYSIS ================

def find_similar_entities(entity_id, data_df, entity_info_df, features, entity_col='player_id', name_col='full_name', top_n=10):
    """Find similar entities (players or teams) based on statistical similarity"""
    entity_data = data_df[data_df[entity_col] == entity_id]
    
    if entity_data.empty:
        return pd.DataFrame()
    
    # Calculate similarity matrix
    similarity_matrix, id_to_idx = calculate_similarity(data_df, features, entity_col)
    
    entity_idx = id_to_idx.get(entity_id)
    if entity_idx is None:
        return pd.DataFrame()
    
    # Get similarities for the target entity
    similarities = similarity_matrix[entity_idx]
    
    # Create dataframe with similarities
    similarity_df = pd.DataFrame({
        entity_col: data_df[entity_col].values,
        'similarity': similarities
    })
    
    # Add entity info
    similarity_df = similarity_df.merge(
        entity_info_df,
        left_on=entity_col,
        right_on='id' if entity_col != 'id' else entity_col,
        how='left'
    ).merge(
        data_df[[entity_col] + features],
        on=entity_col,
        how='left'
    )
    
    # Remove the entity itself and sort by similarity
    similarity_df = similarity_df[similarity_df[entity_col] != entity_id]
    similarity_df = similarity_df.sort_values('similarity', ascending=False).head(top_n)
    
    return similarity_df

def render_player_similarity(player_df, stats_df, teams_df):
    """Render player similarity analysis based on statistical profiles"""
    st.title("Player Similarity Network Analysis")
    st.write("Explore players with similar statistical profiles and playing styles")
    
    if stats_df.empty:
        st.info("No player statistics available for similarity analysis.")
        return
    
    # Select the most recent season for analysis
    available_seasons = sorted(stats_df['season'].unique(), reverse=True)
    selected_season = st.selectbox("Select Season", available_seasons, index=0)
    
    # Get data for selected season
    season_stats = stats_df[stats_df['season'] == selected_season]
    
    # Feature selection for similarity
    st.subheader("Select Features for Similarity Calculation")
    feature_cols = []
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.checkbox("Points Per Game", value=True):
            feature_cols.append('ppg')
        if st.checkbox("Rebounds Per Game", value=True):
            feature_cols.append('rpg')
        if st.checkbox("Assists Per Game", value=True):
            feature_cols.append('apg')
    
    with col2:
        if st.checkbox("Steals Per Game"):
            feature_cols.append('spg')
        if st.checkbox("Blocks Per Game"):
            feature_cols.append('bpg')
        if st.checkbox("Field Goal %"):
            feature_cols.append('fg_pct')
    
    with col3:
        if st.checkbox("3-Point %"):
            feature_cols.append('fg3_pct')
        if st.checkbox("Free Throw %"):
            feature_cols.append('ft_pct')
        if st.checkbox("Minutes Per Game", value=True):
            feature_cols.append('minutes')
    
    # Filter for minimum minutes
    min_minutes = st.slider("Minimum Minutes Per Game", 5, 40, 15)
    
    # Similarity threshold
    similarity_threshold = st.slider(
        "Similarity Threshold", 
        min_value=0.5, 
        max_value=0.95, 
        value=0.75,
        step=0.05,
        help="Higher values show fewer connections but with stronger similarity"
    )
    
    # Network size control for performance
    max_players = st.slider(
        "Maximum Players in Network", 
        min_value=50, 
        max_value=600, 
        value=200,
        step=50,
        help="Limit the network size for better performance"
    )
    
    # Player selection
    qualified_players = season_stats[season_stats['minutes'] >= min_minutes]
    
    if len(qualified_players) == 0:
        st.warning(f"No players found with at least {min_minutes} minutes per game.")
        return
    
    # Get all player names for the dropdown
    # Join with player_df to get full names
    if 'player_id' in qualified_players.columns and 'id' in player_df.columns:
        # Ensure consistent types for join
        qualified_players['player_id'] = qualified_players['player_id'].astype('int64')
        player_df_copy = player_df.copy()
        player_df_copy['id'] = player_df_copy['id'].astype('int64')
        
        player_names_df = qualified_players.merge(
            player_df_copy[['id', 'full_name']], 
            left_on='player_id', 
            right_on='id',
            how='left'
        )
        player_names = player_names_df['full_name'].dropna().unique()
    else:
        st.error("Missing required ID columns in data. Please check the data format.")
        return
    
    # Sort player names alphabetically
    player_names = sorted(player_names)
    
    # Let user select a player
    selected_player = st.selectbox("Select a Player", player_names)
    
    # Check if we have enough features and data
    if len(feature_cols) < 2:
        st.warning("Please select at least 2 features for similarity analysis.")
        return
    
    # Calculate and display similar players
    if st.button("Find Similar Players"):
        with st.spinner("Calculating player similarities..."):
            # Get player ID
            selected_player_data = player_names_df[player_names_df['full_name'] == selected_player]
            if selected_player_data.empty:
                st.error(f"Could not find {selected_player} in the data.")
                return
                
            player_id = selected_player_data['player_id'].iloc[0]
            
            # Build network
            G, valid_data = build_network(
                qualified_players, 
                player_df, 
                feature_cols, 
                similarity_threshold=similarity_threshold,
                max_connections=15
            )
            
            if G is None or valid_data is None:
                st.error("Failed to build player similarity network.")
                return
            
            # Get similar players
            similar_players = find_similar_entities(
                player_id, 
                qualified_players, 
                player_df, 
                feature_cols, 
                top_n=15
            )
            
            # Display similar players
            st.subheader(f"Players Most Similar to {selected_player}")
            
            # Format similarity as percentage
            selected_features = feature_cols
            if 'similarity' in similar_players.columns:
                similar_players['Similarity'] = similar_players['similarity'].apply(lambda x: f"{x:.1%}")
            
            # Create visualization tabs
            net_tab, list_tab = st.tabs(["Network Visualization", "Similar Players"])
            
            with net_tab:
                # Render network visualization
                visualize_network(G, 'player', selected_features)
                
            with list_tab:
                # Display similar players table
                if not similar_players.empty:
                    st.dataframe(
                        similar_players[['full_name', 'Similarity'] + selected_features],
                        use_container_width=True
                    )
                    
                    # Radar chart comparison with most similar player
                    if not similar_players.empty:
                        most_similar = similar_players.iloc[0]
                        st.write(f"### Comparison with Most Similar Player: {most_similar['full_name']}")
                        
                        # Get stats for both players
                        player_stats = season_stats[season_stats['player_id'] == player_id]
                        similar_stats = season_stats[season_stats['player_id'] == most_similar['player_id']]
                        
                        if not player_stats.empty and not similar_stats.empty:
                            # Create radar chart
                            fig = go.Figure()
                            
                            # Calculate stats range for proper scaling
                            min_vals = season_stats[selected_features].min()
                            max_vals = season_stats[selected_features].max()
                            
                            # Get normalized values for both players
                            player_values = []
                            similar_values = []
                            
                            for feature in selected_features:
                                denom = max_vals[feature] - min_vals[feature]
                                if denom == 0:
                                    denom = 1  # Avoid division by zero
                                
                                player_val = (player_stats[feature].iloc[0] - min_vals[feature]) / denom
                                similar_val = (similar_stats[feature].iloc[0] - min_vals[feature]) / denom
                                
                                player_values.append(player_val)
                                similar_values.append(similar_val)
                            
                            # Add traces for both players
                            fig.add_trace(go.Scatterpolar(
                                r=player_values,
                                theta=selected_features,
                                fill='toself',
                                name=selected_player
                            ))
                            
                            fig.add_trace(go.Scatterpolar(
                                r=similar_values,
                                theta=selected_features,
                                fill='toself',
                                name=most_similar['full_name']
                            ))
                            
                            # Update layout
                            fig.update_layout(
                                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                showlegend=True,
                                title="Statistical Profile Comparison"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add detailed side-by-side statistics comparison
                            st.write("### Side-by-Side Statistical Comparison")
                            
                            # Get all available statistics columns to compare
                            all_stat_columns = [
                                'ppg', 'rpg', 'apg', 'spg', 'bpg', 'fg_pct', 
                                'fg3_pct', 'ft_pct', 'minutes', 'games_played'
                            ]
                            
                            # Filter to columns that exist in our data
                            available_stats = [col for col in all_stat_columns 
                                              if col in player_stats.columns and col in similar_stats.columns]
                            
                            # Create comparison dataframe
                            comparison_data = []
                            
                            for stat in available_stats:
                                player_val = player_stats[stat].iloc[0]
                                similar_val = similar_stats[stat].iloc[0]
                                
                                # Calculate difference and percentage difference
                                diff = player_val - similar_val
                                if similar_val != 0:
                                    pct_diff = (diff / similar_val) * 100
                                else:
                                    pct_diff = 0
                                
                                # Format the stat name for display
                                stat_name = stat.upper() if stat in ['ppg', 'rpg', 'apg', 'spg', 'bpg'] else stat
                                if stat.endswith('_pct'):
                                    stat_name = f"{stat.split('_')[0].upper()}%"
                                elif stat == 'minutes':
                                    stat_name = "MPG"
                                elif stat == 'games_played':
                                    stat_name = "Games"
                                
                                # Format values based on stat type
                                player_display = f"{player_val:.1f}" if stat not in ['games_played'] else f"{int(player_val)}"
                                similar_display = f"{similar_val:.1f}" if stat not in ['games_played'] else f"{int(similar_val)}"
                                
                                if stat.endswith('_pct'):
                                    player_display = f"{player_val:.3f}"
                                    similar_display = f"{similar_val:.3f}"
                                
                                # Determine who's better
                                better = None
                                if diff > 0 and stat != 'tov':  # For turnovers, lower is better
                                    better = "player"
                                elif diff < 0 and stat != 'tov':
                                    better = "similar"
                                elif diff < 0 and stat == 'tov':  # Handle turnovers differently
                                    better = "player"
                                elif diff > 0 and stat == 'tov':
                                    better = "similar"
                                
                                comparison_data.append({
                                    "Statistic": stat_name,
                                    selected_player: player_display,
                                    most_similar['full_name']: similar_display,
                                    "Difference": f"{diff:+.2f}",
                                    "% Diff": f"{pct_diff:+.1f}%",
                                    "Better": better
                                })
                            
                            # Convert to DataFrame and display
                            comparison_df = pd.DataFrame(comparison_data)
                            
                            # Drop the Better column for display
                            display_df = comparison_df.drop(columns=['Better'])

                            # Use Streamlit's DataFrame styling instead of HTML
                            def color_differences(val):
                                if isinstance(val, str):
                                    if val.startswith('+'):
                                        return 'color: green; font-weight: bold'
                                    elif val.startswith('-'):
                                        return 'color: red; font-weight: bold'
                                return ''

                            # Apply the styling to specific columns
                            styled_df = display_df.style.applymap(color_differences, subset=['Difference', '% Diff'])

                            # Display the styled dataframe
                            st.dataframe(styled_df)

                            # Add interpretation
                            st.markdown(f"""
                            #### Key Insights:
                            - Values in <span style='color: green; font-weight: bold'>green</span> indicate where **{selected_player}** has better stats
                            - Values in <span style='color: red; font-weight: bold'>red</span> indicate where **{most_similar['full_name']}** has better stats
                            - The percentage difference shows how much stronger one player is in each category
                            """, unsafe_allow_html=True)

# ================ NETWORK ANALYSIS ================

def build_network(data_df, entity_info_df, features, entity_col='player_id', name_col='full_name',
                 similarity_threshold=0.7, max_connections=15):
    """Build a network of entities (players or teams) based on statistical similarity"""
    if data_df.empty:
        return None, None
        
    # Get entities with complete data
    valid_data = data_df.dropna(subset=features)
    
    if len(valid_data) < 5:  # Need minimum number of entities
        return None, None
    
    # Join with entity info if needed
    if entity_col != 'Team':
        valid_data = valid_data.merge(
            entity_info_df[['id', name_col]],
            left_on=entity_col,
            right_on='id',
            how='inner'
        )
    
    # Calculate similarity matrix
    similarity_matrix, _ = calculate_similarity(valid_data, features, entity_col)
    
    # Create network
    G = nx.Graph()
    
    # Add nodes
    for i, entity in valid_data.iterrows():
        entity_id = entity[entity_col]
        entity_name = entity[name_col] if name_col in entity else entity[entity_col]
        
        # Add node attributes
        attrs = {
            'name': entity_name,
            **{feat: entity[feat] for feat in features if feat in entity}
        }
        G.add_node(entity_id, **attrs)
    
    # Add edges (similarities above threshold)
    entity_ids = valid_data[entity_col].tolist()
    
    for i in range(len(entity_ids)):
        # Sort similarities for this entity
        entity_similarities = [(entity_ids[j], similarity_matrix[i, j]) 
                            for j in range(len(entity_ids)) if i != j]
        entity_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Add edges for top connections
        for target_id, sim in entity_similarities[:max_connections]:
            if sim >= similarity_threshold:
                G.add_edge(entity_ids[i], target_id, weight=sim)
    
    # Apply community detection
    partition = best_partition(G)
    nx.set_node_attributes(G, partition, 'community')
    
    return G, valid_data

def render_player_graph_network(player_df, stats_df, teams_df):
    """Render player relationship graph based on statistical similarity"""
    st.title("Player Relationship Network")
    st.write("Explore player connections based on statistical similarity using graph mining techniques")
    
    # Season selection
    selected_season = st.selectbox(
        "Select season for network analysis",
        options=sorted(stats_df['season'].unique(), reverse=True),
        index=0,
        key="network_season"
    )
    
    # Get stats for selected season
    season_stats = stats_df[stats_df['season'] == selected_season]
    
    if season_stats.empty:
        st.warning(f"No data available for the {selected_season} season.")
        return
    
    # Feature selection
    st.subheader("Select Features for Graph Connections")
    
    feature_options = {
        'ppg': ('Points Per Game', True),
        'rpg': ('Rebounds Per Game', True),
        'apg': ('Assists Per Game', True),
        'spg': ('Steals Per Game', False),
        'bpg': ('Blocks Per Game', False),
        'fg_pct': ('Field Goal %', False)
    }
    
    col1, col2 = st.columns(2)
    selected_features = []
    
    for i, (feature, (label, default)) in enumerate(feature_options.items()):
        with col1 if i < 3 else col2:
            if feature in stats_df.columns and st.checkbox(label, value=default, key=f"graph_{feature}"):
                selected_features.append(feature)
    
    # Graph parameters
    col1, col2 = st.columns(2)
    with col1:
        similarity_threshold = st.slider(
            "Similarity threshold", 
            min_value=0.5, 
            max_value=0.95, 
            value=0.7, 
            step=0.05,
            key="similarity_threshold"
        )
    with col2:
        max_connections = st.slider(
            "Max connections per player", 
            min_value=3, 
            max_value=20, 
            value=8,
            key="max_connections"
        )
    
    if len(selected_features) >= 2 and st.button("Generate Player Network"):
        with st.spinner("Building player network graph..."):
            # Build network
            G, valid_players = build_network(
                season_stats, 
                player_df, 
                selected_features, 
                similarity_threshold=similarity_threshold, 
                max_connections=max_connections
            )
            
            if G is not None and valid_players is not None:
                # Network visualization and analysis
                visualize_network(G, 'player', selected_features)
            else:
                st.warning("Insufficient data to build player network for the selected season and features.")

# ================ TEAM NETWORK ANALYSIS ================

def create_team_stat_network(team_stats_df, stats_to_include=None, similarity_threshold=0.7):
    """Create a network graph of teams based on statistical similarity"""
    # If no stats specified, use numeric columns
    if not stats_to_include:
        stats_to_include = filter_numeric_columns(team_stats_df)
    
    # Ensure Team column exists for node labels
    if 'Team' not in team_stats_df.columns:
        return None, "Missing 'Team' column in data"
    
    # Check if we have enough stats columns
    if len(stats_to_include) < 2:
        return None, "Not enough statistical columns for comparison"
    
    # Create network
    G, _ = build_network(
        team_stats_df, 
        team_stats_df,  # team info is in the same dataframe
        stats_to_include, 
        entity_col='Team', 
        name_col='Team',
        similarity_threshold=similarity_threshold
    )
    
    if G is None:
        return None, "Failed to build team network"
    
    # Get node positions using a force-directed layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Create visualization data
    edge_x, edge_y = [], []
    edge_text = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        team1, team2 = edge[0], edge[1]
        similarity = G.edges[edge]['weight']
        
        edge_text.append(f"{team1} - {team2}: {similarity:.2f} similarity")
    
    # Create node data
    node_x, node_y = [], []
    node_text = []
    node_size = []
    node_color = []
    
    # Find the most important stats for each team
    team_key_stats = {}
    for team in G.nodes():
        team_stats = {stat: G.nodes[team].get(stat, 0) for stat in stats_to_include}
        # Find stats where this team exceeds the mean
        stat_diffs = {}
        for stat, value in team_stats.items():
            if stat in team_stats_df.columns:
                mean_value = team_stats_df[stat].mean()
                if mean_value > 0:  # Avoid division by zero
                    stat_diffs[stat] = (value - mean_value) / mean_value
        
        # Get top stats
        top_stats = sorted(stat_diffs.items(), key=lambda x: x[1], reverse=True)[:3]
        team_key_stats[team] = ", ".join([f"{stat}: {team_stats[stat]:.1f}" for stat, _ in top_stats])
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Create hover text with key team stats
        stats_text = "<br>".join([f"{stat}: {G.nodes[node].get(stat, 0):.2f}" 
                                for stat in stats_to_include[:5]
                                if stat in G.nodes[node]])
        
        hover_text = f"<b>{node}</b><br><br>Key stats:<br>{stats_text}"
        if node in team_key_stats:
            hover_text += f"<br><br>Strongest attributes:<br>{team_key_stats[node]}"
            
        node_text.append(hover_text)
        
        # Size based on degree centrality
        size = 15 + 5 * G.degree(node)
        node_size.append(size)
        
        # Color based on community
        partition = nx.get_node_attributes(G, 'community')
        node_color.append(partition[node])
    
    # Create plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='rgba(150,150,150,0.7)'),
        hoverinfo='text',
        text=edge_text,
        mode='lines',
        name='Similarity connections'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_color,
            size=node_size,
            colorbar=dict(
                thickness=15,
                title=dict(
                    text="Community", 
                    side="right"
                ),
                xanchor='left'
            ),
            line=dict(width=2, color='DarkSlateGrey')
        ),
        text=[G.nodes[n]['name'] for n in G.nodes()],
        hoverinfo='text',
        hovertext=node_text,
        textposition="top center",
        name='Teams'
    ))
    
    # Calculate graph metrics
    partition = nx.get_node_attributes(G, 'community')
    metrics = {
        "Communities": len(set(partition.values())),
        "Avg Degree": sum(dict(G.degree()).values()) / len(G),
        "Density": nx.density(G),
        "Connected Components": nx.number_connected_components(G),
        "Community_Assignment": partition
    }
    
    # Update layout
    fig.update_layout(
        title=f"NBA Team Statistical Similarity Network<br><sup>Communities: {metrics['Communities']}, " + 
              f"Avg Connections: {metrics['Avg Degree']:.1f}, Density: {metrics['Density']:.2f}</sup>",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=60),
        annotations=[
            dict(
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                text="Node size = Connection count, Node color = Statistical community",
                font=dict(size=10)
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    return fig, metrics

def analyze_team_graph_metrics(team_stats_df, stats_to_include=None):
    """Analyze team relationships using graph theory metrics"""
    # If no stats specified, use numeric columns
    if not stats_to_include:
        stats_to_include = filter_numeric_columns(team_stats_df)
    
    # Build network
    G, _ = build_network(
        team_stats_df, 
        team_stats_df,
        stats_to_include, 
        entity_col='Team', 
        name_col='Team',
        similarity_threshold=0.7
    )
    
    if G is None or len(G.nodes()) < 2:
        # Return empty metrics
        return pd.DataFrame({
            'Team': team_stats_df['Team'].tolist(),
            'Degree Centrality': [0] * len(team_stats_df),
            'Betweenness Centrality': [0] * len(team_stats_df),
            'Eigenvector Centrality': [0] * len(team_stats_df),
            'Clustering Coefficient': [0] * len(team_stats_df),
        })
    
    # Calculate centrality metrics
    try:
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
        clustering_coefficient = nx.clustering(G)
        
        # Create DataFrame with metrics
        metrics_df = pd.DataFrame({
            'Team': list(degree_centrality.keys()),
            'Degree Centrality': list(degree_centrality.values()),
            'Betweenness Centrality': list(betweenness_centrality.values()),
            'Eigenvector Centrality': list(eigenvector_centrality.values()),
            'Clustering Coefficient': list(clustering_coefficient.values()),
        })
    except Exception as e:
        # Fallback for disconnected graphs
        metrics_df = pd.DataFrame({
            'Team': [node for node in G.nodes()],
            'Degree Centrality': [G.degree(node) / (len(G) - 1) if len(G) > 1 else 0 for node in G.nodes()],
            'Betweenness Centrality': [0] * len(G),
            'Eigenvector Centrality': [0] * len(G),
            'Clustering Coefficient': [0] * len(G),
        })
    
    # Sort by degree centrality
    metrics_df = metrics_df.sort_values('Degree Centrality', ascending=False)
    
    # Add interpretation columns
    metrics_df['Statistical Influence'] = metrics_df['Degree Centrality'].apply(
        lambda x: 'Very High' if x > 0.8 else 'High' if x > 0.6 else 'Medium' if x > 0.4 else 'Low'
    )
    
    metrics_df['Style Uniqueness'] = metrics_df['Clustering Coefficient'].apply(
        lambda x: 'Very Unique' if x < 0.2 else 'Somewhat Unique' if x < 0.5 else 'Similar to Others'
    )
    
    return metrics_df

def identify_similar_teams(team_name, team_stats_df, stats_to_include=None, top_n=5):
    """Find teams most similar to a given team based on statistical profile"""
    # If no stats specified, use numeric columns
    if not stats_to_include:
        stats_to_include = filter_numeric_columns(team_stats_df)
    
    # Ensure team exists
    if team_name not in team_stats_df['Team'].values:
        return pd.DataFrame({'Team': [], 'Similarity': []})
    
    # Simplify by using the general similarity function
    similarities = find_similar_entities(
        team_name, 
        team_stats_df, 
        team_stats_df,  # Team info is in the same dataframe
        stats_to_include,
        entity_col='Team',
        name_col='Team',
        top_n=top_n
    )
    
    # Keep only essential columns
    if not similarities.empty:
        return similarities[['Team', 'similarity']].rename(columns={'similarity': 'Similarity'})
    else:
        return pd.DataFrame({'Team': [], 'Similarity': []})

def visualize_network(G, entity_type='player', features=None):
    """Generic network visualization for both players and teams"""
    if G is None:
        st.warning(f"No {entity_type} network available.")
        return
        
    # Network statistics
    st.subheader("Network Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nodes", len(G.nodes()))
    with col2:
        st.metric("Connections", len(G.edges()))
    with col3:
        communities = len(set(nx.get_node_attributes(G, 'community').values()))
        st.metric("Communities", communities)
    
    # Calculate node statistics
    node_degrees = dict(G.degree())
    try:
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
    except:
        eigenvector_centrality = {node: 1.0/len(G) for node in G.nodes()}
    
    # Add metrics to graph
    nx.set_node_attributes(G, node_degrees, 'connections')
    nx.set_node_attributes(G, eigenvector_centrality, 'centrality')
    
    # Create node trace data for plotly
    pos = nx.spring_layout(G, seed=42)
    
    # Create edge traces
    edge_x, edge_y = [], []
    edge_text = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # entity1 = G.nodes[edge[0]]['name']
        # entity2 = G.nodes[edge[1]]['name']
        similarity = G.edges[edge]['weight']
        
        edge_text.append(f"{edge[0]} - {edge[1]}: {similarity:.2f} similarity")
    
    # Create node traces
    node_x, node_y = [], []
    node_text = []
    node_size = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Node information
        node_name = G.nodes[node]['name']
        degree = G.nodes[node]['connections']
        centrality = G.nodes[node]['centrality']
        community = G.nodes[node]['community']
        
        # Create hover text
        hover_text = f"{entity_type.title()}: {node_name}<br>Connections: {degree}<br>Centrality: {centrality:.3f}"
        if features:
            for feat in features:
                if feat in G.nodes[node]:
                    stat_value = G.nodes[node][feat]
                    stat_format = "{:.3f}" if feat.endswith('_pct') else "{:.1f}"
                    hover_text += f"<br>{feat}: {stat_format.format(stat_value)}"
        
        node_text.append(hover_text)
        node_size.append(10 + (degree * 2))  # Size based on connections
        node_color.append(community)  # Color based on community
    
    # Create plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        text=edge_text,
        mode='lines'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_color,
            size=node_size,
            colorbar=dict(
                thickness=15,
                title=dict(
                    text="Community", 
                    side="right"
                ),
                xanchor='left'
            ),
            line=dict(width=2, color='DarkSlateGrey')
        )
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{entity_type.title()} Similarity Network",
            font=dict(size=16)
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Entity centrality table
    st.subheader(f"Most Central {entity_type.title()}s")
    
    # Create dataframe with centrality measures
    centrality_df = pd.DataFrame({
        'id': list(G.nodes()),
        'name': [G.nodes[n]['name'] for n in G.nodes()],
        'connections': [G.nodes[n]['connections'] for n in G.nodes()],
        'centrality': [G.nodes[n]['centrality'] for n in G.nodes()],
        'community': [G.nodes[n]['community'] for n in G.nodes()]
    })
    
    # Add stats
    if features:
        for feat in features:
            if all(feat in G.nodes[n] for n in G.nodes()):
                centrality_df[feat] = [G.nodes[n][feat] for n in G.nodes()]
    
    # Sort by centrality and show top entities
    top_central = centrality_df.sort_values('centrality', ascending=False).head(15)
    
    # Format
    top_central['centrality'] = top_central['centrality'].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(
        top_central[['name', 'connections', 'centrality', 'community'] + 
                   [f for f in features if f in top_central.columns]],
        use_container_width=True
    )

# Advanced network visualization with pyvis
def render_network_visualization(nodes, edges, selected_player, show_labels=True, physics_enabled=False, use_webgl=True):
    """Render network visualization using PyVis"""
    try:
        from pyvis.network import Network
        import streamlit.components.v1 as components
        
        # Create network
        net = Network(notebook=True, cdn_resources='remote', height="600px", width="100%", directed=False)
        
        # Configure network options
        net.set_options("""
        {
          "nodes": {
            "font": {
              "size": 12,
              "face": "Roboto",
              "color": "#333333"
            },
            "borderWidth": 2,
            "borderWidthSelected": 3,
            "opacity": 0.9
          },
          "edges": {
            "color": {
              "inherit": true,
              "opacity": 0.7
            },
            "smooth": {
              "enabled": true,
              "type": "continuous"
            }
          },
          "physics": {
            "enabled": %s,
            "barnesHut": {
              "gravitationalConstant": -3000,
              "centralGravity": 0.15,
              "springLength": 95,
              "springConstant": 0.05,
              "damping": 0.09
            },
            "solver": "barnesHut",
            "maxVelocity": 50,
            "minVelocity": 0.75,
            "timestep": 0.5
          },
          "interaction": {
            "tooltipDelay": 100,
            "hideEdgesOnDrag": true,
            "hover": true,
            "navigationButtons": false,
            "multiselect": false
          }
        }
        """ % ("true" if physics_enabled else "false"))
        
        # Add nodes and edges
        for node in nodes:
            # Hide labels if show_labels is False
            node_label = node.get('label', '') if show_labels else ''
            
            # Set node options
            net.add_node(
                node['id'], 
                label=node_label, 
                title=node.get('label', f"Player {node['id']}"),
                size=node.get('size', 10),
                group=node.get('group', 0),
                color=node.get('color', None)
            )
        
        for edge in edges:
            net.add_edge(
                edge['from'], 
                edge['to'], 
                value=edge.get('value', 1),
                title=edge.get('title', ''),
                color=edge.get('color', None)
            )
        
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
            temp_path = temp_file.name
            net.save_graph(temp_path)
            
            # Inject WebGL renderer for better performance if enabled
            if use_webgl:
                with open(temp_path, 'r') as f:
                    content = f.read()
                    
                if 'var options =' in content:
                    webgl_option = '"renderer": "webgl",'
                    content = content.replace('var options =', f'var options = {{{webgl_option}')
                    
                    with open(temp_path, 'w') as f:
                        f.write(content)
            
            # Display the graph
            with open(temp_path, 'r', encoding='utf-8') as f:
                html = f.read()
                components.html(html, height=600)
    
    except Exception as e:
        st.error(f"Error rendering network visualization: {str(e)}")
        st.info("Network visualization requires pyvis. Install with: pip install pyvis")
        
        # Fallback to simple display
        st.write(f"Network contains {len(nodes)} players and {len(edges)} connections")
        st.write(f"Selected player: {selected_player}")
        
        # Show first 10 nodes as a fallback
        player_list = [n.get('label', f"Player {n['id']}") for n in nodes[:10]]
        st.write("Players in network (first 10):")
        st.write(", ".join(player_list))

def calculate_player_similarity_network(qualified_players, player_df, feature_cols, similarity_threshold=0.75, max_players=200, central_player=None):
    """Calculate player similarity and create network data"""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity
    import networkx as nx
    import community as community_louvain
    
    try:
        # Prepare data
        complete_data = qualified_players.dropna(subset=feature_cols)
        
        # If we have too many players, limit to the most active ones
        if len(complete_data) > max_players and 'minutes' in complete_data.columns:
            complete_data = complete_data.nlargest(max_players, 'minutes')
        
        # Get player names
        player_df_copy = player_df.copy()
        if 'id' in player_df_copy.columns:
            player_df_copy['id'] = player_df_copy['id'].astype('int64')
        
        if 'player_id' in complete_data.columns:
            complete_data['player_id'] = complete_data['player_id'].astype('int64')
            player_data = complete_data.merge(
                player_df_copy[['id', 'full_name']], 
                left_on='player_id', 
                right_on='id',
                how='left'
            )
        else:
            return None
        
        # Convert features to normalized matrix
        X = player_data[feature_cols].values
        player_ids = player_data['player_id'].values
        player_names = player_data['full_name'].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calculate similarity matrix (cosine similarity)
        similarity_matrix = cosine_similarity(X_scaled)
        
        # Find the central player index if provided
        central_idx = None
        if central_player:
            central_idx = np.where(player_names == central_player)[0]
            if len(central_idx) == 0:
                central_idx = None
            else:
                central_idx = central_idx[0]
        
        # Create a graph
        G = nx.Graph()
        
        # Add nodes
        for i, (pid, name) in enumerate(zip(player_ids, player_names)):
            G.add_node(pid, name=name)
        
        # Add edges for similar players
        for i in range(len(similarity_matrix)):
            # If central player is specified, only add edges connected to them
            if central_idx is not None and i != central_idx:
                if similarity_matrix[i, central_idx] >= similarity_threshold:
                    G.add_edge(player_ids[i], player_ids[central_idx], 
                              weight=similarity_matrix[i, central_idx])
                continue
                
            # Otherwise add all edges above threshold
            if central_idx is None:  
                for j in range(i+1, len(similarity_matrix)):
                    if similarity_matrix[i, j] >= similarity_threshold:
                        G.add_edge(player_ids[i], player_ids[j], 
                                  weight=similarity_matrix[i, j])
        
        # Apply community detection
        partition = community_louvain.best_partition(G)
        
        # Prepare data for visualization
        nodes = []
        for node_id in G.nodes():
            player_idx = np.where(player_ids == node_id)[0][0]
            player_name = player_names[player_idx]
            
            # Set node properties
            node_data = {
                'id': int(node_id),
                'label': player_name,
                'group': partition[node_id] if node_id in partition else 0,
                'size': 10  # Default size
            }
            
            # Make the selected player node larger
            if player_name == central_player:
                node_data['size'] = 20
                node_data['color'] = '#FF5733'  # Highlight color
            
            nodes.append(node_data)
        
        edges = []
        for source, target, data in G.edges(data=True):
            edges.append({
                'from': int(source),
                'to': int(target),
                'value': float(data['weight']) * 5,  # Scale up for visibility
                'title': f"Similarity: {data['weight']:.2f}"
            })
        
        # Get similar players for the selected player
        similar_players = []
        if central_player and central_idx is not None:
            central_player_id = player_ids[central_idx]
            
            # Get neighbors from the graph
            neighbors = list(G.neighbors(central_player_id))
            
            # For each neighbor, get similarity and stats
            for neighbor_id in neighbors:
                neighbor_idx = np.where(player_ids == neighbor_id)[0][0]
                similarity = similarity_matrix[central_idx, neighbor_idx]
                
                # Get player stats
                player_stats = {
                    'name': player_names[neighbor_idx],
                    'similarity': similarity,
                }
                
                # Add selected features
                for feature in feature_cols:
                    player_stats[feature] = player_data.iloc[neighbor_idx][feature]
                
                similar_players.append(player_stats)
            
            # Sort by similarity
            similar_players.sort(key=lambda x: x['similarity'], reverse=True)
        
        return nodes, edges, similar_players
    
    except Exception as e:
        st.error(f"Error calculating player similarity: {str(e)}")
        return None, None, None