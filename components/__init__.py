"""
NBA Dashboard Components Package
--------------------------------
This package contains all visualization and analysis components 
for the NBA Player Performance Prediction dashboard.
"""

import traceback
import importlib
import importlib.util
import logging
import streamlit as st

# Configure component imports
_COMPONENTS = {
    # Analysis components
    'player_analysis': 'render_player_analysis',
    'team_comparison': 'render_team_comparison',
    'predictions': 'render_predictions',
    'trends': 'render_trends',
    
    # Data mining components
    'data_mining': ['render_player_clustering', 'render_player_similarity']
}

# Set up version and metadata
__version__ = '1.0.0'
__all__ = [
    'render_player_analysis',
    'render_team_comparison', 
    'render_predictions',
    'render_trends',
    'render_player_clustering',
    'render_player_similarity'
]

# Import components with structured error handling
_loaded_components = set()
_failed_components = {}

# Function factory to create error handlers
def create_error_func(error_msg, func_name):
    def error_handler(*args, **kwargs):
        st.error(f"{func_name} component failed to load: {error_msg}")
    return error_handler

# Custom error handler for team comparison (with specific signature)
def create_team_comparison_error(error_msg):
    def team_comparison_error(*args, team_stats_df=None, **kwargs):
        st.error(f"Team Comparison component failed to load: {error_msg}")
    return team_comparison_error

for module_name, functions in _COMPONENTS.items():
    try:
        if isinstance(functions, list):
            # Multiple functions from one module
            module = importlib.import_module(f'.{module_name}', package='components')
            for func_name in functions:
                globals()[func_name] = getattr(module, func_name)
                _loaded_components.add(func_name)
        else:
            # Single function from module
            func_name = functions
            module = importlib.import_module(f'.{module_name}', package='components')
            globals()[func_name] = getattr(module, func_name)
            _loaded_components.add(func_name)
    except Exception as e:
        # Track the error for reporting
        error_msg = str(e)
        _failed_components[module_name] = error_msg
        
        # Create fallback function(s) with appropriate signature
        if isinstance(functions, list):
            for func_name in functions:
                globals()[func_name] = create_error_func(error_msg, func_name)
        else:
            func_name = functions
            
            # Create custom fallback function with proper signature
            if func_name == 'render_team_comparison':
                globals()[func_name] = create_team_comparison_error(error_msg)
            else:
                globals()[func_name] = create_error_func(error_msg, func_name)

# Report success or failure
if _failed_components:
    print(f"WARNING: {len(_failed_components)} components failed to load:")
    for component, error in _failed_components.items():
        print(f"  - {component}: {error}")
else:
    print(f"Successfully loaded all {len(_loaded_components)} NBA dashboard components")

# Define helper function to check component status
def get_component_status():
    """Return the loading status of all components"""
    return {
        'loaded': list(_loaded_components),
        'failed': _failed_components,
        'version': __version__
    }