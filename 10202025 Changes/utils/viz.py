from typing import List, Tuple
import threading
import webbrowser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

from utils.sensor import SensorData

def visually_align_signal(signal1: pd.Series, signal2: pd.Series, default_start1=0, default_start2=0, default_window=None):
    app = dash.Dash(__name__)
    max_index = min(len(signal1), len(signal2))
    max_tick = int(np.ceil(max_index / 1000.0)) * 1000
    tick_marks = {i: str(i) for i in range(0, max_tick + 1, 1000)}

    # Calculate default window size as 40% of the signal length
    if default_window is None:
        default_window = int(max_index * 0.4)

    # Create initial figure
    initial_start = default_start1
    initial_end = default_start1 + default_window
    initial_x = np.arange(initial_start, initial_end)
    
    # Convert signals to numpy arrays if they're pandas Series
    sig1 = signal1.values if isinstance(signal1, pd.Series) else signal1
    sig2 = signal2.values if isinstance(signal2, pd.Series) else signal2
    
    # Get the windowed portions of the signals
    sig1_window = sig1[initial_start:initial_end]

    sig2_window = sig2[initial_start:initial_end]
    
    # Create initial figure
    initial_fig = go.Figure()
    initial_fig.add_trace(go.Scatter(x=initial_x, y=sig1_window, mode='lines', name='accel - sensor', line=dict(color='#6B4E71')))
    initial_fig.add_trace(go.Scatter(x=initial_x, y=sig2_window, mode='lines', name='vgrf - treadmill', line=dict(color='#E67E22')))

    # Add offset annotation in the upper middle of the screen
    offset = initial_end - initial_start
    y_range = max(max(sig1_window), max(sig2_window)) - min(min(sig1_window), min(sig2_window))
    y_pos = min(min(sig1_window), min(sig2_window)) + y_range * 0.8  # Position at 80% of the height
    x_mid = initial_start + (initial_end - initial_start)/2
    
    initial_fig.add_annotation(
        x=x_mid,
        y=y_pos,
        text=f'Offset: {offset:+d}',  # +d format adds a + sign for positive numbers
        showarrow=False,
        font=dict(size=14, color='#2C3E50'),
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='#2C3E50',
        borderwidth=1,
        borderpad=4
    )

    initial_fig.update_xaxes(range=[initial_start, initial_end])
    initial_fig.update_layout(
        title="Signal Windows (Click to set window boundaries)",
        xaxis_title="Time (index)",
        yaxis_title="Value",
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#2C3E50',
            borderwidth=1
        ),
        margin=dict(t=40, b=40, l=40, r=40)
    )

    app.layout = html.Div([
        html.H2("Stomp Window Selector"),

        # Window selector slider
        html.Label("Select Stomp Window:"),
        dcc.RangeSlider(
            id='window-slider',
            min=0,
            max=max_tick,
            step=10,
            value=[default_start1, default_start1 + default_window],
            allowCross=False,
            tooltip={"placement": "bottom", "always_visible": True},
            marks=tick_marks
        ),

        # Offset input field
        html.Div([
            html.Label("Manual Offset:"),
            dcc.Input(
                id='offset-input',
                type='number',
                value=0,
                style={'width': '100px', 'marginLeft': '10px'}
            ),
        ], style={'marginTop': '20px', 'marginBottom': '20px'}),

        # Store for click state
        dcc.Store(id='click-state', data={'left': None, 'right': None}),

        dcc.Graph(
            id='time-series-plot',
            figure=initial_fig,
            config={'displayModeBar': False}  # Hide the modebar for cleaner interface
        )
    ], style={"width": "80%", "margin": "auto", "fontFamily": "Arial"})

    @app.callback(
        [Output('time-series-plot', 'figure'),
         Output('window-slider', 'value'),
         Output('click-state', 'data')],
        [Input('window-slider', 'value'),
         Input('time-series-plot', 'clickData'),
         Input('time-series-plot', 'hoverData'),
         Input('click-state', 'data'),
         Input('offset-input', 'value')],
        prevent_initial_call=False
    )
    def update(window_range, click_data, hover_data, current_click_state, manual_offset):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update

        # Get the trigger info
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        trigger_value = ctx.triggered[0]['value']
        trigger_prop = ctx.triggered[0]['prop_id'].split('.')[1]

        # Get current click state or initialize if None
        click_state = current_click_state if current_click_state else {'left': None, 'right': None}

        # Handle clicks - only process if it's a click event
        if trigger_id == 'time-series-plot' and trigger_prop == 'clickData' and trigger_value:
            click_x = trigger_value['points'][0]['x']
            
            # If neither boundary is set, set the left boundary
            if click_state['left'] is None and click_state['right'] is None:
                click_state['left'] = click_x
            # If left boundary is set but right isn't, set the right boundary
            elif click_state['left'] is not None and click_state['right'] is None:
                click_state['right'] = click_x
                # Ensure left is less than right
                if click_state['right'] < click_state['left']:
                    click_state['left'], click_state['right'] = click_state['right'], click_state['left']
                window_range = [click_state['left'], click_state['right']]
                click_state = {'left': None, 'right': None}
            # If right boundary is set but left isn't, set the left boundary
            elif click_state['left'] is None and click_state['right'] is not None:
                click_state['left'] = click_x
                # Ensure left is less than right
                if click_state['left'] > click_state['right']:
                    click_state['left'], click_state['right'] = click_state['right'], click_state['left']
                window_range = [click_state['left'], click_state['right']]
                click_state = {'left': None, 'right': None}
            # If both boundaries are set, start over with the new click as left boundary
            else:
                click_state = {'left': click_x, 'right': None}

        # Extract windows
        start, end = window_range

        # Create figure
        fig = go.Figure()
        
        # Add the windowed signals
        x = np.arange(start, end)
        
        # Convert signals to numpy arrays if they're pandas Series
        sig1 = signal1.values if isinstance(signal1, pd.Series) else signal1
        sig2 = signal2.values if isinstance(signal2, pd.Series) else signal2
        
        # Get the windowed portions of the signals
        sig1_window = sig1[start:end]
        sig2_window = sig2[start:end]
        
        # Add the original signals
        fig.add_trace(go.Scatter(x=x, y=sig1_window, mode='lines', name='accel', line=dict(color='#6B4E71')))
        fig.add_trace(go.Scatter(x=x, y=sig2_window, mode='lines', name='vgrf', line=dict(color='#E67E22')))

        # Add shifted signal if manual offset is provided
        if manual_offset is not None and manual_offset != 0:
            # Calculate the shifted x range
            x_shifted = x + manual_offset
            # Only plot the shifted signal within the visible window
            mask = (x_shifted >= start) & (x_shifted < end)
            if any(mask):
                fig.add_trace(go.Scatter(
                    x=x_shifted[mask],
                    y=sig2_window[mask],
                    mode='lines',
                    name='vgrf (shifted)',
                    line=dict(color='#E67E22', width=3, dash='dot'),
                    opacity=1
                ))

        # Add offset annotation in the upper middle of the screen
        offset = end - start
        y_range = max(max(sig1_window), max(sig2_window)) - min(min(sig1_window), min(sig2_window))
        y_pos = min(min(sig1_window), min(sig2_window)) + y_range * 0.8  # Position at 80% of the height
        x_mid = start + (end - start)/2
        
        fig.add_annotation(
            x=x_mid,
            y=y_pos,
            text=f'Offset: {offset:+d}',  # +d format adds a + sign for positive numbers
            showarrow=False,
            font=dict(size=14, color='#2C3E50'),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#2C3E50',
            borderwidth=1,
            borderpad=4
        )

        # Add hover line
        if hover_data:
            hover_x = hover_data['points'][0]['x']
            fig.add_vline(x=hover_x, line_dash="dot", line_color="#34495E", opacity=0.8)

        # Add vertical line for click marker if it exists
        if click_state['left'] is not None:
            fig.add_vline(x=click_state['left'], line_dash="dot", line_color="#3498DB", opacity=0.5)

        # Use the window range
        fig.update_xaxes(range=[start, end])

        fig.update_layout(
            title="Signal Windows (Click to set window boundaries)",
            xaxis_title="Time (index)",
            yaxis_title="Value",
            legend=dict(
                x=0.98,
                y=0.98,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='#2C3E50',
                borderwidth=1
            ),
            margin=dict(t=40, b=40, l=40, r=40)
        )

        return fig, window_range, click_state

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:8050/")

    threading.Timer(1, open_browser).start()
    # app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8050)
    app.run(debug=True, use_reloader=False, host='127.0.0.1', port=8050)