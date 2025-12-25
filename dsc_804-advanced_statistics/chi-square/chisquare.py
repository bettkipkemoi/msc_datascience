import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from scipy.stats import chi2

# Create a Dash application
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    # Title of the app
    html.H1("Interactive Chi-Square Distribution", style={'textAlign': 'center'}),

    # Slider for degrees of freedom
    dcc.Slider(
        id='df-slider',
        min=1,
        max=50,
        step=1,
        value=5,  # Default value
        marks={i: str(i) for i in range(1, 51, 5)},  # Marks at every 5th df
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    
    # Graph to display the Chi-Square distribution
    dcc.Graph(id='chi-square-graph')
])

# Define the callback to update the graph
@app.callback(
    Output('chi-square-graph', 'figure'),
    Input('df-slider', 'value')
)
def update_graph(df):
    # Define the x range (0 to 100)
    x = np.linspace(0, 100, 1000)
    
    # Calculate the Chi-Square PDF for the selected df
    y = chi2.pdf(x, df)
    
    # Create the plotly figure
    fig = go.Figure()

    # Add the Chi-Square distribution line to the graph
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'df = {df}'))
    
    # Update layout and titles
    fig.update_layout(
        title=f'Chi-Square Distribution for df = {df}',
        xaxis_title='x',
        yaxis_title='Probability Density',
        showlegend=True
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
