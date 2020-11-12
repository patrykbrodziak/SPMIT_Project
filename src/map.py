from typing import Optional

import pandas as pd
from plotly import graph_objects as go


class Map:
    def __init__(self):
        self.figure = go.Figure()

    def __call__(self, scatter_df: Optional[pd.DataFrame] = None, traces_df: Optional[pd.DataFrame] = None):
        """Show map with scatter locations and connections between them"""
        if scatter_df is not None:
            self.scatter_points(scatter_df)

        if traces_df is not None:
            self.draw_connections(traces_df)

        self.figure.show()

    def scatter_points(self, frame: pd.DataFrame) -> None:
        """
        Scatter points on map

        :param frame: tidy form DataFrame with columns: `long`, `lat`, `city`
        """
        self.figure.add_trace(
            go.Scattermapbox(
                lon=frame["long"],
                lat=frame["lat"],
                hoverinfo="text",
                text=frame["city"],
                mode="markers",
                marker=dict(size=10, color="rgb(255, 0, 0)",),
            )
        )

        self.figure.update_layout(
            showlegend=False,
            geo=dict(
                scope="north america",
                projection_type="azimuthal equal area",
                showland=True,
                landcolor="rgb(243, 243, 243)",
                countrycolor="rgb(204, 204, 204)",
            ),
        )

        self.figure.update_layout(
            margin={"l": 0, "t": 0, "b": 0, "r": 0},
            mapbox={"center": {"lon": 20, "lat": 50}, "style": "open-street-map", "zoom": 5},
        )

    def draw_connections(self, frame: pd.DataFrame) -> None:
        """Draw connections on map"""
        for i in range(len(frame)):
            self.figure.add_trace(
                go.Scattermapbox(
                    lon=[frame["start_lon"][i], frame["end_lon"][i]],
                    lat=[frame["start_lat"][i], frame["end_lat"][i]],
                    mode="lines",
                    text="",
                )
            )
