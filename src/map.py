from typing import Optional

import pandas as pd
from plotly import graph_objects as go


class Map:
    def __init__(self):
        self.figure = go.Figure()

    def __call__(self, scatter_df: Optional[pd.DataFrame] = None, traces: list = None):
        """Show map with scatter locations and connections between them"""
        if scatter_df is not None:
            self.scatter_points(scatter_df)

        if traces:
            self.draw_connections(scatter_df, traces)

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
                marker=dict(size=10, color="rgb(255, 0, 0)"),
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

    def draw_connections(self, scatter_df: pd.DataFrame, traces: list) -> None:
        """Draw connections on map

        :param scatter_df: DataFrame containing latitudes and longitudes of all scattered cities
                           with keys `lat` and `lon`
        :param traces: list of list of IDs of cities containing order in which they are traversed
        """
        self.colors = ["rgb(3, 0, 253)", "rgb(128, 0, 0)", "rgb(254, 254, 51)", "rgb(0, 255, 255)", "rgb(168, 8, 8)"]
        self.color = 0
        for trace in traces:
            for start_id, end_id in zip(trace[:-1], trace[1:]):
                self.figure.add_trace(
                    go.Scattermapbox(
                        lon=[scatter_df["long"][start_id], scatter_df["long"][end_id]],
                        lat=[scatter_df["lat"][start_id], scatter_df["lat"][end_id]],
                        mode="lines",
                        text="",
                        line=dict(color=self.colors[self.i])
                    )
                )
            self.i+=1
