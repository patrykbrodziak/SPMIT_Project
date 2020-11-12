import pandas as pd
from plotly import graph_objects as go


class Map:
    def __init__(self, csv_file):
        self.cities2 = pd.read_csv(csv_file)
        fig = go.Figure()
        fig.add_trace(go.Scattermapbox(
            lon = self.cities2['long'],
            lat = self.cities2['lat'],
            hoverinfo = 'text',
            text = self.cities2['city'],
            mode = 'markers',
            marker = dict(
                size = 10,
                color = 'rgb(255, 0, 0)',
            )))
        # for i in range(len(connects)):
        #     fig.add_trace(
        #         go.Scattermapbox(
        #             lon = [connects['start_lon'][i], connects['end_lon'][i]],
        #             lat = [connects['start_lat'][i], connects['end_lat'][i]],
        #             mode = 'lines',
        #             text = cities['city'],
        #         )
        #     )
        fig.update_layout(
            showlegend = False,
            geo = dict(
                scope = 'north america',
                projection_type = 'azimuthal equal area',
                showland = True,
                landcolor = 'rgb(243, 243, 243)',
                countrycolor = 'rgb(204, 204, 204)',
            ),
        )
        fig.update_layout(
                margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
                mapbox={
                    'center': {'lon': 20, 'lat': 50},
                    'style': "open-street-map",
                    'zoom': 5})
        fig.show()