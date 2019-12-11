import datetime
import pathlib
from datetime import datetime as dt

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import pandas as pd
from dash.dependencies import ClientsideFunction, Input, Output

from utils.helpers import create_plot

app = dash.Dash(
    __name__,
    meta_tags=[{
        "name": "viewport",
        "content": "width=device-width, initial-scale=1"
    }],
)

server = app.server
app.config.suppress_callback_exceptions = True

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

# Read data
dataset = pd.read_csv(DATA_PATH.joinpath("framingham.csv"))
tsne = pd.read_csv(DATA_PATH.joinpath("tsne.csv"))

desc = dataset.iloc[:, :-1].groupby(["male"]).mean()
gender = pd.DataFrame(["Female", "Male"])
desc.insert(0, "Gender", gender)

risk = pd.read_csv(DATA_PATH.joinpath("risk.csv"))

FIGURE = create_plot(x=tsne.iloc[:, 0],
                     y=tsne.iloc[:, 1],
                     z=tsne.iloc[:, 2],
                     color=tsne.iloc[:, 3],
                     xlabel="dimension 1",
                     ylabel="dimension 2",
                     zlabel="dimension 3")


def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("Programmation et Visualisation de données"),
            html.H3("Heart Disease Prediction Dashboard"),
            html.Div(
                id="intro",
                children=[
                    dcc.Markdown("""
                    
                    The dataset is publically available on the Kaggle website, and is
                    from an ongoing cardiovascular study on residents of the town of Framingham,
                    Massachusetts. The classification goal is to predict whether the patient
                    has 10-year risk of future coronary heart disease (CHD).The dataset provides
                    the patients’ information. It includes over 4,000 records and 15 attributes.
                    Variables Each attribute is a potential risk factor. There are both
                    demographic, behavioral and medical risk factors.

                    [The original dataset can be found here.](https://www.kaggle.com/dileep070/heart-disease-prediction-using-logistic-regression)

                    Original authors focused on a classical approach: a logistic regression to
                    determine if patients were at risk to develop CHD diseases in a 10-year
                    time frame. My approach is different to meet course"s specifications:
                    using *DataViz* to answer a personal problematic.
                    My approach is using unsupervised machine learning to detect individuals
                    with a high risk of CHD through *t-distributed stochastic neighbor embedding*
                    (t-SNE), *density-based spatial clustering of applications with noise* (DBSCAN)
                    and *self-organizing maps* (SOM).

                    Copyright (C) 2019 POIRET Clément.

                    This program is free software: you can redistribute it and/or modify
                    it under the terms of the GNU Affero General Public License as published
                    by the Free Software Foundation, either version 3 of the License, or
                    (at your option) any later version.

                    This program is distributed in the hope that it will be useful,
                    but WITHOUT ANY WARRANTY; without even the implied warranty of
                    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
                    GNU Affero General Public License for more details.

                    You should have received a copy of the GNU Affero General Public License
                    along with this program.  If not, see <https://www.gnu.org/licenses/>.

                    """)
                ],
            ),
        ],
    )


app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[
                html.Img(src=app.get_asset_url("unirouen.jpg")),
                html.Img(src=app.get_asset_url("nu.jpg"))
            ],
        ),

        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card()] + [
                html.Div(["initial child"],
                         id="output-clientside",
                         style={"display": "none"})
            ],
        ),

        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                # Descriptive Statistics
                html.Div(
                    id="data_description_card",
                    children=[
                        html.B("Descriptive Statistics"),
                        html.Hr(),
                        dash_table.DataTable(
                            id="desc_table",
                            columns=[{
                                "name": i,
                                "id": i
                            } for i in desc.columns],
                            data=desc.to_dict("records"),
                            style_table={"overflowX": "scroll"},
                            style_as_list_view=True,
                            style_cell={"padding": "5px"},
                            style_header={
                                "backgroundColor": "white",
                                "fontWeight": "bold"
                            }),
                    ],
                ),

                # tSNE
                html.Div(
                    id="tsne_interactive_card",
                    children=[
                        html.B("t-distributed Stochastic Neighbor Embedding"),
                        html.Hr(),
                        "Colouring obtained by Self-Organizing Map (SOM). 0 is safe; 1 is at risk.",
                        html.Br(),
                        dcc.Graph(
                            id="clickable-graph",
                            hoverData={"points": [{
                                "pointNumber": 0
                            }]},
                            figure=FIGURE,
                        ),
                    ],
                ),
                # Patient Wait time by Department
                html.Div(
                    id="patients_card",
                    children=[
                        html.B("Searching a Patient"),
                        html.Hr(),
                        "Here, you can search a patient from its UID, and see if he presents a risk.",
                        html.Div(
                            id="patients_table_div",
                            children=[
                                dash_table.DataTable(
                                    id="patients_table",
                                    columns=[{
                                        "name": i,
                                        "id": i
                                    } for i in risk.columns],
                                    data=risk.to_dict("records"),
                                    sort_action="native",
                                    sort_mode="multi",
                                    selected_columns=[],
                                    selected_rows=[],
                                    page_action="native",
                                    page_current=0,
                                    page_size=10,
                                    style_table={
                                        "overflowX": "scroll",
                                        "maxHeight": "300px",
                                        "overflowY": "scroll"
                                    },
                                    style_as_list_view=True,
                                    style_cell={"padding": "5px"},
                                    style_header={
                                        "backgroundColor": "white",
                                        "fontWeight": "bold"
                                    },
                                    style_data_conditional=[
                                        {
                                            "if": {
                                                "column_id": "UID",
                                                "filter_query": "{risk} eq 1"
                                            },
                                            "backgroundColor": "#DE1738",
                                            "color": "white",
                                        },
                                    ]),
                            ]),
                        html.Div(id="patients_table_container")
                    ],
                ),
            ],
        ),
    ],
)

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
