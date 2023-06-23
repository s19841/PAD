from dash import Dash, html, dcc, Output, Input, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# --------------------------------------------------------
from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ----------------------------------------------------------------------------------------------------------------------
# Dane

fields = [
    "id",
    "ccf",
    "age",
    "sex",
    "pinloc",
    "painexer",
    "relrest",
    "pncaden",
    "trestbps",
    "cp",
    "htn",
    "chol",
    "smoke",
    "cigs",
    "years",
    "fbs",
    "dm",
    "famhist",
    "restecg",
    "ekgmo",
    "ekgday",
    "ekgyr",
    "dig",
    "prop",
    "nitr",
    "pro",
    "diuretic",
    "proto",
    "thaldur",
    "thaltime",
    "met",
    "thalach",
    "thalrest",
    "tpeakbps",
    "tpeakbpd",
    "dummy",
    "trestbpd",
    "exang",
    "xhypo",
    "oldpeak",
    "slope",
    "rldv5",
    "rldv5e",
    "ca",
    "restckm",
    "exerckm",
    "restef",
    "restwm",
    "exeref",
    "exerwm",
    "thal",
    "thalsev",
    "thalpul",
    "earlobe",
    "cmo",
    "cday",
    "cyr",
    "num",
    "lmt",
    "ladprox",
    "laddist",
    "diag",
    "cxmain",
    "ramus",
    "om1",
    "om2",
    "rcaprox",
    "rcadist",
    "lvx1",
    "lvx2",
    "lvx3",
    "lvx4",
    "lvf",
    "cathef",
    "junk",
    "name"
]

df_cleveland = (
    pd.read_csv('data_formated/cleveland.csv', sep=' ', header=None, names=fields,
                na_values=["?", -9.0])).drop(columns=['id'])
df_hungarian = pd.read_csv('data_formated/hungarian.csv', sep=' ', header=None,
                           names=fields, na_values=["?", -9.0]).drop(columns=['id'])
df_switzerland = pd.read_csv('data_formated/switzerland.csv', sep=' ', header=None,
                             names=fields, na_values=["?", -9.0]).drop(columns=['id'])
df_long_beach_va = pd.read_csv('data_formated/long-beach-va.csv', sep=' ', header=None,
                               names=fields, na_values=["?", -9.0]).drop(columns=['id'])

frames = [df_cleveland, df_hungarian, df_switzerland, df_long_beach_va]

df = pd.concat(frames)
df = df.rename(columns={"num": "target"})
# ----------------------------------------------------------------------------------------------------------------------
# Wstępne czyszczenie
df = df.drop(
    columns=['dummy', 'restckm', 'exerckm', 'thalsev', 'thalpul', 'earlobe', 'lvx1', 'lvx2', 'lvx3', 'lvx4', 'lvf',
             'cathef', 'junk', 'name'])
missing_values_sorted = []
percent_missing_all = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing_all}).reset_index(drop=True)

with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    missing_values_sorted = missing_value_df.sort_values('percent_missing')

# Czyszczenie danych z nadmiarem pustych wartości
df = df.drop(
    columns=['pncaden', 'exeref', 'exerwm', 'restef', 'restwm', 'dm', 'smoke', 'ca', 'om2', 'ramus', 'diag', 'thaltime',
             'years', 'rldv5', 'famhist', 'cigs', 'slope', 'slope', 'relrest', 'painexer', 'pinloc', 'lmt', 'om1',
             'rcadist', 'laddist', 'rcaprox', 'ladprox', 'cxmain', 'rldv5e', 'proto', 'met', 'fbs'])

# Czyszczenie wierszy z pustymi wartościami
df = df.dropna(subset=df.columns, thresh=30)


def multi_histogram_age():
    m_his = px.histogram(df, x="age", color="sex")
    newnames = {'1': 'Male', '0': 'Female'}
    m_his.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                            legendgroup=newnames[t.name],
                                            hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])
                                            )
                         )
    return m_his


def multi_histogram_advanced():
    m_his = px.histogram(df, x="age", color="target")
    return m_his


def corelation_matrix():
    corr_matrix = df.corr()

    corr_m = px.imshow(corr_matrix)
    corr_m.layout.height = 1000
    corr_m.layout.width = 1000
    return corr_m


def cross_validation():
    df_out = pd.DataFrame(columns=['Model', 'Accuracy'])
    knn_d = KNeighborsClassifier(n_neighbors=5)
    knn_d.fit(X_train, y_train)

    y_pred_d = knn_d.predict(X_train)
    report = classification_report(y_train, y_pred_d)

    # -----------------------------------------------------------------------------------------------
    lr_d = LogisticRegression(random_state=0)
    lr_d.fit(X_train, y_train)

    y_lr_pred_d = lr_d.predict(X_train)
    report_lr = classification_report(y_train, y_lr_pred_d)

    # -----------------------------------------------------------------------------------------------

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)

    y_clf_pred_d = clf.predict(X_train)
    report_clf = classification_report(y_train, y_clf_pred_d)

    # -----------------------------------------------------------------------------------------------

    qlf = QuadraticDiscriminantAnalysis()
    qlf.fit(X_train, y_train)

    y_qlf_pred_d = qlf.predict(X_train)
    report_qlf = classification_report(y_train, y_qlf_pred_d)

    # -----------------------------------------------------------------------------------------------
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    y_gnb_pred_d = gnb.predict(X_train)
    report_gnb = classification_report(y_train, y_gnb_pred_d)

    # -----------------------------------------------------------------------------------------------
    svc = SVC()
    svc.fit(X_train, y_train)

    y_svc_pred = svc.predict(X_train)
    report_svc = classification_report(y_train, y_svc_pred)

    models = {"KNeighborsClassifier": knn_d, "LogisticRegression": lr_d, "LinearDiscriminantAnalysis": clf,
              "QuadraticDiscriminantAnalysis": qlf, "GaussianNB": gnb,
              "Support Vector Classification": svc}

    for model in models.keys():
        scores = cross_val_score(models[model], X_train, y_train, cv=10).mean()
        df_out = pd.concat([df_out, pd.DataFrame({'Model': [model], 'Accuracy': [str(round(scores.mean(), 2))]})],
                           ignore_index=True)

    return df_out


X = df.drop(columns=['target'])

# Y dla clasyfikacji 2 klas 0 = 0, 1 = [1,2,3,4]
df.loc[df["target"] >= 1, "target"] = 1
y = df['target']
scaler = preprocessing.RobustScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,
                                                    random_state=0)
val_df = cross_validation()
# ----------------------------------------------------------------------------------------------------------------------

app = Dash(external_stylesheets=[dbc.themes.QUARTZ])
att = []
global fig

for col in df.columns:
    att.append(col)

global report
# Lista modeli do wyboru
models = {
    'knn': 'KNeighborsClassifier',
    'lr': 'LogisticRegression',
    'clf': 'LinearDiscriminantAnalysis',
    'qlf': 'QuadraticDiscriminantAnalysis',
    'gnb': 'GaussianNB',
    'svc': 'Support Vector Classification'}

app.layout = html.Div(children=[
    html.Div([
        # Tytuł
        html.H1(children='Heart Disease', style={'textAlign': 'center'}),
        # Wyświetlenie dataframe
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in df.columns],
            fixed_rows={'headers': True},

            page_size=20,
            style_table={'height': '300px', 'overflowY': 'auto', 'overflowX': 'auto'},
            style_data={
                'color': 'black',
                'backgroundColor': 'white'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(220, 220, 220)',
                }
            ],
            style_header={
                'backgroundColor': 'rgb(210, 210, 210)',
                'color': 'black',
                'fontWeight': 'bold'
            }

        ),

        # Puste wartości
        html.H4(children='Tablica przedstawiająca % brakujących wartości', style={'textAlign': 'center'}),

        dash_table.DataTable(
            data=missing_values_sorted.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in missing_values_sorted.columns],
            fixed_rows={'headers': True},
            style_table={'height': 500},
            style_data={
                'color': 'black',
                'backgroundColor': 'white'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(220, 220, 220)',
                }
            ],
            style_header={
                'backgroundColor': 'rgb(210, 210, 210)',
                'color': 'black',
                'fontWeight': 'bold'
            }
        ),
        html.H4(children='Histogram przedstawiający ilość rekorów w zależności od wieku z podzieleniem na płeć',
                style={'textAlign': 'center'}),
        dcc.Graph(mathjax=True, figure=multi_histogram_age())
    ], id="static output"),

    html.Div([
        # Box plot
        html.H4(children='Wykres pudełkowy', style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='choose_box_att',
            options=att,
            value=att[1],
            clearable=False,
            style={'color': 'Black'}
        ),
        dcc.Graph(id="box_plot")
    ], id="box_output"),

    html.Div([
        # Histogram
        html.H4(children='Histogram atrybutów', style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='choose_his_att',
            options=att,
            value=att[1],
            clearable=False,
            style={'color': 'Black'}
        ),
        dcc.Graph(id="histogram")
    ], id="first_output"),
    # Macierz korelacji
    html.H4(children='Macierz korelacji atrybutów', style={'textAlign': 'center'}),
    html.Div([dcc.Graph(figure=corelation_matrix())], id="corelation_matrix"),
    # Walidacja krzyżowa
    html.H4(children='Walidacja krzyżowa', style={'textAlign': 'center'}),
    html.Div(dash_table.DataTable(
        data=val_df.to_dict('records'),
        columns=[{'id': c, 'name': c} for c in val_df.columns],
        fixed_rows={'headers': True},
        style_table={'height': 600},
        style_data={
            'color': 'black',
            'backgroundColor': 'white'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(220, 220, 220)',
            }
        ],
        style_header={
            'backgroundColor': 'rgb(210, 210, 210)',
            'color': 'black',
            'fontWeight': 'bold'
        }
    )),
    # Modele
    html.Div([
        # Wybór modelu
        html.H4(children='Wybór modelu', style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='dropdown',
            options=models,
            value='Regression',
            clearable=False,
            style={'color': 'Black'}
        ),
        dcc.Graph(id="graph"),

    ], id="second_output")
])


@app.callback(
    Output("box_plot", "figure", allow_duplicate=True),
    Input('choose_box_att', "value"),
    prevent_initial_call=True
)
# Box plot
def display_box(attr):
    box = px.box(df, y=attr, points="all")
    return box


@app.callback(
    Output("histogram", "figure", allow_duplicate=True),
    Input('choose_his_att', "value"),
    prevent_initial_call=True
)
# Histogram
def display_histogram(attr):
    his = px.histogram(df, x=attr)
    return his


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Input('dropdown', "value"),
    prevent_initial_call=True

)
# Walidacja krzyżowa

# Modelowanie
def train_and_display(name):
    global report
    cm = []
    match name:
        case 'knn':
            knn_d = KNeighborsClassifier(n_neighbors=5)
            knn_d.fit(X_train, y_train)

            y_pred = knn_d.predict(X_test)
            report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

        case 'lr':
            lr = LogisticRegression(random_state=0)
            lr.fit(X_train, y_train)

            y_pred = lr.predict(X_test)
            report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

        case 'clf':
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
        case 'qlf':
            qlf = QuadraticDiscriminantAnalysis()
            qlf.fit(X_train, y_train)

            y_pred = qlf.predict(X_test)
            report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
        case 'gnb':
            gnb = GaussianNB()
            gnb.fit(X_train, y_train)

            y_pred = gnb.predict(X_test)
            report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
        case 'svc':
            svc = SVC()
            svc.fit(X_train, y_train)

            y_pred = svc.predict(X_test)
            report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

    heatmap = go.Heatmap(z=cm, colorscale='Blues')

    # create the layout
    layout = go.Layout(title='Macież omyłek')

    # create the figure
    cm_plot = go.Figure(data=[heatmap], layout=layout)
    print(report)
    return cm_plot


app.run_server(debug=True)
if __name__ == '__main__':
    app.run_server(debug=True)
