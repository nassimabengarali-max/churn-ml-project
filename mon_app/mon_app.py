import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.pipeline            import Pipeline
from sklearn.linear_model        import LogisticRegression
from sklearn.svm                 import SVC
from sklearn.ensemble            import RandomForestClassifier
from xgboost                     import XGBClassifier
from sklearn.metrics             import (roc_curve, roc_auc_score,
                                          confusion_matrix, accuracy_score,
                                          f1_score)
from imblearn.over_sampling      import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# ─────────────────────────────────────────────
df_raw = pd.read_csv('Churn_Modelling.csv')
df = df_raw.copy()

df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

le = LabelEncoder()
df['Gender']    = le.fit_transform(df['Gender'])
df['Geography'] = le.fit_transform(df['Geography'])

# Feature engineering (3 nouvelles variables du papier)
df['TenureByAge']         = df['Tenure'] / (df['Age'] + 1)
df['BalanceSalaryRatio']  = df['Balance'] / (df['EstimatedSalary'] + 1)
df['CreditScoreGivenAge'] = df['CreditScore'] / (df['Age'] + 1)

X = df.drop('Exited', axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# ─────────────────────────────────────────────
# ENTRAÎNEMENT DES 4 MODÈLES
# ─────────────────────────────────────────────
def make_pipeline(model):
    return Pipeline([('scaler', StandardScaler()), ('model', model)])

models = {
    'Logistic regression': make_pipeline(LogisticRegression(max_iter=1000, random_state=42)),
    'SVM'                : make_pipeline(SVC(probability=True, random_state=42)),
    'Random forest'      : make_pipeline(RandomForestClassifier(n_estimators=100, random_state=42)),
    'XGBoost'            : make_pipeline(XGBClassifier(use_label_encoder=False,
                                                        eval_metric='logloss', random_state=42))
}

trained = {}
metrics_all = []

for name, pipe in models.items():
    pipe.fit(X_train_sm, y_train_sm)
    trained[name] = pipe

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    cm     = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()

    metrics_all.append({
        'Algorithm'  : name,
        'Accuracy'   : round(accuracy_score(y_test, y_pred), 5),
        'Sensitivity': round(TP / (TP + FN), 5),
        'Specificity': round(TN / (TN + FP), 5),
        'AUC'        : round(roc_auc_score(y_test, y_prob), 5),
        'F1 score'   : round(f1_score(y_test, y_pred), 5),
    })

metrics_df = pd.DataFrame(metrics_all)

# ─────────────────────────────────────────────
# COULEURS DU PAPIER
# ─────────────────────────────────────────────
BG_DARK    = '#0d1b2a'
BG_HEADER  = '#112240'
ACCENT     = '#1a6eb5'
CORAL      = '#e07b54'
BLUE_LIGHT = '#5b9bd5'
WHITE      = '#ffffff'
GOLD       = '#f0a500'

# ─────────────────────────────────────────────
# APP DASH
# ─────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Customer Churn Analysis"

# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
app.layout = html.Div(style={'backgroundColor': BG_DARK, 'minHeight': '100vh',
                              'fontFamily': 'Segoe UI, sans-serif'}, children=[

    # Titre principal
    html.Div(style={'backgroundColor': BG_HEADER, 'padding': '16px 32px',
                    'borderBottom': f'1px solid {ACCENT}'}, children=[
        html.H4("Customer Churn Analysis Dashboard",
                style={'color': WHITE, 'margin': 0, 'fontWeight': '500'})
    ]),

    # Onglets
    dcc.Tabs(id='tabs', value='tab-data',
             style={'backgroundColor': BG_HEADER},
             colors={'border': ACCENT, 'primary': ACCENT, 'background': BG_HEADER},
             children=[
        dcc.Tab(label='Data Analysis',  value='tab-data',
                style={'color': '#aaa', 'backgroundColor': BG_HEADER, 'border': 'none'},
                selected_style={'color': WHITE, 'backgroundColor': ACCENT, 'border': 'none'}),
        dcc.Tab(label='Model Analysis', value='tab-model',
                style={'color': '#aaa', 'backgroundColor': BG_HEADER, 'border': 'none'},
                selected_style={'color': WHITE, 'backgroundColor': ACCENT, 'border': 'none'}),
        dcc.Tab(label='Prediction',     value='tab-pred',
                style={'color': '#aaa', 'backgroundColor': BG_HEADER, 'border': 'none'},
                selected_style={'color': WHITE, 'backgroundColor': ACCENT, 'border': 'none'}),
    ]),

    html.Div(id='tab-content', style={'padding': '24px 32px'})
])

# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────
@app.callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_tab(tab):

    # ══════════════════════════════════════════
    # TAB 1 — DATA ANALYSIS (Fig. 10, 11)
    # ══════════════════════════════════════════
    if tab == 'tab-data':
        cat_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'NumOfProducts']
        num_cols = ['CreditScore', 'Age', 'Balance', 'Tenure', 'EstimatedSalary']

        return html.Div([
            html.H5("Categorical attributes",
                    style={'color': WHITE, 'marginBottom': '16px'}),

            html.Div([
                html.Label("Select attribute", style={'color': '#aaa', 'fontSize': '13px'}),
                dcc.Dropdown(id='cat-dropdown', options=[{'label': c, 'value': c} for c in cat_cols],
                             value='Geography', clearable=False,
                             style={'backgroundColor': BG_HEADER, 'color': WHITE,
                                    'border': f'1px solid {ACCENT}', 'width': '220px'})
            ], style={'marginBottom': '20px'}),

            html.Div(id='cat-charts',
                     style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap': '16px'}),

            html.Hr(style={'borderColor': ACCENT, 'margin': '32px 0'}),

            html.H5("Numerical attributes",
                    style={'color': WHITE, 'marginBottom': '16px'}),

            html.Div([
                html.Label("Select attribute", style={'color': '#aaa', 'fontSize': '13px'}),
                dcc.Dropdown(id='num-dropdown', options=[{'label': c, 'value': c} for c in num_cols],
                             value='CreditScore', clearable=False,
                             style={'backgroundColor': BG_HEADER, 'color': WHITE,
                                    'border': f'1px solid {ACCENT}', 'width': '220px'})
            ], style={'marginBottom': '20px'}),

            html.Div(id='num-charts',
                     style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap': '16px'}),
        ])

    # ══════════════════════════════════════════
    # TAB 2 — MODEL ANALYSIS (Fig. 9, 12)
    # ══════════════════════════════════════════
    elif tab == 'tab-model':
        model_opts = [{'label': m, 'value': m} for m in trained.keys()]
        return html.Div([
            html.Div([
                html.Label("Select model", style={'color': '#aaa', 'fontSize': '13px'}),
                dcc.Dropdown(id='model-dropdown', options=model_opts,
                             value='Random forest', clearable=False,
                             style={'backgroundColor': BG_HEADER, 'color': WHITE,
                                    'border': f'1px solid {ACCENT}', 'width': '260px',
                                    'marginBottom': '24px'})
            ]),
            html.Div(id='model-stats')
        ])

    # ══════════════════════════════════════════
    # TAB 3 — PREDICTION (Fig. 13)
    # ══════════════════════════════════════════
    elif tab == 'tab-pred':
        return html.Div([
            html.H5("Set customer parameters", style={'color': WHITE, 'marginBottom': '20px'}),

            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr',
                            'gap': '16px', 'maxWidth': '700px'}, children=[

                html.Div([html.Label("Gender", style={'color': '#aaa', 'fontSize': '13px'}),
                          dcc.Dropdown(id='p-gender', options=[{'label':'Female','value':0},
                                                                {'label':'Male','value':1}],
                                       value=0, clearable=False,
                                       style={'backgroundColor': BG_HEADER, 'color': WHITE,
                                              'border': f'1px solid {ACCENT}'})]),

                html.Div([html.Label("Geography", style={'color': '#aaa', 'fontSize': '13px'}),
                          dcc.Dropdown(id='p-geo', options=[{'label':'France','value':0},
                                                             {'label':'Germany','value':1},
                                                             {'label':'Spain','value':2}],
                                       value=0, clearable=False,
                                       style={'backgroundColor': BG_HEADER, 'color': WHITE,
                                              'border': f'1px solid {ACCENT}'})]),

                html.Div([html.Label("Age", style={'color': '#aaa', 'fontSize': '13px'}),
                          dcc.Input(id='p-age', type='number', value=25, min=18, max=100,
                                    style={'width': '100%', 'backgroundColor': BG_HEADER,
                                           'color': WHITE, 'border': f'1px solid {ACCENT}',
                                           'padding': '8px', 'borderRadius': '4px'})]),

                html.Div([html.Label("Balance", style={'color': '#aaa', 'fontSize': '13px'}),
                          dcc.Input(id='p-balance', type='number', value=2100, min=0,
                                    style={'width': '100%', 'backgroundColor': BG_HEADER,
                                           'color': WHITE, 'border': f'1px solid {ACCENT}',
                                           'padding': '8px', 'borderRadius': '4px'})]),

                html.Div([html.Label("Credit score", style={'color': '#aaa', 'fontSize': '13px'}),
                          dcc.Input(id='p-credit', type='number', value=300, min=300, max=850,
                                    style={'width': '100%', 'backgroundColor': BG_HEADER,
                                           'color': WHITE, 'border': f'1px solid {ACCENT}',
                                           'padding': '8px', 'borderRadius': '4px'})]),

                html.Div([html.Label("Estimated salary", style={'color': '#aaa', 'fontSize': '13px'}),
                          dcc.Input(id='p-salary', type='number', value=110000,
                                    style={'width': '100%', 'backgroundColor': BG_HEADER,
                                           'color': WHITE, 'border': f'1px solid {ACCENT}',
                                           'padding': '8px', 'borderRadius': '4px'})]),

                html.Div([html.Label("Has credit card", style={'color': '#aaa', 'fontSize': '13px'}),
                          dcc.Dropdown(id='p-crcard', options=[{'label':'No','value':0},
                                                                {'label':'Yes','value':1}],
                                       value=0, clearable=False,
                                       style={'backgroundColor': BG_HEADER, 'color': WHITE,
                                              'border': f'1px solid {ACCENT}'})]),

                html.Div([html.Label("Number of products", style={'color': '#aaa', 'fontSize': '13px'}),
                          dcc.Dropdown(id='p-products', options=[{'label': str(i), 'value': i}
                                                                  for i in range(1, 5)],
                                       value=1, clearable=False,
                                       style={'backgroundColor': BG_HEADER, 'color': WHITE,
                                              'border': f'1px solid {ACCENT}'})]),

                html.Div([html.Label("Is active member", style={'color': '#aaa', 'fontSize': '13px'}),
                          dcc.Dropdown(id='p-active', options=[{'label':'No','value':0},
                                                                {'label':'Yes','value':1}],
                                       value=0, clearable=False,
                                       style={'backgroundColor': BG_HEADER, 'color': WHITE,
                                              'border': f'1px solid {ACCENT}'})]),

                html.Div([html.Label("Tenure", style={'color': '#aaa', 'fontSize': '13px'}),
                          dcc.Input(id='p-tenure', type='number', value=3, min=0, max=10,
                                    style={'width': '100%', 'backgroundColor': BG_HEADER,
                                           'color': WHITE, 'border': f'1px solid {ACCENT}',
                                           'padding': '8px', 'borderRadius': '4px'})]),
            ]),

            html.Button("Predict", id='predict-btn', n_clicks=0,
                        style={'marginTop': '24px', 'backgroundColor': ACCENT,
                               'color': WHITE, 'border': 'none', 'padding': '10px 32px',
                               'borderRadius': '6px', 'fontSize': '15px', 'cursor': 'pointer'}),

            html.Div(id='prediction-results', style={'marginTop': '28px'})
        ])


# ── Callback categorical charts ──────────────
@app.callback(Output('cat-charts', 'children'), Input('cat-dropdown', 'value'))
def update_cat(col):
    df_plot = df_raw.copy()

    # Donut — exited %
    churn_counts = df_plot['Exited'].value_counts()
    fig_donut = go.Figure(go.Pie(
        labels=['Nochurn', 'Churn'],
        values=[churn_counts[0], churn_counts[1]],
        hole=0.55,
        marker_colors=[BLUE_LIGHT, CORAL]
    ))
    fig_donut.update_layout(title='Exited percentage', paper_bgcolor=BG_DARK,
                            plot_bgcolor=BG_DARK, font_color=WHITE,
                            margin=dict(t=40, b=10, l=10, r=10), showlegend=True,
                            legend=dict(font=dict(color=WHITE)))

    # Donut — col %
    col_counts = df_plot[col].value_counts()
    fig_col = go.Figure(go.Pie(
        labels=col_counts.index.astype(str),
        values=col_counts.values,
        hole=0.55
    ))
    fig_col.update_layout(title=f'{col} percentage', paper_bgcolor=BG_DARK,
                          plot_bgcolor=BG_DARK, font_color=WHITE,
                          margin=dict(t=40, b=10, l=10, r=10))

    # Bar — col distribution by churn
    grp = df_plot.groupby([col, 'Exited']).size().reset_index(name='Count')
    fig_bar = px.bar(grp, x=col, y='Count', color='Exited',
                     barmode='group', color_discrete_map={0: BLUE_LIGHT, 1: CORAL},
                     title=f'{col} distribution by churn')
    fig_bar.update_layout(paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK,
                          font_color=WHITE, margin=dict(t=40, b=10, l=10, r=10))
    fig_bar.update_xaxes(gridcolor='#1e3050')
    fig_bar.update_yaxes(gridcolor='#1e3050')

    return [
        dcc.Graph(figure=fig_donut, style={'backgroundColor': BG_DARK}),
        dcc.Graph(figure=fig_col,   style={'backgroundColor': BG_DARK}),
        dcc.Graph(figure=fig_bar,   style={'backgroundColor': BG_DARK}),
    ]


# ── Callback numerical charts ─────────────────
@app.callback(Output('num-charts', 'children'), Input('num-dropdown', 'value'))
def update_num(col):
    df_plot = df_raw.copy()
    churn0 = df_plot[df_plot['Exited'] == 0]
    churn1 = df_plot[df_plot['Exited'] == 1]

    # KDE density
    import scipy.stats as stats
    x_range = np.linspace(df_plot[col].min(), df_plot[col].max(), 300)
    kde0 = stats.gaussian_kde(churn0[col])
    kde1 = stats.gaussian_kde(churn1[col])

    fig_density = go.Figure()
    fig_density.add_trace(go.Scatter(x=x_range, y=kde0(x_range), mode='lines',
                                      name='Churn', line=dict(color=CORAL)))
    fig_density.add_trace(go.Scatter(x=x_range, y=kde1(x_range), mode='lines',
                                      name='Nochurn', line=dict(color=BLUE_LIGHT)))
    fig_density.update_layout(title=f'Density plot of {col}',
                               paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK,
                               font_color=WHITE, margin=dict(t=40, b=10, l=10, r=10))

    # Scatter vs Age
    fig_scatter = px.scatter(df_plot, x='Age', y=col, color='Exited',
                              color_discrete_map={0: BLUE_LIGHT, 1: '#FF69B4'},
                              opacity=0.4, title=f'Scatter plot of {col} v.s. age',
                              labels={'Exited': 'Exited'})
    fig_scatter.update_traces(marker=dict(size=4))
    fig_scatter.update_layout(paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK,
                               font_color=WHITE, margin=dict(t=40, b=10, l=10, r=10))

    # Box plot
    fig_box = px.box(df_plot, x=df_plot['Exited'].map({0:'Churn', 1:'Nochurn'}),
                     y=col, color=df_plot['Exited'].map({0:'Churn', 1:'Nochurn'}),
                     color_discrete_map={'Churn': CORAL, 'Nochurn': BLUE_LIGHT},
                     title=f'Box plot of {col}')
    fig_box.update_layout(paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK,
                           font_color=WHITE, margin=dict(t=40, b=10, l=10, r=10))

    return [
        dcc.Graph(figure=fig_density, style={'backgroundColor': BG_DARK}),
        dcc.Graph(figure=fig_scatter, style={'backgroundColor': BG_DARK}),
        dcc.Graph(figure=fig_box,     style={'backgroundColor': BG_DARK}),
    ]


# ── Callback model stats ──────────────────────
@app.callback(Output('model-stats', 'children'), Input('model-dropdown', 'value'))
def update_model(model_name):
    pipe   = trained[model_name]
    row    = metrics_df[metrics_df['Algorithm'] == model_name].iloc[0]
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                  fill='tozeroy', fillcolor='rgba(230,120,80,0.3)',
                                  line=dict(color=CORAL, width=2), name='ROC'))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                  line=dict(color='gray', dash='dash'), showlegend=False))
    fig_roc.update_layout(title=f'ROC curve', xaxis_title='FPR', yaxis_title='TPR',
                           paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK,
                           font_color=WHITE, margin=dict(t=40, b=40, l=40, r=10))
    fig_roc.update_xaxes(gridcolor='#1e3050')
    fig_roc.update_yaxes(gridcolor='#1e3050')

    # Feature importance
    inner = pipe.named_steps['model']
    if hasattr(inner, 'feature_importances_'):
        fi = pd.Series(inner.feature_importances_, index=X.columns).sort_values()
    elif hasattr(inner, 'coef_'):
        fi = pd.Series(np.abs(inner.coef_[0]), index=X.columns).sort_values()
    else:
        fi = pd.Series(np.ones(len(X.columns)), index=X.columns)

    fig_fi = go.Figure(go.Bar(
        x=fi.values, y=fi.index, orientation='h',
        marker_color=CORAL
    ))
    fig_fi.update_layout(title='Features importance', xaxis_title='Importance',
                          paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK,
                          font_color=WHITE, margin=dict(t=40, b=10, l=10, r=10))
    fig_fi.update_xaxes(gridcolor='#1e3050')

    # Metric cards
    metric_style = {'backgroundColor': '#112240', 'padding': '16px',
                    'borderRadius': '8px', 'textAlign': 'center',
                    'border': f'1px solid {ACCENT}'}

    return html.Div([
        # Ligne de métriques
        html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(6, 1fr)',
                        'gap': '12px', 'marginBottom': '24px'}, children=[
            html.Div([
                html.P(k, style={'color': '#aaa', 'fontSize': '12px', 'margin': '0 0 4px'}),
                html.P(str(row[k]), style={'color': GOLD, 'fontSize': '20px',
                                            'fontWeight': '600', 'margin': 0})
            ], style=metric_style)
            for k in ['Accuracy', 'Sensitivity', 'Specificity', 'AUC', 'F1 score']
        ] + [html.Div([
            html.P('Test-train split', style={'color': '#aaa', 'fontSize': '12px', 'margin': '0 0 4px'}),
            html.P('20%–80%', style={'color': GOLD, 'fontSize': '20px',
                                      'fontWeight': '600', 'margin': 0})
        ], style=metric_style)]),

        # ROC + Feature importance
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1.5fr', 'gap': '16px'}, children=[
            dcc.Graph(figure=fig_roc),
            dcc.Graph(figure=fig_fi),
        ])
    ])


# ── Callback prediction ───────────────────────
@app.callback(
    Output('prediction-results', 'children'),
    Input('predict-btn', 'n_clicks'),
    [dash.dependencies.State('p-gender',   'value'),
     dash.dependencies.State('p-geo',      'value'),
     dash.dependencies.State('p-age',      'value'),
     dash.dependencies.State('p-balance',  'value'),
     dash.dependencies.State('p-credit',   'value'),
     dash.dependencies.State('p-salary',   'value'),
     dash.dependencies.State('p-crcard',   'value'),
     dash.dependencies.State('p-products', 'value'),
     dash.dependencies.State('p-active',   'value'),
     dash.dependencies.State('p-tenure',   'value')],
    prevent_initial_call=True
)
def predict(n, gender, geo, age, balance, credit, salary, crcard, products, active, tenure):
    row = pd.DataFrame([{
        'CreditScore'     : credit,
        'Geography'       : geo,
        'Gender'          : gender,
        'Age'             : age,
        'Tenure'          : tenure,
        'Balance'         : balance,
        'NumOfProducts'   : products,
        'HasCrCard'       : crcard,
        'IsActiveMember'  : active,
        'EstimatedSalary' : salary,
        'TenureByAge'     : tenure / (age + 1),
        'BalanceSalaryRatio' : balance / (salary + 1),
        'CreditScoreGivenAge': credit / (age + 1),
    }])

    results = []
    for name, pipe in trained.items():
        pred  = pipe.predict(row)[0]
        label = 'Churn' if pred == 1 else 'Not churn'
        color = CORAL if pred == 1 else BLUE_LIGHT
        results.append(html.Div([
            html.P(f'Predicted: {label}',
                   style={'color': color, 'fontSize': '16px',
                          'fontWeight': '600', 'margin': '0 0 4px'}),
            html.P(name, style={'color': '#aaa', 'fontSize': '13px', 'margin': 0})
        ], style={'backgroundColor': '#112240', 'padding': '20px', 'borderRadius': '8px',
                  'border': f'2px solid {color}', 'textAlign': 'center'}))

    return html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)',
                            'gap': '16px'}, children=results)


# ─────────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=8050)