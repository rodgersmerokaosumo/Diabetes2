# Import necessary libraries
import json
import base64
import io
import os

import pandas as pd
from dash import Dash, dcc, html, callback, Input, Output, State, dash_table
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from reportlab.lib.pagesizes import letter, landscape 
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


# Define external stylesheets for Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Initialize Dash app
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Define layout for the app
import dash_core_components as dcc
import dash_html_components as html

app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Upload(
                    id='upload-data',
                    children=html.Button('Select Files', style={
                        'background-color': 'green', 'color': 'white', 'fontWeight': 'bold',
                        'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px',
                        'cursor': 'pointer', 'transition': 'background-color 0.3s'
                    }),
                    multiple=False
                ),
                html.Div(id='output-data-upload'),
                html.Button('Show DataFrame Info', id='show-info-btn', n_clicks=0,
                            style={'width': '100%', 'margin-top': '10px'}),
                html.Div(id='dataframe-info'),
                html.Button('Show Missing Values Info', id='show-missing-values-btn', n_clicks=0,
                            style={'width': '100%', 'margin-top': '10px'}),
                html.Div(id='missing-values-info'),
                html.Button('Convert to Categorical', id='convert-to-categorical-button', n_clicks=0,
                            style={'width': '100%', 'margin-top': '10px'}),
                dcc.Dropdown(id='column-to-convert', multi=True, style={'margin-top': '10px'}),
                html.Div(id='conversion-result'),
                html.Button('Show Summary Statistics', id='show-summary-btn', n_clicks=0,
                            style={'width': '100%', 'margin-top': '10px'}),
                html.Div(id='summary-statistics'),
                html.Button('Show Visualizations', id='show-visualizations-btn', n_clicks=0,
                            style={'width': '100%', 'margin-top': '10px'}),
                html.Div(id='visualization'),
                html.Div(id='heatmap'),
                html.Button('Resample', id='resample-button', n_clicks=0,
                            style={'width': '100%', 'margin-top': '10px'}),
                html.Div(id='resampled-data'),
                html.Div(id='resampled-data-table'),
                dcc.Dropdown(id='y-variable', style={'margin-top': '10px'}),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[
                        {'label': 'Random Forest', 'value': 'RandomForestClassifier'},
                        {'label': 'Support Vector Machine', 'value': 'SVC'},
                        {'label': 'Logistic Regression', 'value': 'LogisticRegression'}
                    ],
                    value=[],
                    multi=True,
                    style={'margin-top': '10px'}
                ),
                html.Button('Evaluate', id='evaluate-button', n_clicks=0,
                            style={'width': '100%', 'margin-top': '10px'}),
                html.Div(id='model-performance-metrics'),
                html.Div(id='stored-df', style={'display': 'none'}),
                html.Button('Generate Report', id='generate-report-btn', n_clicks=0,
                            style={'width': '100%', 'margin-top': '10px'}),
                html.Div(id='report-status'),
                html.Button('Help', id='toggle-help-btn', n_clicks=0,
                            style={'width': '100%', 'margin-top': '10px'}),
            ],
            style={
                'flex': 1,
                'padding': '20px',
                'minWidth': '50%'
            }
        ),
        html.Div(
            id='help-text',
            style={
                'display': 'none',
                'backgroundColor': '#E6F4EA',
                'padding': '20px',
                'overflowY': 'scroll',
                'height': '100vh',
                'minWidth': '50%'
            },
            children=[
                html.H4('How to Use This Application'),
                html.H5('Upload your data files'),
                html.P('Click the "Select Files" button to upload CSV or Excel files from your computer. Once selected, the file will be automatically uploaded, and you can view a preview of the dataset including the first and last few rows.'),
                html.H5('DataFrame Information'),
                html.P('After uploading, click "Show DataFrame Info" to display information about the columns, data types, and the amount of data in your DataFrame. It helps you understand the structure of your data.'),
                html.H5('Missing Values Information'),
                html.P('Use "Show Missing Values Info" to identify columns that contain missing values. This feature provides a quick summary of missing data counts per column, which is crucial for data cleaning and preparation.'),
                html.H5('Convert Columns to Categorical'),
                html.P('If your dataset includes categorical data, use the "Convert to Categorical" button after selecting columns from the dropdown menu. This conversion is necessary for certain types of data analysis and machine learning model training.'),
                html.H5('Summary Statistics'),
                html.P('Click "Show Summary Statistics" to get descriptive statistics for each column in your dataset. This includes count, mean, standard deviation, min, and max values, which are essential for initial data analysis.'),
                html.H5('Data Visualizations'),
                html.P('Use "Show Visualizations" to generate histograms and bar charts for numeric and categorical data, respectively. Visualizations help in understanding data distribution and spotting patterns.'),
                html.H5('Data Resampling'),
                html.P('The "Resample" button allows for balancing data, particularly useful in dealing with imbalanced datasets where some classes dominate others. This feature helps in improving model accuracy.'),
                html.H5('Model Training and Evaluation'),
                html.P('After data preprocessing, use the "Evaluate" button to run machine learning models selected from the dropdown. This will provide performance metrics like accuracy, precision, recall, and F1-score for each model.'),
                html.H5('Generating Reports'),
                html.P('Click the "Generate Report" button to create a PDF report of your data. Ensure your data is properly loaded and processed before generating the report. The report will be saved to your designated path, which you can access to view or distribute.')
            ]
        )
    ],
    style={
        'display': 'flex',
        'flexDirection': 'row'
    }
)

# Function to parse uploaded contents
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return df

# Callback to update output data upload
@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        sliced_df = pd.concat([df.head(4), df.tail(4)])
        return html.Div([
            html.H5(f'File uploaded: {filename}'),
            dash_table.DataTable(data=sliced_df.to_dict('records')),
        ])

# Callback to show DataFrame info
@app.callback(
    [Output('dataframe-info', 'children'),
     Output('missing-values-info', 'children')],
    Input('show-info-btn', 'n_clicks'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def show_info(n_clicks, contents, filename):
    if n_clicks and contents is not None:
        df = parse_contents(contents, filename)
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_string = buffer.getvalue()
        missing_values = df.isnull().sum()
        missing_values_df = pd.DataFrame(missing_values, columns=['Missing Values'])
        return (html.Div([
                    html.H5('DataFrame Info:'),
                    html.Pre(info_string)
                ]),
                html.Div([
                    html.H5('Missing Values Info:'),
                    dash_table.DataTable(data=missing_values_df.to_dict('records'), columns=[{"name": i, "id": i} for i in missing_values_df.columns])
                ]))
    return html.Div(), html.Div()

# Callback to update dropdown options for column conversion
@app.callback(
    Output('column-to-convert', 'options'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_dropdown(contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        return [{'label': i, 'value': i} for i in df.columns]
    return []

# Callback to convert selected columns to categorical
@app.callback(
    Output('conversion-result', 'children'),
    Input('convert-to-categorical-button', 'n_clicks'),
    State('column-to-convert', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def convert_to_categorical(n_clicks, selected_columns, contents, filename):
    if n_clicks and contents is not None:
        df = parse_contents(contents, filename)
        if not isinstance(selected_columns, list):
            selected_columns = [selected_columns]
        for col in selected_columns:
            df[col] = df[col].astype('category')
        return html.Div([
            html.H5('Conversion Result:'),
            html.P(f'Successfully converted {", ".join(selected_columns)} to categorical.')
        ])

# Callback to show summary statistics
@app.callback(
    Output('summary-statistics', 'children'),
    Input('show-summary-btn', 'n_clicks'),
    State('column-to-convert', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def show_summary_statistics(n_clicks, selected_columns, contents, filename):
    if n_clicks and contents is not None:
        df = parse_contents(contents, filename)
        if not isinstance(selected_columns, list):
            selected_columns = [selected_columns]
        for col in selected_columns:
            df[col] = df[col].astype('category')
        
        summary = df.describe(include='all').transpose()
        summary.insert(0, 'Column Name', summary.index)
        summary = summary.reset_index(drop=True)
        
        return html.Div([
            html.H5('Summary Statistics:'),
            dash_table.DataTable(data=summary.to_dict('records'), columns=[{"name": i, "id": i} for i in summary.columns])
        ])
        
# Callback to show visualizations
@app.callback(
    Output('visualization', 'children'),
    Input('show-visualizations-btn', 'n_clicks'),
    State('column-to-convert', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def show_visualizations(n_clicks, selected_columns, contents, filename):
    if n_clicks and contents is not None:
        df = parse_contents(contents, filename)
        if not isinstance(selected_columns, list):
            selected_columns = [selected_columns]
        for col in selected_columns:
            df[col] = df[col].astype('category')
        
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['category']).columns.tolist()
        
        figs = []
        if numerical_cols and categorical_cols:
            for col in numerical_cols:
                fig = px.histogram(df, x=col)
                figs.append(html.Div(dcc.Graph(figure=fig)))
            
            for col in categorical_cols:
                value_counts_df = df[col].value_counts().reset_index()
                value_counts_df.columns = [col, 'count']
                fig = px.bar(value_counts_df, x=col, y='count')
                figs.append(html.Div(dcc.Graph(figure=fig)))
                
            for num_col in numerical_cols:
                for cat_col in categorical_cols:
                    fig = px.histogram(df, x=num_col, color=cat_col, marginal="violin", hover_data=df.columns)
                    figs.append(html.Div(dcc.Graph(figure=fig)))
        else:
            figs.append(html.Div([
                html.H5('Visualizations:'),
                html.P('No numerical and categorical column pairs found for visualization.')
            ]))
        
        return html.Div([
            html.H5('Visualizations:'),
            *figs
        ])

# Callback to show heatmap
@app.callback(
    Output('heatmap', 'children'),
    Input('show-visualizations-btn', 'n_clicks'),
    State('column-to-convert', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def show_heatmap(n_clicks, selected_columns, contents, filename):
    if n_clicks and contents is not None:
        df = parse_contents(contents, filename)
        if not isinstance(selected_columns, list):
            selected_columns = [selected_columns]
        for col in selected_columns:
            df[col] = df[col].astype('category')

        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numerical_cols:
            corr = df[numerical_cols].corr().round(2)
            heatmap = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), colorscale='Viridis')
            return html.Div([
                html.H5('Heatmap:'),
                dcc.Graph(figure=heatmap)
            ])
        else:
            return html.Div([
                html.H5('Heatmap:'),
                html.P('No numerical columns found for heatmap visualization.')
            ])
    return html.Div([
        html.H5('Heatmap:'),
        html.P('Upload a file and select columns to generate heatmap.')
    ])
    
    
# Callback to update y variable dropdown for resampling
@app.callback(
    Output('y-variable', 'options'),
    Input('convert-to-categorical-button', 'n_clicks'),
    State('column-to-convert', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_y_variable_dropdown(n_clicks, selected_columns, contents, filename):
    if n_clicks and contents is not None:
        df = parse_contents(contents, filename)
        if selected_columns:
            options = [{'label': col, 'value': col} for col in df.columns]
            return options
    return []

@app.callback(
    Output('stored-df', 'children'),  # Store the DataFrame in the hidden Div
    [Input('show-visualizations-btn', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('upload-data', 'filename')]
)
def store_dataframe(n_clicks, contents, filename):
    if n_clicks and contents is not None:
        df = parse_contents(contents, filename)
        return df.to_json(date_format='iso', orient='split')  # Store the DataFrame in JSON format
    return None

# Callback to resample data
@app.callback(
    [Output('resampled-data', 'children'),
     Output('resampled-data-table', 'children')],
    [Input('resample-button', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('upload-data', 'filename')]
)
def resample_data_and_display_table(n_clicks, contents, filename):
    if n_clicks and contents is not None:
        df = parse_contents(contents, filename)
        
        # Assuming 'Outcome' is the y variable column for resampling
        y_variable = 'Outcome'  # Change this according to your actual column name
        
        if y_variable in df.columns:
            df_majority = df[df[y_variable] == df[y_variable].value_counts().idxmax()]
            df_minority = df[df[y_variable] == df[y_variable].value_counts().idxmin()]

            df_minority_upsampled = resample(df_minority,
                                             replace=True,
                                             n_samples=df_majority.shape[0],
                                             random_state=123)

            df_resampled = pd.concat([df_majority, df_minority_upsampled])

            value_counts = df_resampled[y_variable].value_counts().reset_index()
            value_counts.columns = [y_variable, 'count']

            # Convert resampled data to DataTable
            resampled_table = dash_table.DataTable(
                id='resampled-data-table',
                columns=[{"name": i, "id": i} for i in value_counts.columns],
                data=value_counts.to_dict('records')
            )

            return value_counts.to_json(date_format='iso', orient='split'), resampled_table
    return None, None

@app.callback(
    Output('model-performance-metrics', 'children'),
    [Input('evaluate-button', 'n_clicks')],
    [State('model-dropdown', 'value'),
     State('stored-df', 'children')]  # Retrieve the stored DataFrame
)
def evaluate_models_and_display_metrics(n_clicks, selected_models, stored_df_json):
    if n_clicks and selected_models and stored_df_json:
        try:
            # Convert the stored DataFrame from JSON to DataFrame
            stored_df = pd.read_json(stored_df_json, orient='split')

            # Assuming 'Outcome' is your target variable
            X = stored_df.drop(columns=['Outcome'])
            y = stored_df['Outcome']

            # Split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # Preprocessing for numerical features
            numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
            numerical_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            # Preprocessing for categorical features
            categorical_features = X.select_dtypes(include=['object']).columns
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            # Define models
            models = {
                'RandomForestClassifier': RandomForestClassifier(),
                'SVM': SVC(),
                'LogisticRegression': LogisticRegression()
                # Add more models as needed
            }

            # Create a pipeline for each selected model
            pipelines = {}
            for name, model in models.items():
                if name in selected_models:
                    pipelines[name] = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', model)
                    ])

            # Define an empty dictionary to store evaluation metrics
            results = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': []}

            # Evaluate each model and store the results
            for name, pipeline in pipelines.items():
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='macro')
                recall = recall_score(y_test, y_pred, average='macro')
                f1 = f1_score(y_test, y_pred, average='macro')
                results['Model'].append(name)
                results['Accuracy'].append(accuracy)
                results['Precision'].append(precision)
                results['Recall'].append(recall)
                results['F1-score'].append(f1)

            # Create a DataFrame from the results dictionary
            results_df = pd.DataFrame(results)

            # Convert DataFrame to DataTable
            model_metrics_table = dash_table.DataTable(
                id='model-performance-metrics-table',
                columns=[{"name": i, "id": i} for i in results_df.columns],
                data=results_df.to_dict('records')
            )

            return model_metrics_table
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return html.Div("An error occurred while evaluating models.")
    return html.Div("Please select models and resampled data, then click Evaluate.")


def generate_text_report(df, filename, info_string, missing_values_df, summary_statistics):
    # Create the text file buffer
    with open(f"{filename}_report.txt", "w") as text_file:
        
        # Adding the title and general information
        text_file.write(f"Report for: {filename}\n")
        text_file.write("="*50 + "\n")
        
        # Adding DataFrame info
        text_file.write("DataFrame Information:\n")
        text_file.write(info_string + "\n")
        text_file.write("="*50 + "\n")
        
        # Missing Values Section
        text_file.write("Missing Values Summary:\n")
        for index, row in missing_values_df.iterrows():
            line = f"{index}: {row['Missing Values']}\n"
            text_file.write(line)
        text_file.write("="*50 + "\n")
        
        # Summary Statistics
        text_file.write("Summary Statistics:\n")
        for index, row in summary_statistics.iterrows():
            line = f"{index}:\n{row.to_string()}\n"
            text_file.write(line)
        text_file.write("="*50 + "\n")
        
    return f"{filename}_report.txt"

@app.callback(
    Output('report-status', 'children'),
    Input('generate-report-btn', 'n_clicks'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def generate_report(n_clicks, contents, filename):
    if n_clicks and contents:
        df = parse_contents(contents, filename)

        # DataFrame Info and Missing Values
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_string = buffer.getvalue()
        missing_values = df.isnull().sum()
        missing_values_df = pd.DataFrame(missing_values, columns=['Missing Values'])

        # Summary Statistics
        summary_statistics = df.describe(include='all').transpose()

        # Generate Text Report
        report_file = generate_text_report(df, filename, info_string, missing_values_df, summary_statistics)
        return html.Div(f'Report generated successfully as "{report_file}". Please check your server directory.')
    
    return html.Div("Please upload a file and click 'Generate Report' to view the output.")



@app.callback(
    Output('help-text', 'style'),
    [Input('toggle-help-btn', 'n_clicks')],
    [State('help-text', 'style')]
)
def toggle_help(n_clicks, style):
    if n_clicks % 2 == 1:
        return {'display': 'block'}  # Show the help text
    else:
        return {'display': 'none'}  # Hide the help text

if __name__ == "__main__":
    app.run_server(debug=True, port=int(os.environ.get("PORT", 5000)))


def load_and_process_data(filepath):
    # Example processing function that might do more than just read the file
    df = pd.read_csv(filepath)
    # Assume some processing here
    return df