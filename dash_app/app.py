import base64
import pickle

import dash
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
import visdcc


import dash_app.app_config as cfg
from probinf import xl_reader
from probinf.inference import elimination_ask


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# ______________________________________________________________________________
# Layout

upload = dcc.Upload(
    id='upload-data',
    children=dbc.Button('Upload xls file', color="success"),
    style={
        'width': '100%',
        'height': '90vh',
        'padding': '40vh 0',
        'textAlign': 'center',
    }
)

variables_selection_column = [
    dbc.Row(
        [
            html.H2('Query variable:'),
            html.Hr(),
        ],
        id='query-variable-row'),
    dbc.Row(
        [
            html.H2('Evidence:'),
            html.Hr(),
            dbc.Container(id='evidence-variable-container', className='scrollbar')
        ]
    )]

app.layout = dbc.Container([
    dcc.Store(id='bayes-net-store'),
    dbc.Row([
        dbc.Col(variables_selection_column, id='variables-selection-column', width={'size': 3}),
        dbc.Col([
            dbc.Row(upload, id='bayes-net-row'),
            dbc.Row(id='output-row', className='scrollbar')
        ])
    ])
],
    fluid=True
)


# ______________________________________________________________________________
# Callbacks

@app.callback(
    Output('bayes-net-row', 'children'),
    Output('bayes-net-store', 'data'),
    Input('upload-data', 'contents'),
    State('bayes-net-row', 'children'))
def create_bayes_net(content, current_state):
    """Triggered when a file is uploaded.
    Creates the bayes net visualization and stores the BayesNet-Object to dcc-store"""
    if content is None:
        return current_state, None

    content = content.encode("utf8").split(b";base64,")[1]
    content = base64.decodebytes(content)

    bayes_net = xl_reader.get_net_from_xls(file_contents=content)

    nodes = []
    for variable in bayes_net.variables:
        nodes.append({'id': variable, 'label': variable,
                      'shape': cfg.NODE_SHAPE, 'size': cfg.NODE_SIZE, 'font': cfg.NODE_FONT})

    edges = []
    for node in bayes_net.nodes:
        for parent in node.parents:
            edges.append({'from': parent, 'to': node.variable, 'width': cfg.EDGE_SIZE})

    data = {'nodes': nodes, 'edges': edges}

    network = visdcc.Network(id='bayes-net',
                             data=data,
                             options=dict(edges={'arrows': {'to': True}},
                                          interaction={'hover': True})
                             )
    button = dbc.Button('Physics: On', id='physics-button', color=cfg.PHYSICS_BUTTON_COLOR)

    container = dbc.Container([network, button, html.Hr()])

    return container, pickle.dumps(bayes_net).decode('latin-1')


@app.callback(
    Output('bayes-net', 'options'),
    Output('physics-button', 'children'),
    Output('physics-button', 'color'),
    Input('physics-button', 'n_clicks'))
def physics_controller(n_clicks):
    """Callback for button to disable or enable physics in bayes net visualization"""
    if n_clicks is None or n_clicks % 2 == 0:
        return {'physics': True}, 'Physics: On', 'success'
    return {'physics': False}, 'Physics: Off', 'danger'


@app.callback(
    Output('query-variable-row', 'children'),
    Input('bayes-net-store', 'data'),
    State('query-variable-row', 'children'))
def create_query_dropdown(bayes_net, children):
    """Creates the dropdown to select the query variable"""
    if bayes_net is None:
        return children

    bayes_net = pickle.loads(bayes_net.encode('latin-1'))

    options = [{'label': var, 'value': var} for var in bayes_net.variables]

    dropdown = dbc.Select(
        id='query-variable-dropdown',
        options=options,
        placeholder='Select query variable...'
    )

    children.append(dropdown)

    return children


@app.callback(
    Output('evidence-variable-container', 'children'),
    Input('bayes-net-store', 'data'))
def create_evidence_dropdowns(bayes_net):
    """Creates for every variable a dropdown to specify evidence"""
    if bayes_net is None:
        return None

    bayes_net = pickle.loads(bayes_net.encode('latin-1'))

    children = []
    for node in bayes_net.nodes:
        options = [{'label': val, 'value': val} for val in node.variable_values]
        options.append({'label': 'No evidence', 'value': 'No evidence'})

        label = dbc.InputGroupText(node.variable, id='evidence-variable-label')
        dropdown = dbc.Select(
            id={'type': 'evidence-variable-dropdown', 'variable': node.variable},
            value='No evidence',
            options=options,
        )

        children.append(dbc.InputGroup([label, dropdown],
                                       id='evidence-variable-input-group'))

    return children


@app.callback(
    Output('bayes-net', 'data'),
    Input('query-variable-dropdown', 'value'),
    Input({'type': 'evidence-variable-dropdown', 'variable': ALL}, 'value'),
    State('bayes-net', 'data'))
def highlight_selected_variables(query_variable, evidence_values, bayes_net_vis):
    """Highlights the nodes in bayes net visualization based on query and evidence"""
    for evidence_value, node in zip(evidence_values, bayes_net_vis['nodes']):

        if node['id'] == query_variable:
            node['color'] = '#00ff00'
        elif evidence_value != 'No evidence':
            node['color'] = '#bd2f2f'
        else:
            node['color'] = '#97c2fc'

    return bayes_net_vis


@app.callback(
    Output({'type': 'evidence-variable-dropdown', 'variable': MATCH}, 'disabled'),
    Output({'type': 'evidence-variable-dropdown', 'variable': MATCH}, 'value'),
    Input('query-variable-dropdown', 'value'),
    State({'type': 'evidence-variable-dropdown', 'variable': MATCH}, 'id'),
    State({'type': 'evidence-variable-dropdown', 'variable': MATCH}, 'value'))
def disable_evidence_dropdown_for_query_variable(query_value, dropdown_id, current_value):
    """Disables the evidence dropdown when variable is selected as query variable"""
    if query_value == dropdown_id['variable']:
        return True, 'No evidence'
    else:
        return False, current_value


@app.callback(
    Output('output-row', 'children'),
    Input('query-variable-dropdown', 'value'),
    Input({'type': 'evidence-variable-dropdown', 'variable': ALL}, 'value'),
    Input('bayes-net', 'selection'),
    State('bayes-net-store', 'data'),
    State('output-row', 'children'))
def update_output_row(query_variable, evidence_values, selection, bayes_net, children):
    """Adds either a cpt or the inference result to screen"""
    ctx = dash.callback_context

    bayes_net = pickle.loads(bayes_net.encode('latin-1'))

    if ctx.triggered[0]['prop_id'] == 'bayes-net.selection':
        output, className = generate_cpt(selection, bayes_net)
    else:
        output, className = inference(query_variable, evidence_values, bayes_net)

    if children is None:
        children = []

    if output is None:
        return children
    else:
        children.insert(0, dbc.Alert(output, dismissable=True, is_open=True, className=className))
        return children


def inference(query_variable, evidence_values, bayes_net):
    if query_variable:
        evidence_variables = dict(zip(bayes_net.variables, evidence_values))

        evidence_variables = {var: val for var, val in evidence_variables.items() if val != 'No evidence'}

        if query_variable in evidence_variables:
            del evidence_variables[query_variable]

        result = elimination_ask(query_variable, evidence_variables, bayes_net)

        label = f'P({result.var_name})'

        df = pd.DataFrame({val: [prob] for val, prob in result.prob.items()})
        table = dbc.Table.from_dataframe(df, striped=True, bordered=True)

        return [label, table], 'result'
    else:
        return None, None


def generate_cpt(selection, bayes_net):
    if len(selection['nodes']) > 0:
        node = bayes_net.get_node(selection['nodes'][0])
        cpt_df = node.as_table()

        if node.parents:
            index = True
            label = f"P({node.variable}|{', '.join(node.parents)})"
        else:
            index = False
            label = f'P({node.variable})'

        label = dbc.Label(label)

        table = dbc.Table.from_dataframe(cpt_df, index=index, striped=True, bordered=True)

        return [label, table], 'cpt'
    else:
        return None, None


if __name__ == '__main__':
    app.run_server(debug=True)
