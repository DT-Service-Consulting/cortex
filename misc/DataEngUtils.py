from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_group_variator_from_fixators(df: DataFrame, fixators: list, variator: list):
    """
    Check if variator columns have unique values for given fixator combinations
    """
    # Build filter condition for non-null fixators and variator
    filter_condition = F.col(fixators[0]).isNotNull()
    for fixator in fixators[1:]:
        filter_condition = filter_condition & F.col(fixator).isNotNull()
    for var in variator:
        filter_condition = filter_condition & F.col(var).isNotNull()
    
    # Group by fixators and count distinct values in variator
    return (df.filter(filter_condition)
            .groupBy(*fixators))

def get_column_memory_sizes(df: DataFrame):
    type_sizes = {'int': 4, 'bigint': 8, 'float': 4, 'double': 8, 'timestamp': 8, 'date': 4}
    dtypes = dict(df.dtypes)

    aggregations = []
    for i, col_name in enumerate(df.columns):
        aggregations.append(F.count(df[col_name]).alias(f"n{i}"))
        if dtypes[col_name] == 'string':
            aggregations.append(F.avg(F.length(df[col_name])).alias(f"l{i}"))

    stats = df.agg(*aggregations).collect()[0]

    column_sizes = {}
    for i, col_name in enumerate(df.columns):
        if dtypes[col_name] == 'string':
            base_size = stats[f"l{i}"] or 0
        else:
            base_size = type_sizes.get(dtypes[col_name], 4)
        column_sizes[col_name] = (base_size * stats[f"n{i}"]) / (1024 * 1024)

    # Créer DataFrame et calculer Pareto
    size_df = pd.DataFrame(list(column_sizes.items()), columns=["colonne", "taille_mb"]).sort_values("taille_mb", ascending=False)
    size_df['pourcentage_cumule'] = (size_df['taille_mb'].cumsum() / size_df['taille_mb'].sum()) * 100

    # Graphique Pareto
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=size_df["colonne"], y=size_df["taille_mb"], name="Taille (MB)", marker_color='lightblue'), secondary_y=False)
    fig.add_trace(go.Scatter(x=size_df["colonne"], y=size_df["pourcentage_cumule"], mode='lines+markers', name="% Cumulé", line=dict(color='red', width=2)), secondary_y=True)
    fig.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="80%", secondary_y=True)

    fig.update_xaxes(title_text="Colonnes", tickangle=45)
    fig.update_yaxes(title_text="Taille (MB)", secondary_y=False)
    fig.update_yaxes(title_text="Pourcentage Cumulé (%)", secondary_y=True, range=[0, 100])
    fig.update_layout(title="Analyse de Pareto - Poids des colonnes en base de données SQL", height=600, width=1200, showlegend=True)
    fig.show()

    return size_df