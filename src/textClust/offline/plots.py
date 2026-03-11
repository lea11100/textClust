import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from textClust.history.manager import HistoryManager

from typing import List

def metric_history_plot(eval_history: List[dict]) -> go.Figure:
    """Create a Plotly figure showing evaluation metric trajectories over time."""
    assert len(eval_history) > 0, "Evaluation history is empty."

    df = pd.DataFrame(eval_history)
    df = df.melt(
        id_vars="time",
        value_vars=[
            "homogeneity",
            "completeness",
            "v_measure",
            "homogeneity_final",
            "completeness_final",
            "v_measure_final"
        ],
        var_name="metric",
        value_name="value"
    )

    fig = go.Figure()

    for metric, g in df.groupby("metric"):
        fig.add_trace(
            go.Scatter(
                x=g["time"],
                y=g["value"],
                mode="lines",
                name=metric
            )
        )

    fig.update_layout(
        title="Evaluation Metrics Over Time",
        xaxis_title="Time",
        yaxis_title="Score",
        legend_title="Metric"
    )

    return fig 
    
def sankey_plot(history_manager: HistoryManager) -> go.Figure:
    """Create a Plotly Sankey diagram showing cluster merges over time."""
    true_assignments = [int(c_id) for _, c_id in sorted(history_manager.true_assignments, key=lambda x: x[0])]  # sort by text_id
    predicted_assignments = [int(c_id) for _, _, c_id in sorted(history_manager.initial_text_assignments, key=lambda x: x[1])]
    final_assignments = [int(c_id) for _, c_id in sorted(history_manager.get_final_assignments(), key=lambda x: x[0])]

    word_counts_true = _top_k_per_cluster(history_manager.true_assignments, history_manager)
    word_counts_pred = _top_k_per_cluster([(t_id, c_id) for _, t_id, c_id in history_manager.initial_text_assignments], history_manager)
    word_counts_final = _top_k_per_cluster([(t_id, c_id) for t_id, c_id in history_manager.get_final_assignments()], history_manager)
    # Crosstab: counts of true → predicted
    flows = pd.crosstab(pd.Series(true_assignments, name="true"), pd.Series(final_assignments, name="pred"))
    
    # Labels and node indices
    true_nodes = [f"T{c}" for c in flows.index]
    pred_nodes = [f"P{c}" for c in flows.columns]
    labels = true_nodes + pred_nodes
    hover_labels = ["<br>".join(f"{term}:{count}" for term, count in word_counts_true[c_id].items()) for c_id in flows.index] + \
                    ["<br>".join(f"{term}:{count}" for term, count in word_counts_final[int(c_id[1:])].items()) for c_id in pred_nodes]
    label_idx = {label:i for i,label in enumerate(labels)}

    # Sources, targets, values
    sources, targets, values = [], [], []
    for t in flows.index:
        for p in flows.columns:
            count = int(flows.loc[t,p]) #type: ignore
            if count > 0:
                sources.append(label_idx[f"T{t}"])
                targets.append(label_idx[f"P{p}"])
                values.append(count)

    # Sankey plot
    fig = go.Figure(
        go.Sankey(
            node=dict(
                label=labels,
                hovertemplate="%{label}<br>%{customdata}",
                customdata=hover_labels), 
            link=dict(
                source=sources, 
                target=targets, 
                value=values))) 
    return fig      

def weight_plot(history_manager: HistoryManager, top_k: int = 20):
    """Create a Plotly figure showing weight trajectories of top clusters."""
    df_list = []
    for s in history_manager.snapshots:
        clusters = [
            [cid, weight, s.time]
            for cid, weight in zip(s.microcluster_ids, s.microcluster_weights)
        ]
        df_list.extend(clusters)

    df = pd.DataFrame(df_list, columns=["cluster_id", "weight", "time"])

    # select top clusters by max weight
    top_clusters = (
        df.groupby("cluster_id")["weight"]
        .max()
        .nlargest(top_k)
        .index
    )

    df = df[df["cluster_id"].isin(top_clusters)]

    fig = px.line(
        df,
        x="time",
        y="weight",
        color="cluster_id",
        title="Microcluster Weight Evolution Over Time",
        labels={
            "time": "Time",
            "weight": "Weight",
            "cluster_id": "Cluster ID"
        },
        color_discrete_sequence=px.colors.qualitative.Alphabet
    )

    fig.update_layout(
        template="plotly_white",
        legend_title="Cluster ID"
    )

    return fig

def threshold_over_time_plot(history_manager: HistoryManager):
    """Create a Plotly figure showing threshold changes over time."""
    if len(history_manager.threshold_history) == 0:
        return None

    df = pd.DataFrame(history_manager.threshold_history, columns=["time", "threshold"])

    fig = go.Figure(
        go.Scatter(
            x=df["time"],
            y=df["threshold"],
            mode="lines+markers",
            name="Threshold"
        )
    )

    fig.update_layout(
        title="Adaptive Threshold Over Time",
        xaxis_title="Time",
        yaxis_title="Threshold Value"
    )

    return fig

def _top_k_per_cluster(assignments: List[tuple], history_manager: HistoryManager, top_k: int = 10) -> dict[int, dict[str, int]]:
    """Get top-k terms by weight for each cluster based on assignment history."""
    cluster_word_counts = _count_words_per_cluster(assignments, history_manager)
    top_terms = {}
    for cluster_id, term_counts in cluster_word_counts.items():
        sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
        top_terms[cluster_id] = {term: count for term, count in sorted_terms[:top_k]}
    return top_terms

def _count_words_per_cluster(assignments: List[tuple], history_manager: HistoryManager) -> dict[int, dict[str, int]]:
    """Count total words assigned to each cluster based on text tokens."""
    cluster_word_counts = {}
    for text_id, cluster_id in assignments:
        tokens = history_manager.texts.get(text_id, [])
        cluster_word_counts[cluster_id] = cluster_word_counts.get(cluster_id, {})
        for term in tokens:
            cluster_word_counts[cluster_id][term] = cluster_word_counts[cluster_id].get(term, 0) + 1
    return cluster_word_counts