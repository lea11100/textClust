import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from textClust.history.manager import HistoryManager

from typing import List
from collections import defaultdict

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
    
def sankey_plot(history_manager: HistoryManager, use_final_assignments: bool = False) -> go.Figure:
    """Create a Plotly Sankey diagram showing cluster merges over time."""
    true_assignments = [int(c_id) for _, c_id in sorted(history_manager.true_assignments, key=lambda x: x[0])]  # sort by text_id
    if use_final_assignments:
        predicted_assignments = [int(c_id) for _, c_id in sorted(history_manager.get_final_assignments(), key=lambda x: x[0])]
    else:
        predicted_assignments = [int(c_id) for _, _, c_id in sorted(history_manager.initial_text_assignments, key=lambda x: x[1])]

    word_counts_true = _top_k_per_cluster(history_manager.true_assignments, history_manager)
    if use_final_assignments:
        word_counts_pred = _top_k_per_cluster([(t_id, c_id) for t_id, c_id in history_manager.get_final_assignments()], history_manager)
    else:
        word_counts_pred = _top_k_per_cluster([(t_id, c_id) for _, t_id, c_id in history_manager.initial_text_assignments], history_manager)
    # Crosstab: counts of true → predicted
    flows = pd.crosstab(pd.Series(true_assignments, name="true"), pd.Series(predicted_assignments, name="pred"))
    
    # Labels and node indices
    true_nodes = [f"T{c}" for c in flows.index]
    pred_nodes = [f"P{c}" for c in flows.columns]
    labels = true_nodes + pred_nodes
    hover_labels = ["<br>".join(f"{term}:{count}" for term, count in word_counts_true[c_id].items()) for c_id in flows.index] + \
                    ["<br>".join(f"{term}:{count}" for term, count in word_counts_pred[int(c_id[1:])].items()) for c_id in pred_nodes]
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

def labled_weight_plot(history_manager: HistoryManager, top_k: int = 10):
    """Create a Plotly figure showing weight trajectories of top clusters with labels."""
    df_list = []
    for s in history_manager.snapshots_macro:
        clusters = [
            [cid, weight, s.time, label]
            for cid, weight, label in zip(s.macrocluster_ids, s.macrocluster_weights, s.macrocluster_labels)
        ]
        df_list.extend(clusters)

    df = pd.DataFrame(df_list, columns=["cluster_id", "weight", "time", "label"])

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
        color="label",
        title="Macrocluster Weight Evolution Over Time",
        labels={
            "time": "Time",
            "weight": "Weight",
            "label": "Cluster Label"
        },
        color_discrete_sequence=px.colors.qualitative.Alphabet
    )

    fig.update_layout(
        template="plotly_white",
        legend_title="Cluster Label"
    )

    return fig

def weight_plot(history_manager: HistoryManager, top_k: int = 10):
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

def long_sankey_plot(history_manager: HistoryManager):
    true_map = dict(history_manager.true_assignments)
    pred_map = dict([(x[1], x[2]) for x in history_manager.initial_text_assignments])
    merge_map = dict([(x[1], x[2]) for x in history_manager.cluster_merge_history])  # merged_cluster_id -> destination_cluster_id
    tweet_token_map = history_manager.texts
    
    flows = defaultdict(int)
    
    tokens_per_cluster = defaultdict(dict)

    def get_merge_path(cluster_id):
        path = [cluster_id]
        while cluster_id in merge_map:
            cluster_id = merge_map[cluster_id]
            path.append(cluster_id)
        return path

    for tweet in pred_map:
        flows[(f"T:{true_map[tweet]}", f"C:{pred_map[tweet]}")] += 1
        for token in tweet_token_map.get(tweet, []):
            tokens_per_cluster[f"C:{pred_map[tweet]}"][token] = tokens_per_cluster[f"C:{pred_map[tweet]}"].get(token, 0) + 1
            tokens_per_cluster[f"T:{true_map[tweet]}"][token] = tokens_per_cluster[f"T:{true_map[tweet]}"].get(token, 0) + 1
        path = get_merge_path(pred_map[tweet])
        for i in range(len(path) - 1):
            flows[(f"C:{path[i]}", f"C:{path[i+1]}")] += 1
            for token in tweet_token_map.get(tweet, []):
                tokens_per_cluster[f"C:{path[i+1]}"][token] = tokens_per_cluster[f"C:{path[i+1]}"].get(token, 0) + 1

    
    
    nodes = list(set([src for src, _ in flows] + [tgt for _, tgt in flows]))
    node_index = {node: i for i, node in enumerate(nodes)}
    
    top_5_tokens = {node: sorted(tokens.items(), key=lambda x: x[1], reverse=True)[:5] for node, tokens in tokens_per_cluster.items()}
    hover_labels = []
    for node in nodes:
        tokens = top_5_tokens.get(node, [])
        hover_labels.append("<br>".join(f"{term}:{count}" for term, count in tokens))

    source = []
    target = []
    value = []
    for (src, tgt), v in flows.items():
        source.append(node_index[src])
        target.append(node_index[tgt])
        value.append(v)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            hovertemplate="%{label}<br>%{customdata}",
            customdata=hover_labels
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    )])

    fig.update_layout(height=1000)

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