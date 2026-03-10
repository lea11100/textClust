"""Offline macro-clustering interfaces and basic TF-IDF implementation."""

from typing import List

import torch

from textClust.history.manager import HistoryManager
from textClust.online.cluster import Microcluster, TfIdfMicrocluster, EmbeddingMicrocluster

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import homogeneity_completeness_v_measure
import time
import plotly.graph_objects as go
import plotly.tools as tls
from plotly.subplots import make_subplots
import plotly.express as px

class MacroCluster(Microcluster):
    """Macro-level cluster representation built from microclusters."""

    def __init__(self, id, current_time: float, n: int, weight = 1.0):
        """Initialize a macrocluster."""
        super().__init__(id, current_time, n, weight)
        
class TfIdfMacroCluster(MacroCluster):
    """Macro-level cluster representation with TF-IDF summary."""

    def __init__(self, id, tf:dict[str, float], current_time: float, n: int, weight = 1.0):
        """Initialize a macrocluster."""
        super().__init__(id, current_time, n, weight)
        self.tf = tf
        
class EmbeddingMacroCluster(MacroCluster):
    """Macro-level cluster representation with embedding summary."""

    def __init__(self, id, current_time: float, n: int, embedding:torch.Tensor, weight = 1.0):
        """Initialize a macrocluster."""
        super().__init__(id, current_time, n, weight)
        self.embedding = embedding
        
class MacroClusterer:
    """Base interface for macro-clustering and evaluation components."""

    def __init__(self, *args, **kwargs):
        """Initialize macroclusterer instance."""
        pass
    
    def cluster(self, micro_clusters: List[Microcluster], current_time: float, *args, **kwargs) -> List[MacroCluster]:
        """Create macroclusters from active microclusters."""
        raise NotImplementedError("cluster() must be implemented by subclass")
    
    def evaluate(self, micro_clusters: List[MacroCluster], history_manager: HistoryManager, current_time: float, *args, **kwargs) -> dict:
        """Evaluate macroclustering results and return metric dictionary."""
        raise NotImplementedError("evaluate() must be implemented by subclass")
    
    def display(self, evaluation_result: dict) -> None:
        """Render intermediate evaluation output."""
        raise NotImplementedError("display() must be implemented by subclass")
    
    def display_final(self, micro_clusters: List[Microcluster], history_manager: HistoryManager, current_time: float, *args, **kwargs) -> None:
        """Render final output at the end of processing."""
        raise NotImplementedError("display_final() must be implemented by subclass")
    
class SimpleMacroClusterer(MacroClusterer):
    """Simple baseline macroclusterer that mirrors microclusters one-to-one."""

    def __init__(self, *args, **kwargs):
        """Initialize simple macroclusterer state."""
        super().__init__(*args, **kwargs)
        self.walltime_at_clustering = []
        self.last_eval_time = None
        self.eval_history = []
    
    def cluster(self, micro_clusters: List[Microcluster], current_time: float, *args, **kwargs) -> List[MacroCluster]:
        """Convert each microcluster into a same-identity macrocluster."""
        now = time.time()
        self.walltime_at_clustering.append(now)
        macro_clusters = []
        for mc in micro_clusters:
            if isinstance(mc, EmbeddingMicrocluster):
                macro_clusters.append(EmbeddingMacroCluster(mc.id, current_time, mc.n, mc.embedding, mc.weight))
            elif isinstance(mc, TfIdfMicrocluster):
                macro_clusters.append(TfIdfMacroCluster(mc.id, mc.tf, current_time, mc.n, mc.weight))
            else:
                macro_clusters.append(MacroCluster(mc.id, current_time, mc.n, mc.weight))
        return macro_clusters
    
    def evaluate(self, macro_clusters: List[MacroCluster], history_manager: HistoryManager, current_time: float, top_k = 5, *args, **kwargs) -> dict:
        """Compute summary stats and optional ground-truth clustering metrics."""
        macro_clusters.sort(key=lambda x: x.weight, reverse=True)
        top = macro_clusters[:top_k]
        walltime_diff = np.diff(self.walltime_at_clustering) if len(self.walltime_at_clustering) > 1 else []
        
        eval_result = {"time": current_time, "k": top_k, "top_k": top, "walltime_diff": walltime_diff, "num_macro_clusters": len(macro_clusters)}
        if len(history_manager.true_assignments) > 0:
            res = self._eval_with_ground_truth(history_manager, self.last_eval_time or 0.0)
            eval_result.update(res)
            res["time"] = current_time
            self.eval_history.append(res)
        self.last_eval_time = current_time
        return eval_result
    
    def _eval_with_ground_truth(self, history_manager: HistoryManager, last_eval_time: float) -> dict:
        """Compute homogeneity/completeness/V-measure against truth labels."""
        assert len(history_manager.true_assignments) > 0, "No ground truth available for evaluation"
        initial_assignments = np.array(history_manager.initial_text_assignments)
        final_assignments = np.array(self._make_final_assignent(history_manager))
        true_assignments = np.array(history_manager.true_assignments)
        
        initial_assignments = initial_assignments[initial_assignments[:, 1].argsort()]  # sort by point_id
        final_assignments = final_assignments[final_assignments[:, 0].argsort()]  # sort by point_id
        true_assignments = true_assignments[true_assignments[:, 0].argsort()]  # sort by point_id
        
        after_last_eval_mask = initial_assignments[:, 0].astype(float) > last_eval_time
        initial_assignments = initial_assignments[after_last_eval_mask]
        final_assignments = final_assignments[after_last_eval_mask]
        true_assignments = true_assignments[after_last_eval_mask]
        
        assert np.array_equal(initial_assignments[:, 1], true_assignments[:, 0]), "Mismatch in point IDs between initial and true assignments"
        assert np.array_equal(final_assignments[:, 0], true_assignments[:, 0]), "Mismatch in point IDs between final and true assignments"
        
        y_pred_init = initial_assignments[:, 2].astype(int)
        y_pred_final = final_assignments[:, 1].astype(int)
        y_true = true_assignments[:, 1].astype(int)
        
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_true, y_pred_init)
        homogeneity_final, completeness_final, v_measure_final = homogeneity_completeness_v_measure(y_true, y_pred_final)
        return {
            "homogeneity": homogeneity,
            "completeness": completeness,
            "v_measure": v_measure,
            "homogeneity_final": homogeneity_final,
            "completeness_final": completeness_final,
            "v_measure_final": v_measure_final
        }
    
    def _make_final_assignent(self, history_manager: HistoryManager):
        """Derive final text-to-cluster assignments after merge history replay."""
        assignments = { t_id : c_id for _, t_id, c_id in history_manager.initial_text_assignments }
        rev_assignments = {}
        for t_id, c_id in assignments.items():
            if c_id not in rev_assignments:
                rev_assignments[c_id] = []
            rev_assignments[c_id].append(t_id)
        for _, old_id, new_id in history_manager.cluster_merge_history:
            if old_id in rev_assignments:
                if new_id not in rev_assignments:
                    rev_assignments[new_id] = []
                rev_assignments[new_id].extend(rev_assignments[old_id])
                del rev_assignments[old_id]
                
        final_assignements = []
        for c_id, t_ids in rev_assignments.items():
            for t_id in t_ids:
                final_assignements.append((t_id, c_id))
        return final_assignements
        
    
    def display(self, evaluation_result: dict) -> None:
        """Print a textual summary of current top macroclusters and metrics."""
        res = evaluation_result
        print("=" * 70)  
        time_diff = res['walltime_diff'][-1] if len(res['walltime_diff']) > 0 else 0
        print(f"⏰ Time: {time_diff} sec")
        print(f"📊 Number of Micro-Clusters: {res['num_macro_clusters']}")
        print("=" * 70)  
        if "homogeneity" in res:
            print(f"Evaluation Metrics:")
            print(f"  Initial Clustering - Homogeneity: {res['homogeneity']:.4f}, Completeness: {res['completeness']:.4f}, V-measure: {res['v_measure']:.4f}")
            print(f"  Final Clustering   - Homogeneity: {res['homogeneity_final']:.4f}, Completeness: {res['completeness_final']:.4f}, V-measure: {res['v_measure_final']:.4f}")
            print("=" * 70)
        
    def display_final(self, micro_clusters: List[Microcluster], history_manager: HistoryManager, current_time: float, top_k: int = 20, *args, **kwargs) -> None:
        """Plot microcluster weight trajectories over time for top clusters."""
        fig1 = self._make_weight_plot(history_manager, top_k=top_k)
        fig2 = self._make_sankey_plot(history_manager)
        fig3 = self._make_eval_history_plot()

        fig = make_subplots(
            rows=3,
            cols=1,
            specs=[[{"type": "xy"}], [{"type": "domain"}], [{"type": "xy"}]],
            subplot_titles=(
                "Microcluster Weight Evolution Over Time",
                "Cluster Assignment Flow",
                "Evaluation Metrics Over Time"
            ),
            vertical_spacing=0.12,
        )

        if fig1 is not None:
            for trace in fig1.data:
                fig.add_trace(trace, row=1, col=1)

        if fig2 is not None:
            for trace in fig2.data:
                fig.add_trace(trace, row=2, col=1)
                
        if fig3 is not None:
            for trace in fig3.data:
                fig.add_trace(trace, row=3, col=1)
                
        fig.update_layout(height=3000)
        fig.show()
        
    def _make_eval_history_plot(self):
        """Create a Plotly figure showing evaluation metric trajectories over time."""
        if len(self.eval_history) == 0:
            return None

        df = pd.DataFrame(self.eval_history)
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
        
    def _make_sankey_plot(self, history_manager: HistoryManager) -> go.Figure:
        """Create a Plotly Sankey diagram showing cluster merges over time."""
        true_assignments = [int(c_id) for _, c_id in sorted(history_manager.true_assignments, key=lambda x: x[0])]  # sort by text_id
        predicted_assignments = [int(c_id) for _, _, c_id in sorted(history_manager.initial_text_assignments, key=lambda x: x[1])]
        final_assignments = [int(c_id) for _, c_id in sorted(self._make_final_assignent(history_manager), key=lambda x: x[0])]

        word_counts_true = self._top_k_per_cluster(history_manager.true_assignments, history_manager)
        word_counts_pred = self._top_k_per_cluster([(t_id, c_id) for _, t_id, c_id in history_manager.initial_text_assignments], history_manager)
        word_counts_final = self._top_k_per_cluster([(t_id, c_id) for t_id, c_id in self._make_final_assignent(history_manager)], history_manager)
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

    def _make_weight_plot(self, history_manager: HistoryManager, top_k: int = 20):
        """Create a Plotly figure showing weight trajectories of top clusters."""
        df_list = []
        for s in history_manager.snapshots:
            clusters = [[id, weight, s.time] for id, weight in zip(s.microcluster_ids, s.microcluster_weights)]
            df_list.extend(clusters)
        
        df = pd.DataFrame(df_list, columns=["cluster_id", "weight", "time"])
        top_clusters = df.groupby('cluster_id')['weight'].max().nlargest(top_k).index
        df = df[df["cluster_id"].isin(top_clusters)]
        ax = sns.lineplot(data=df, x="time", y="weight", hue="cluster_id", palette="tab20")
        
        ax.set_title("Microcluster Weight Evolution Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Weight")
        
        fig_plotly = tls.mpl_to_plotly(ax.get_figure())
        return fig_plotly
    
    def _top_k_per_cluster(self, assignments: List[tuple], history_manager: HistoryManager, top_k: int = 10) -> dict[int, dict[str, int]]:
        """Get top-k terms by weight for each cluster based on assignment history."""
        cluster_word_counts = self._count_words_per_cluster(assignments, history_manager)
        top_terms = {}
        for cluster_id, term_counts in cluster_word_counts.items():
            sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
            top_terms[cluster_id] = {term: count for term, count in sorted_terms[:top_k]}
        return top_terms
    
    def _count_words_per_cluster(self, assignments: List[tuple], history_manager: HistoryManager) -> dict[int, dict[str, int]]:
        """Count total words assigned to each cluster based on text tokens."""
        cluster_word_counts = {}
        for text_id, cluster_id in assignments:
            tokens = history_manager.texts.get(text_id, [])
            cluster_word_counts[cluster_id] = cluster_word_counts.get(cluster_id, {})
            for term in tokens:
                cluster_word_counts[cluster_id][term] = cluster_word_counts[cluster_id].get(term, 0) + 1
        return cluster_word_counts