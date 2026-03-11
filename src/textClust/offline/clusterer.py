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
import textClust.offline.plots as plots

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
        final_assignments = np.array(history_manager.get_final_assignments())
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
        fig1 = plots.weight_plot(history_manager, top_k=top_k)
        fig2 = plots.sankey_plot(history_manager)
        fig3 = plots.metric_history_plot(self.eval_history)
        fig4 = plots.threshold_over_time_plot(history_manager)

        fig = make_subplots(
            rows=4,
            cols=1,
            specs=[[{"type": "xy"}], [{"type": "domain"}], [{"type": "xy"}], [{"type": "xy"}]],
            subplot_titles=(
                "Microcluster Weight Evolution Over Time",
                "Cluster Assignment Flow",
                "Evaluation Metrics Over Time",
                "Adaptive Threshold Over Time"
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
                
        if fig4 is not None:
            for trace in fig4.data:
                fig.add_trace(trace, row=4, col=1)
                
        fig.update_layout(height=4000)
        fig.show()
        
    