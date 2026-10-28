"""Offline macro-clustering interfaces and basic TF-IDF implementation."""
    
from typing import List, Optional


from textClust.history.manager import HistoryManager
from textClust.offline.cluster import Macrocluster
from textClust.online.cluster import Microcluster

import time
from plotly.subplots import make_subplots
import textClust.offline.plots as plots
from textClust.utils.eval import eval_against_truth

class MacroClusterer:
    """Base interface for macro-clustering and evaluation components."""

    def __init__(self, *args, **kwargs):
        """Initialize macroclusterer instance."""
        pass
    
    def cluster(self, micro_clusters: List[Microcluster], history_manager: HistoryManager, current_time: float, *args, **kwargs) -> tuple[List[Macrocluster], Optional[dict[int, int]]]:
        """Create macroclusters from active microclusters."""
        raise NotImplementedError("cluster() must be implemented by subclass")
    
    def evaluate(self, micro_clusters: List[Macrocluster], history_manager: HistoryManager, current_time: float, *args, **kwargs) -> dict:
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
    
    def cluster(self, micro_clusters: List[Microcluster], history_manager: HistoryManager, current_time: float, *args, **kwargs) -> tuple[List[Macrocluster], Optional[dict[int, int]]]:
        """Convert each microcluster into a same-identity macrocluster."""
        now = time.time()
        history_manager.walltime_at_clustering.append(now)
        macro_clusters = []
        for mc in micro_clusters:
            macro_clusters.append(Macrocluster(mc.id, current_time, mc.embedding, mc.n, mc.weight))
        return macro_clusters, None
    
    def evaluate(self, macro_clusters: List[Macrocluster], history_manager: HistoryManager, current_time: float, top_k = 5, *args, **kwargs) -> dict:
        """Compute summary stats and optional ground-truth clustering metrics."""
        macro_clusters.sort(key=lambda x: x.weight, reverse=True)
        top = macro_clusters[:top_k]
        walltime_diff = history_manager.walltime_at_clustering[-1] - history_manager.walltime_at_clustering[-2] if len(history_manager.walltime_at_clustering) > 1 else 1
        
        eval_result = {"time": current_time, "k": top_k, "top_k": top, "walltime_diff": walltime_diff, "num_macro_clusters": len(macro_clusters)}
        if len(history_manager.true_assignments) > 0:
            res = eval_against_truth(
                initial_text_assignments=history_manager.initial_text_assignments,
                final_text_assignments=history_manager.get_final_assignments(),
                true_assignments=history_manager.true_assignments,
                last_eval_time=history_manager.last_eval_time
            )
            res["time"] = current_time
            res["throughput"] = res.get("num_evaluated_points", 0) / walltime_diff
            eval_result.update(res)   
            history_manager.eval_history.append(eval_result)
        history_manager.last_eval_time = current_time
        return eval_result
    
    def display(self, evaluation_result: dict) -> None:
        """Print a textual summary of current top macroclusters and metrics."""
        res = evaluation_result
        print("=" * 70)  
        time_diff = res['walltime_diff']
        print(f"⏰ Time: {time_diff} sec")
        print(f"📊 Number of Micro-Clusters: {res['num_macro_clusters']}")
        print("=" * 70)  
        if "homogeneity" in res:
            print(f"Evaluation Metrics:")
            print(f"  Initial Clustering - Homogeneity: {res['homogeneity']:.4f}, Completeness: {res['completeness']:.4f}, V-measure: {res['v_measure']:.4f}")
            print(f"  Final Clustering   - Homogeneity: {res['homogeneity_final']:.4f}, Completeness: {res['completeness_final']:.4f}, V-measure: {res['v_measure_final']:.4f}")
            print("=" * 70)
            
        if "throughput" in res:
            print(f"Throughput: {res['throughput']:.2f} points/sec")
            print("=" * 70)
        
    def display_final(self, micro_clusters: List[Microcluster], history_manager: HistoryManager, current_time: float, top_k: int = 20, *args, **kwargs) -> None:
        """Plot microcluster weight trajectories over time for top clusters."""
        fig1 = plots.weight_plot(history_manager, top_k=top_k)
        fig2 = plots.sankey_plot(history_manager, use_final_assignments=True)
        fig3 = plots.sankey_plot(history_manager, use_final_assignments=False)
        fig4 = plots.metric_history_plot(history_manager.eval_history)
        fig5 = plots.threshold_over_time_plot(history_manager)
        fig6 = plots.labled_weight_plot(history_manager, top_k=top_k)
        fig7 = plots.long_sankey_plot(history_manager)

        fig = make_subplots(
            rows=7,
            cols=1,
            specs=[[{"type": "xy"}], [{"type": "domain"}], [{"type": "domain"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "domain"}]],
            subplot_titles=(
                "Microcluster Weight Evolution Over Time",
                "Cluster Assignment Flow",
                "Cluster Assignment Flow (Initial)",
                "Evaluation Metrics Over Time",
                "Adaptive Threshold Over Time",
                "Microcluster Weight Evolution with Labels",
                "Cluster Assignment Flow (2)"
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
                
        if fig5 is not None:
            for trace in fig5.data:
                fig.add_trace(trace, row=5, col=1)
                
        if fig6 is not None:
            for trace in fig6.data:
                fig.add_trace(trace, row=6, col=1)
                
        if fig7 is not None:
            for trace in fig7.data:
                fig.add_trace(trace, row=7, col=1)
                
        fig.update_layout(height=7000)
        fig.show()
                
    
