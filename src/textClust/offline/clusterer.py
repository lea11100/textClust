from typing import List
from textClust.history.manager import HistoryManager
from textClust.online.cluster import Microcluster

import numpy as np
from sklearn.metrics import homogeneity_completeness_v_measure
import datetime

class MacroCluster(Microcluster):
    def __init__(self, id, tf:dict[str, float], current_time: float, n: int, embedding=None, weight = 1.0):
        super().__init__(id, tf, current_time, n, embedding, weight)

class MacroClusterer:
    def __init__(self, *args, **kwargs):
        pass
    
    def cluster(self, micro_clusters: List[Microcluster], current_time: float, *args, **kwargs) -> List[MacroCluster]:
        raise NotImplementedError("cluster() must be implemented by subclass")
    
    def evaluate(self, micro_clusters: List[MacroCluster], history_manager: HistoryManager, current_time: float, *args, **kwargs) -> dict:
        raise NotImplementedError("evaluate() must be implemented by subclass")
    
    def display(self, evaluation_result: dict) -> None:
        raise NotImplementedError("display() must be implemented by subclass")
    
class SimpleTfidfMacroClusterer(MacroClusterer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.walltime_at_clustering = []
    
    def cluster(self, micro_clusters: List[Microcluster], current_time: float, *args, **kwargs) -> List[MacroCluster]:
        now = datetime.datetime.now()
        self.walltime_at_clustering.append(now)
        macro_clusters = []
        for mc in micro_clusters:
            macro_clusters.append(MacroCluster(mc.id, mc.tf, current_time, mc.n, mc.embedding, mc.weight))
        return macro_clusters
    
    def evaluate(self, macro_clusters: List[MacroCluster], history_manager: HistoryManager, current_time: float, top_k = 5, *args, **kwargs) -> dict:
        macro_clusters.sort(key=lambda x: x.weight, reverse=True)
        top = macro_clusters[:top_k]
        
        walltime_diff = np.diff(self.walltime_at_clustering) if len(self.walltime_at_clustering) > 1 else []
        
        eval_result = {"time": current_time, "k": top_k, "top_k": top, "walltime_diff": walltime_diff}
        if len(history_manager.true_assignments) > 0:
            eval_result.update(self._eval_with_ground_truth(history_manager))
        
        return eval_result
    
    def _eval_with_ground_truth(self, history_manager: HistoryManager):
        assert len(history_manager.true_assignments) > 0, "No ground truth available for evaluation"
        initial_assignments = np.array(history_manager.initial_text_assignments)
        final_assignments = np.array(self._make_final_assignent(history_manager))
        true_assignments = np.array(history_manager.true_assignments)
        
        initial_assignments = initial_assignments[initial_assignments[:, 1].argsort()]  # sort by point_id
        final_assignments = final_assignments[final_assignments[:, 0].argsort()]  # sort by point_id
        true_assignments = true_assignments[true_assignments[:, 0].argsort()]  # sort by point_id
        
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
        res = evaluation_result
        print(f"Time {res['time']}:")
        print("=" * 70)
        print(f"🔍 Top {res['k']} Macro-Clusters")
        print("=" * 70)
        for rank, mc in enumerate(res['top_k'], 1):
            print(f"  #{rank} | Cluster {mc.id} | Weight: {mc.weight:.2f} | Docs: {mc.n}")
            print(f"       Top Terms:")
            for term, weight in sorted(mc.tf.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"         {term}: {weight:.2f}")
            print("-" * 70)
            
        if "homogeneity" in res:
            print(f"Evaluation Metrics:")
            print(f"  Initial Clustering - Homogeneity: {res['homogeneity']:.4f}, Completeness: {res['completeness']:.4f}, V-measure: {res['v_measure']:.4f}")
            print(f"  Final Clustering   - Homogeneity: {res['homogeneity_final']:.4f}, Completeness: {res['completeness_final']:.4f}, V-measure: {res['v_measure_final']:.4f}")
        print("\n\n")
            
        