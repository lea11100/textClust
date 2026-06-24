import numpy as np
from sklearn import metrics
from typing import Optional

def eval_against_truth(initial_text_assignments, final_text_assignments, true_assignments, last_eval_time: Optional[float] = None) -> dict:
        """Compute homogeneity/completeness/V-measure against truth labels."""
        assert len(true_assignments) > 0, "No ground truth available for evaluation"
        initial_assignments = np.array(initial_text_assignments)
        final_assignments = np.array(final_text_assignments)
        true_assignments = np.array(true_assignments)
        
        initial_assignments = initial_assignments[initial_assignments[:, 1].argsort()]  # sort by point_id
        final_assignments = final_assignments[final_assignments[:, 0].argsort()]  # sort by point_id
        true_assignments = true_assignments[true_assignments[:, 0].argsort()]  # sort by point_id
        
        if last_eval_time is not None:
            after_last_eval_mask = initial_assignments[:, 0].astype(float) > last_eval_time
            initial_assignments = initial_assignments[after_last_eval_mask]
            final_assignments = final_assignments[after_last_eval_mask]
            true_assignments = true_assignments[after_last_eval_mask]
        
        assert np.array_equal(initial_assignments[:, 1], true_assignments[:, 0]), "Mismatch in point IDs between initial and true assignments"
        assert np.array_equal(final_assignments[:, 0], true_assignments[:, 0]), "Mismatch in point IDs between final and true assignments"
        
        y_pred_init = initial_assignments[:, 2]
        y_pred_final = final_assignments[:, 1]
        y_true = true_assignments[:, 1]
        
        # Initial evaluation metrics
        rand = metrics.rand_score(y_true, y_pred_init)
        adj_rand = metrics.adjusted_rand_score(y_true, y_pred_init)
        mi = metrics.mutual_info_score(y_true, y_pred_init)
        adj_mi = metrics.adjusted_mutual_info_score(y_true, y_pred_init)
        norm_mi = metrics.normalized_mutual_info_score(y_true, y_pred_init)
        homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(y_true, y_pred_init)
        fowlkes_mallows = metrics.fowlkes_mallows_score(y_true, y_pred_init)
        
        # Final evaluation metrics
        rand_final = metrics.rand_score(y_true, y_pred_final)
        adj_rand_final = metrics.adjusted_rand_score(y_true, y_pred_final)
        mi_final = metrics.mutual_info_score(y_true, y_pred_final)
        adj_mi_final = metrics.adjusted_mutual_info_score(y_true, y_pred_final)
        norm_mi_final = metrics.normalized_mutual_info_score(y_true, y_pred_final)
        homogeneity_final, completeness_final, v_measure_final = metrics.homogeneity_completeness_v_measure(y_true, y_pred_final)
        fowlkes_mallows_final = metrics.fowlkes_mallows_score(y_true, y_pred_final)
        
        num_evaluated_points = np.count_nonzero(initial_assignments[:, 0].astype(float) > last_eval_time) if last_eval_time is not None else len(initial_assignments)
        num_clusters = len(np.unique(y_pred_init))
        num_clusters_final = len(np.unique(y_pred_final))
        num_clusters_true = len(np.unique(y_true))
        
        return {
            "rand": rand,
            "adj_rand": adj_rand,
            "mutual_info": mi,
            "adj_mutual_info": adj_mi,
            "normalized_mutual_info": norm_mi,
            "homogeneity": homogeneity,
            "completeness": completeness,
            "v_measure": v_measure,
            "fowlkes_mallows": fowlkes_mallows,
            "rand_final": rand_final,
            "adj_rand_final": adj_rand_final,
            "mutual_info_final": mi_final,
            "adj_mutual_info_final": adj_mi_final,
            "normalized_mutual_info_final": norm_mi_final,
            "homogeneity_final": homogeneity_final,
            "completeness_final": completeness_final,
            "v_measure_final": v_measure_final,
            "fowlkes_mallows_final": fowlkes_mallows_final,
            "num_evaluated_points": num_evaluated_points,
            "num_clusters_initial": num_clusters,
            "num_clusters_final": num_clusters_final,
            "num_clusters_true": num_clusters_true
        }