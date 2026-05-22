import numpy as np
from sklearn.metrics import homogeneity_completeness_v_measure, normalized_mutual_info_score
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
        
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_true, y_pred_init)
        nmi = normalized_mutual_info_score(y_true, y_pred_init)
        homogeneity_final, completeness_final, v_measure_final = homogeneity_completeness_v_measure(y_true, y_pred_final)
        nmi_final = normalized_mutual_info_score(y_true, y_pred_final)
        return {
            "homogeneity": homogeneity,
            "completeness": completeness,
            "v_measure": v_measure,
            "nmi": nmi,
            "homogeneity_final": homogeneity_final,
            "completeness_final": completeness_final,
            "v_measure_final": v_measure_final,
            "nmi_final": nmi_final,
            "num_evaluated_points": np.count_nonzero(initial_assignments[:, 0].astype(float) > last_eval_time) if last_eval_time is not None else len(initial_assignments)
        }    