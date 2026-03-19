"""Offline macro-clustering interfaces and basic TF-IDF implementation."""

import json
import os
from typing import List, Optional

from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import torch

from textClust.history.manager import HistoryManager
from textClust.offline.cluster import Macrocluster, TfIdfMacrocluster, EmbeddingMacrocluster, TfIdfEmbeddingMacrocluster
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
from transformers import AutoTokenizer, AutoModelForCausalLM
import textClust.offline.plots as plots

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
            if isinstance(mc, EmbeddingMicrocluster):
                macro_clusters.append(EmbeddingMacrocluster(mc.id, current_time, mc.n, mc.embedding, mc.weight))
            elif isinstance(mc, TfIdfMicrocluster):
                macro_clusters.append(TfIdfMacrocluster(mc.id, mc.tf, current_time, mc.n, mc.weight))
            else:
                macro_clusters.append(Macrocluster(mc.id, current_time, mc.n, mc.weight))
        return macro_clusters, None
    
    def evaluate(self, macro_clusters: List[Macrocluster], history_manager: HistoryManager, current_time: float, top_k = 5, *args, **kwargs) -> dict:
        """Compute summary stats and optional ground-truth clustering metrics."""
        macro_clusters.sort(key=lambda x: x.weight, reverse=True)
        top = macro_clusters[:top_k]
        walltime_diff = np.diff(history_manager.walltime_at_clustering) if len(history_manager.walltime_at_clustering) > 1 else []
        
        eval_result = {"time": current_time, "k": top_k, "top_k": top, "walltime_diff": walltime_diff, "num_macro_clusters": len(macro_clusters)}
        if len(history_manager.true_assignments) > 0:
            res = self._eval_with_ground_truth(history_manager, history_manager.last_eval_time or 0.0)
            eval_result.update(res)
            res["time"] = current_time
            history_manager.eval_history.append(eval_result)
        history_manager.last_eval_time = current_time
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
        
class TfidfToEmbeddingMacroClusterer(SimpleMacroClusterer):
    def __init__(self, embedding_model: SentenceTransformer, epsilon: float = 0.4, min_samples: int = 4, *args, **kwargs):
        """Initialize macroclusterer instance."""
        self.embedding_model = embedding_model
        self.walltime_at_clustering = []
        self.last_eval_time = None
        self.eval_history = []
        self.epsilon = epsilon
        self.min_samples = min_samples
    
    def cluster(self, micro_clusters: List[TfIdfMicrocluster], history_manager: HistoryManager, current_time: float, *args, **kwargs) -> tuple[List[Macrocluster], Optional[dict[int, int]]]:
        """Create macroclusters from active microclusters."""
        now = time.time()
        self.walltime_at_clustering.append(now)
        macro_clusters = []
        
        final_assignments = np.array(history_manager.get_final_assignments(), dtype=int)
        
        sentences = [" ".join(sorted(mc.tf, key=lambda k: mc.tf[k], reverse=True)[:10]) for mc in micro_clusters]
        embeddings = self.embedding_model.encode(sentences, convert_to_tensor=True)
        #sentences = [np.random.choice(final_assignments[final_assignments[:, 1] == mc.id, 0], size=20) for mc in micro_clusters]
        #sentences = [[" ".join(history_manager.texts.get(str(text_id), "")) for text_id in sentence] for sentence in sentences]
        #sentences_flattened = [s for sublist in sentences for s in sublist]
        
        #embeddings = self.embedding_model.encode(sentences_flattened, convert_to_tensor=True)
        #embeddings = embeddings.reshape(len(micro_clusters), 20, -1).mean(dim=1)  # Average embeddings of top terms for each microcluster
        
        
        # Compute cosine similarity matrix and apply DBSCAN to find macroclusters
        sim = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        
        
        clusterer = DBSCAN(eps=self.epsilon, min_samples=self.min_samples, metric="cosine")
        labels = clusterer.fit_predict(embeddings.cpu().numpy())
        merge_history = {}
        for l in np.unique(labels):
            if l == -1:
                continue
            else:
                cluster_mask = labels == l
                cluster_embeddings = embeddings[cluster_mask]
                cluster_embedding = cluster_embeddings.mean(dim=0)
                cluster_mcs = [mc for mc, mask in zip(micro_clusters, cluster_mask) if mask]
                n = sum(mc.n for mc in cluster_mcs)
                weight = sum(mc.weight for mc in cluster_mcs)
                tf = {}
                id = cluster_mcs[0].id
                for mc in cluster_mcs:
                    merge_history[mc.id] = id
                    for term, freq in mc.tf.items():
                        tf[term] = tf.get(term, 0) + freq
                macro_clusters.append(TfIdfEmbeddingMacrocluster(id, tf, current_time, n, cluster_embedding, weight))
                
        for i, (mc, l) in enumerate(zip(micro_clusters, labels)):
            if l == -1:
                macro_clusters.append(TfIdfEmbeddingMacrocluster(mc.id, mc.tf, current_time, mc.n, embeddings[i], mc.weight))
                merge_history[mc.id] = mc.id
        return macro_clusters, merge_history
    
    
class LlmMacroClusterer(SimpleMacroClusterer):
    def __init__(self, model_name:str, use_hf=True, *args, **kwargs):
        """Initialize macroclusterer instance."""
        self.walltime_at_clustering = []
        self.last_eval_time = None
        self.eval_history = []
        self.model_name = model_name
        self.use_hf = use_hf
        if use_hf:
            self.client = InferenceClient(api_key=os.environ["HF_TOKEN"])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.previous_labels = []
            
    def cluster(self, micro_clusters: List[TfIdfMicrocluster], history_manager: HistoryManager, current_time: float, *args, **kwargs) -> tuple[List[Macrocluster], Optional[dict[int, int]]]:
        """Create macroclusters from active microclusters."""
        now = time.time()
        self.walltime_at_clustering.append(now)
        macro_clusters = []
        
        sentences = [f"{' '.join(sorted(mc.tf, key=lambda k: mc.tf[k], reverse=True)[:10])} ({mc.id})" for mc in micro_clusters]
        sentences = ".\n".join(sentences) + "."
        
        instruction = """You find words that construct sentences that might be syntactically incorrect. Each sentence has an associated ID in parentheses.
        Cluster these sentences by their semantic meaning.
        Return a JSON object with sentence IDs as keys and cluster labels as values. Use meaningful cluster labels based on the common themes of the sentences in each cluster. 
        Be concise but descriptive in your cluster labels. Do not change the order of the IDs in the output.
        Some sentenced may be impure and contain multiple themes, so assign them to a cluster named 'Mixed Themes'. 
        Use only one word as labels but keep the 'Mixed Themes' label as it is. If you are unsure about a sentence, assign it to 'Mixed Themes'.
        Return only the JSON object, without any additional text or explanation. 
        Further, gain inspiration for cluster labels from previous cluster labels, but do not hesitate to introduce new labels if the themes have shifted significantly.
        """
        
        message = [
            {
                "role": "user",
                "content": f"{instruction}\n\nSentences:\n{sentences}\n\nPrevious cluster labels: {', '.join(set(self.previous_labels)) if self.previous_labels else ''}"
            }
        ]
        if self.use_hf:
            completion = self.client.chat.completions.create(model=self.model_name, messages=message, max_tokens=1000)
            response = completion.choices[0].message.content
        else:
            inputs = self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            outputs = self.model.generate(**inputs, max_new_tokens=512)
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        assert response is not None, "LLM did not return a response"
        
        cluster_result = json.loads(response)
        cluster_result = sorted(cluster_result.items(), key=lambda x: int(x[0]))  # Sort by ID
        cluster_result_ids = [int(item[0]) for item in cluster_result]
        cluster_result_labels = np.array([item[1] for item in cluster_result])
        self.previous_labels = list(set(cluster_result_labels))
        
        mc_sorted = sorted(micro_clusters, key=lambda mc: mc.id)
        mc_ids = [mc.id for mc in mc_sorted]
        
        assert cluster_result_ids == mc_ids, "Mismatch between microcluster IDs and LLM output IDs"
        
        merge_history = {}
        for l in np.unique(cluster_result_labels):
            if l == "Mixed Themes":
                continue
            else:
                cluster_mask = cluster_result_labels == l
                cluster_mcs = [mc for mc, mask in zip(mc_sorted, cluster_mask) if mask]
                n = sum(mc.n for mc in cluster_mcs)
                weight = sum(mc.weight for mc in cluster_mcs)
                tf = {}
                id = cluster_mcs[0].id
                for mc in cluster_mcs:
                    merge_history[mc.id] = id
                    for term, freq in mc.tf.items():
                        tf[term] = tf.get(term, 0) + freq
                macro_clusters.append(TfIdfMacrocluster(id, tf, current_time, n, weight, label=l))
                
        for i, (mc, l) in enumerate(zip(mc_sorted, cluster_result_labels)):
            if l == "Mixed Themes":
                #macro_clusters.append(TfIdfEmbeddingMacrocluster(mc.id, mc.tf, current_time, mc.n, embeddings[i], mc.weight))
                merge_history[mc.id] = -1  # Indicate to delete
        return macro_clusters, merge_history
    
    def display(self, evaluation_result: dict) -> None:
        super().display(evaluation_result)
        
        # Display cluster labels for top macroclusters
        for mc in evaluation_result["top_k"]:
            if hasattr(mc, "label") and mc.label is not None:
                print(f"Macrocluster {mc.id} - Label: {mc.label} (Weight: {mc.weight:.2f})")
                
    
