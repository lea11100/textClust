from textClust.online.cluster import TfIdfMicrocluster
from textClust.online.clusterer import Microclusterer

import math
import numpy as np
from typing import List, Optional, Sequence, Tuple, final
from copy import deepcopy
import warnings
from sklearn.decomposition import PCA
import graphviz

from textClust.online.fade import TfIdfExpFade, TfIdfDampedFade
from textClust.online.merge import TfIdfMerger, SumMerger
from textClust.online.metrics import DistanceMetric, TfidfCosineDistance

class MicroclusterNorm: 
    def __init__(self, tf, idf):
        
        self.tf_squared_sum = 0
        self.tf_squared_log_c_sum = 0
        self.tf_log_c_sum = 0
        self.tf_squared_log_c_squared_sum = 0
        
        for k, v in tf.items():
            c = idf.get(k, 1)
            log_c = math.log(c) if c > 0 else 0        
            self.tf_squared_sum += v ** 2
            self.tf_squared_log_c_sum += (v ** 2) * log_c
            self.tf_log_c_sum += v * log_c
            self.tf_squared_log_c_squared_sum += (v ** 2) * (log_c ** 2)

    def get_norm(self, n):
        log_squared_n = math.log(n) ** 2
        log_n_squared = math.log(n ** 2)
        
        norm = log_squared_n * self.tf_squared_sum
        norm += log_n_squared * self.tf_squared_sum - log_n_squared * self.tf_squared_log_c_sum
        norm += self.tf_squared_sum - 2 * self.tf_log_c_sum + self.tf_squared_log_c_squared_sum
        
        return math.sqrt(np.abs(norm)) 
    
    def remove_entry(self, tf, c, log_c = None):
        log_c = log_c if log_c is not None else math.log(c) if c > 0 else 0
        
        self.tf_squared_sum -= tf ** 2
        if log_c is not None:
            self.tf_squared_log_c_sum -= (tf ** 2) * log_c
            self.tf_log_c_sum -= tf * log_c
            self.tf_squared_log_c_squared_sum -= (tf ** 2) * (log_c ** 2)
            
    def add_entry(self, tf, c, log_c = None):
        log_c = log_c if log_c is not None else math.log(c) if c > 0 else 0
        
        self.tf_squared_sum += tf ** 2
        if log_c is not None:
            self.tf_squared_log_c_sum += (tf ** 2) * log_c
            self.tf_log_c_sum += tf * log_c
            self.tf_squared_log_c_squared_sum += (tf ** 2) * (log_c ** 2)
            
    def update_tf(self, old_tf, new_tf, c, log_c = None):
        log_c = log_c if log_c is not None else math.log(c) if c > 0 else 0
        
        self.remove_entry(old_tf, c, log_c)
        self.add_entry(new_tf, c, log_c)
        
    def update_log(self, tf, old_c, new_c, old_log_c = None, new_log_c = None):
        old_log_c = old_log_c if old_log_c is not None else math.log(old_c) if old_c > 0 else 0
        new_log_c = new_log_c if new_log_c is not None else math.log(new_c) if new_c > 0 else 0
        
        self.remove_entry(tf, old_c, old_log_c)
        self.add_entry(tf, new_c, new_log_c)
    
    def update_all(self, old_tf, new_tf, old_c, new_c, old_log_c = None, new_log_c = None):
        old_log_c = old_log_c if old_log_c is not None else math.log(old_c) if old_c > 0 else 0
        new_log_c = new_log_c if new_log_c is not None else math.log(new_c) if new_c > 0 else 0
        
        self.remove_entry(old_tf, old_c, old_log_c)
        self.add_entry(new_tf, new_c, new_log_c)

class TreeMicrocluster(TfIdfMicrocluster):
    def __init__(self, id, tf:dict[str, float], current_time: float, n: int, parent: Optional['TreeMicrocluster']=None, weight = 1.0):
        super().__init__(id, tf, current_time, n, weight)
        self.children: list['TreeMicrocluster'] = []
        self.is_centroid = False
        self.parent = parent
        
        self.idf = {}
        self.norm = MicroclusterNorm(tf, self.parent.idf) if self.parent is not None else MicroclusterNorm({},{})  # Only compute norm for leaf nodes
        
    def set_parent(self, parent: 'TreeMicrocluster'):
        self.parent = parent
        if parent is not None:
            self.norm = MicroclusterNorm(self.tf, self.parent.idf)
        else:
            self.norm = MicroclusterNorm({}, {})
        
    def set_tf(self,term, tf):
        old_tf = self.tf.get(term, 0)
        self.tf[term] = tf
        
        if self.norm is not None and self.parent is not None:
            self.norm.update_tf(old_tf, tf, self.parent.idf.get(term, 1) if self.parent is not None else 1)
            
        if tf == 0:
            del self.tf[term]
            
    def set_idf(self, term, idf):
        old_idf = self.idf.get(term, 1)
        self.idf[term] = idf
        
        if idf == 0:
            del self.idf[term]
            
        for c in self.children:
            if c.norm is not None:
                c.norm.update_log(c.tf.get(term, 0), old_idf, idf)
        
    def add_tf(self, other: TfIdfMicrocluster):
        for term, tf in other.tf.items():
            self.set_tf(term, self.tf.get(term, 0) + tf)
            
    def add_new_idf(self, other: TfIdfMicrocluster):
        diff = set(other.tf.keys()) - set(self.idf.keys())
        for term in diff:
            self.set_idf(term, 1)
            
    def add_to_existing_idf(self, other: 'TreeMicrocluster'):
        intersection = set(other.tf.keys()) & set(self.idf.keys())
        for term in intersection:
            self.set_idf(term, self.idf.get(term, 0) + 1)
    
    def get_idf_direct(self):
        result = {}
        for c in self.children:
            for term in c.tf.keys():
                result[term] = result.get(term, 0) + 1
        return result
    
    def get_tf_direct(self):
        result = {}
        for c in self.children:
            for term, tf in c.tf.items():
                result[term] = result.get(term, 0) + tf
        return result
    
    def get_n_direct(self):
        return sum(c.n for c in self.children)
    
    def get_weight_direct(self) -> float:
        return sum(c.weight for c in self.children)
    
    def get_tfidf_matrix_of_children(self):
        all_terms = set([term for c in self.children for term in c.tf.keys()])
        term_index = {term: i for i, term in enumerate(all_terms)}
        n = len(self.children)
        matrix = np.zeros((len(self.children), len(all_terms)))
        for i, c in enumerate(self.children):
            for term, info in c.tf.items():
                matrix[i, term_index[term]] = info * (1 + math.log(n/self.idf.get(term, 1)))
                
        return matrix, all_terms    
            
    def split_children(self, id1, id2, current_time: float):
        data_matrix, _ = self.get_tfidf_matrix_of_children()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pca = PCA(n_components=1).fit_transform(data_matrix)
        threshold = np.median(pca)  # Use mean of projections as threshold
        above_the_threshold = pca > threshold
        left_children = [child for child, t in zip(self.children, above_the_threshold) if not t]
        right_children = [child for child, t in zip(self.children, above_the_threshold) if t]
        self.is_centroid = True # Should be ususally the case except for the first split
        
        left_mc = TreeMicrocluster(id1, {}, current_time, 0, parent=self, weight=0)
        left_mc.is_centroid = True
        tf, idf = combine_children(left_children)
        left_mc.tf = tf
        left_mc.idf = idf
        left_mc.children = left_children
        left_mc.weight = left_mc.get_weight_direct()
        left_mc.n = left_mc.get_n_direct()
        for c in left_children:
            c.set_parent(left_mc)
            
        right_mc = TreeMicrocluster(id2, {}, current_time, 0, parent=self, weight=0)
        right_mc.is_centroid = True
        tf, idf = combine_children(right_children)
        right_mc.tf = tf
        right_mc.idf = idf
        right_mc.children = right_children
        right_mc.weight = right_mc.get_weight_direct()
        right_mc.n = right_mc.get_n_direct()
        for c in right_children:
            c.set_parent(right_mc)
            
        self.children = [left_mc, right_mc]
        self.idf = self.get_idf_direct()
        self.tf = self.get_tf_direct()
        self.weight = self.get_weight_direct()
        for c in self.children:
            c.set_parent(self)
    
    def visualize_tree(self, output_path: str = "tree_visualization", format: str = "png", top_k_terms: int = 5):
        """
        Visualize the tree structure using graphviz.
        
        Args:
            output_path (str): Path where the visualization will be saved (without extension)
            format (str): Output format (png, pdf, svg, etc.)
            top_k_terms (int): Number of top terms to display for each node
            
        Returns:
            graphviz.Digraph: The graph object (also saved to file)
        """
        try:
            from graphviz import Digraph
        except ImportError:
            raise ImportError("graphviz package is required for tree visualization. Install it with: pip install graphviz")
        
        # Create a new directed graph
        dot = Digraph(comment='TreeMicrocluster Visualization')
        dot.attr(rankdir='TB')  # Top to bottom layout
        dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
        
        def add_node_to_graph(node: 'TreeMicrocluster', parent_id: Optional[str] = None):
            """Recursively add nodes to the graph"""
            node_id = f"node_{node.id}"
            
            # Get top k terms by frequency
            if node.tf:
                sorted_terms = sorted(node.tf.items(), key=lambda x: x[1], reverse=True)[:top_k_terms]
                top_terms = ",\\n".join([f"{term}({tf:.2f})" for term, tf in sorted_terms])
            else:
                top_terms = "No terms"
            
            # Create node label with weight, n, and top terms
            label = f"ID: {node.id}\\n"
            label += f"Weight: {node.weight:.2f}\\n"
            label += f"N: {node.n}\\n"
            label += f"Top Terms:\\n{top_terms}"
            
            # Different color for centroid nodes vs leaf nodes
            fillcolor = 'lightcoral' if node.is_centroid else 'lightgreen'
            if not node.children:  # Leaf node
                fillcolor = 'lightyellow'
            
            dot.node(node_id, label, fillcolor=fillcolor)
            
            # Add edge from parent if exists
            if parent_id is not None:
                dot.edge(parent_id, node_id)
            
            # Recursively add children
            for child in node.children:
                add_node_to_graph(child, node_id)
        
        # Start visualization from root (self)
        add_node_to_graph(self)
        
        # Render the graph
        dot.render(output_path, format=format, cleanup=True)
        
        return dot
    
    def accurate_distance(self, other: 'TreeMicrocluster') -> float:
        if not isinstance(other, TreeMicrocluster):
            raise ValueError("accurate_distance requires another TreeMicrocluster instance")
        
        assert self.parent == other.parent, "Both TreeMicroclusters must have the same parent to compute accurate distance"
        
        metric = TfidfCosineDistance()
    
        if self.parent is not None:
            idf = self.get_parent_idf_direct()
            dist = metric(self, other, idf=idf)
            return dist
        else:
            return -1.0
        
    def get_parent_idf_direct(self):
        result = {}
        if self.parent is not None:
            for c in self.parent.children:
                for term in c.tf.keys():
                    result[term] = result.get(term, 0) + 1
        return result

def combine_children(children: Sequence['TfIdfMicrocluster']):
    terms = set([term for c in children for term in c.tf.keys()])
    result_tf = {}
    result_idf = {}
    for term in terms:
        for c in children:
            result_tf[term] = result_tf.get(term, 0) + c.tf.get(term, 0)
            result_idf[term] = result_idf.get(term, 0) + (1 if term in c.tf else 0)
    return result_tf, result_idf

class TfidfCosineDistanceTree(DistanceMetric):
    def __init__(self):
        super().__init__()
        self.requires_idf = False
        
    def __call__(self, mc1: TreeMicrocluster, mc2: TreeMicrocluster, *args, **kwargs) -> float:
        if not isinstance(mc1, TreeMicrocluster) or not isinstance(mc2, TreeMicrocluster):
            raise ValueError("TfidfCosineDistanceTree requires TreeMicrocluster instances")
        assert mc1.parent is not None, "TreeMicroclusters must have a parent to compute IDF"
        
        tf1 = mc1.tf
        tf2 = mc2.tf
        
        idf = mc1.parent.idf
              
        if not tf1 or not tf2:
            return 1.0
        
        # Create uniform embeddings
        intersection = set(tf1.keys()) & set(tf2.keys())
        
        if not intersection:
            return 1.0
        
        inner_product = 0.0
        
        for term in intersection:
            tf_val1 = tf1[term]
            tf_val2 = tf2[term]
            
            if idf and term in idf:
                idf_i = 1 + math.log(len(mc1.parent.children) / idf.get(term, 1))
                inner_product += idf[term] ** 2 * tf_val1 * tf_val2
            else:
                inner_product += tf_val1 * tf_val2
                
        n = len(mc1.parent.children)
        norm1 = mc1.norm.get_norm(n)
        norm2 = mc2.norm.get_norm(n)

        similarity = inner_product / (norm1 * norm2)
        
        distance = min(1, 1 - similarity)
        distance = max(0, distance)
        return distance

class TreeSumMerger(SumMerger):
    """Extension of SumMerger that also merges tree structure for TreeMicroclusters."""

    def __call__(self, mc1: TreeMicrocluster, mc2: TreeMicrocluster, current_time: float, fader: Optional[TfIdfDampedFade] = None) -> TfIdfMicrocluster:
        """Merge two TreeMicroclusters and combine their children."""
        merged = super().__call__(mc1, mc2, current_time, fader)
        merged = TreeMicrocluster(merged.id, merged.tf, merged.last_update_time, merged.n, None, merged.weight)
        merged.parent = mc1.parent  # Keep the same parent
        merged.children = mc1.children + mc2.children  # Combine children from both clusters
        intersection = set(mc1.tf.keys()) & set(mc2.tf.keys())
        for term in intersection:
            mc1.norm.remove_entry(mc1.tf[term], mc1.idf.get(term, 1))
            mc2.norm.remove_entry(mc2.tf[term], mc2.idf.get(term, 1))
            merged.norm.add_entry(mc1.tf[term] + mc2.tf[term], merged.idf.get(term, 1))
            
        merged.norm.tf_squared_sum += mc2.norm.tf_squared_sum
        merged.norm.tf_squared_log_c_sum += mc2.norm.tf_squared_log_c_sum
        merged.norm.tf_log_c_sum += mc2.norm.tf_log_c_sum
        merged.norm.tf_squared_log_c_squared_sum += mc2.norm.tf_squared_log_c_squared_sum
        
        return merged
    
class TreeFader(TfIdfExpFade):
    """Fader that also applies fading to TreeMicrocluster norms."""
    
    def __init__(self, decay_rate: float, term_fading: bool = True, min_weight: float = 0.5):
        super().__init__(decay_rate, term_fading, min_weight)

    def __call__(self, mc: TreeMicrocluster, current_time: float) -> tuple[TreeMicrocluster, bool]:
        factor = 2 ** (- self.decay_rate * (current_time - mc.last_update_time)) # Calc factor before fading as fading will update the last_update_time
        
        faded_mc, is_expired = super().__call__(mc, current_time)
        faded_mc = TreeMicrocluster(faded_mc.id, faded_mc.tf, faded_mc.last_update_time, faded_mc.n, None, faded_mc.weight)
        faded_mc.parent = mc.parent
        faded_mc.children = mc.children
        
        
        faded_mc.norm.tf_squared_sum *= factor ** 2
        faded_mc.norm.tf_squared_log_c_sum *= factor ** 2
        faded_mc.norm.tf_log_c_sum *= factor
        faded_mc.norm.tf_squared_log_c_squared_sum *= factor ** 2

        return faded_mc, is_expired

class TreeClusterer(Microclusterer):
    def __init__(self, max_depth, max_children, 
                 distance_metric: DistanceMetric, fader: TreeFader, merger: TreeSumMerger, sigma=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tree = None
        self.max_depth = max_depth
        self.max_children = max_children
        self.dist_method = distance_metric
        self.fader = fader
        self.merger = merger
        self.sigma = sigma
        
        self.num_merged_obs = 0
        self.outlier_dist_sum = 0.0
        self.outlier_dist_squaresum = 0.0
        self.dist_sum = 0.0
        
    def _update(self, mc: TfIdfMicrocluster, current_time: float, *args, root: Optional[TreeMicrocluster] = None, **kwargs) -> Tuple[TreeMicrocluster, Optional[int]]:
        mc = TreeMicrocluster(mc.id, mc.tf, mc.last_update_time, mc.n, None, mc.weight)
        root = self.tree if root is None else root
        clusterId = None
        if root is None:
            self._make_tree(mc, current_time)
            root = self.tree
        else:
            parent_node = root
            make_new_cluster = False
            min_child_idx = -1
            min_child_dist = float('inf')
            while len(parent_node.children) > 0:
                parent_node.n += mc.n
                parent_node.weight += mc.weight
                mc.set_parent(parent_node)
                child_dists = [self.dist_method(mc, child) for child in parent_node.children]
                parent_node.add_tf(mc)
                min_child_idx = np.argmin(child_dists)
                min_child_dist = child_dists[min_child_idx]
                child_dists.pop(min_child_idx)
                
                std_dev = np.std(child_dists) if len(child_dists) > 0 else 0.0
                mean_dist = np.mean(child_dists) if len(child_dists) > 0 else 0.0
                if min_child_dist < (mean_dist - self.sigma * std_dev):  # If the closest child is within one std dev, insert there
                    parent_node.add_new_idf(mc)
                    parent_node = parent_node.children[min_child_idx]
                else:
                    make_new_cluster = True
                    break
            if make_new_cluster:
                parent_node.add_to_existing_idf(mc)
                parent_node.add_new_idf(mc)
                parent_node.children.append(mc)
                mc.set_parent(parent_node) #Technically should already be set in the loop but this also makes sure the norm is updated
                if len(parent_node.children) > self.max_children:
                    parent_node.split_children(self.pull_new_clusterId(), self.pull_new_clusterId(), current_time)
                clusterId = None
            else:
                if parent_node is not None and parent_node.parent is not None and min_child_idx >= 0:
                    mc.set_parent(parent_node.parent)
                    merged = self.merger(parent_node, mc, current_time=current_time, fader=self.fader)
                    if isinstance(merged, TreeMicrocluster):
                        parent_node.parent.children[min_child_idx] = merged
                clusterId = parent_node.id
                self.num_merged_obs += 1
                self.dist_sum += min_child_dist
                
        return mc, clusterId
            
    def _make_tree(self, mc: TreeMicrocluster, current_time: float) -> None:
        self.tree = TreeMicrocluster(self.pull_new_clusterId(), deepcopy(mc.tf), current_time, mc.n, None, mc.weight)
        self.tree.is_centroid = True
        self.tree.add_new_idf(mc)
        self.tree.children.append(mc)
        mc.set_parent(self.tree)
    
    def get_clusters(self, node: Optional[TreeMicrocluster] = None) -> List[TfIdfMicrocluster]:
        node = self.tree if node is None else node
        if node is None:
            return []
        result = []
        for c in node.children:
            if len(c.children) > 0:
                result.extend(self.get_clusters(c))
            else:
                result.append(c)
        return result
    
    def check_centroid(self):
        def _check_centroid_rec(node: TreeMicrocluster):
            if node is None:
                return True
            if node.is_centroid and len(node.children) == 0:
                raise ValueError(f"Node {node.id} is marked as centroid but has no children")
            if not node.is_centroid and len(node.children) > 0:
                raise ValueError(f"Node {node.id} is not marked as centroid but has children")
            for c in node.children:
                _check_centroid_rec(c)
        if self.tree is not None:
            _check_centroid_rec(self.tree)
    
    def check_tree(self):
        def _check_tree_rec(node: TreeMicrocluster):
            if node is None:
                return True
            if node.parent is not None:
                assert node in node.parent.children, f"Node {node.id} is not in its parent's children"
            for c in node.children:
                assert c.parent == node, f"Child {c.id} does not have correct parent reference"
                _check_tree_rec(c)
        if self.tree is not None:
            _check_tree_rec(self.tree)
    
    def clean_up(self, current_time: float, *args, **kwargs) -> tuple[List, dict]:
        expired_ids = self._clean_up_clusters(current_time=current_time, node=self.tree)
        merged_ids = self._merge_clusters(current_time=current_time, *args, **kwargs)
        return expired_ids, merged_ids
    
    def _merge_children(self, node: TreeMicrocluster, threshold: float, current_time: float) -> dict[int, int]:
        merge_history = {}
        if len(node.children) > 0:
            for child in node.children:
                merge_history.update(self._merge_children(child, threshold, current_time))
        else:
            return merge_history
        
        merged = np.zeros((len(node.children)), dtype=bool)
        for i in range(len(node.children)):
            if merged[i]:
                continue
            for j in range(i+1, len(node.children)):
                if merged[j]:
                    continue
                dist = self.dist_method(node.children[i], node.children[j])
                if dist < threshold:
                    if not node.children[i].is_centroid and not node.children[j].is_centroid:
                        merged_mc = self.merger(node.children[i], node.children[j], current_time=current_time, fader=self.fader)
                        if isinstance(merged_mc, TreeMicrocluster):
                            node.children[i] = merged_mc
                        merge_history[node.children[j].id] = node.children[i].id
                        merged[j] = True
                    elif not node.children[i].is_centroid and node.children[j].is_centroid:
                        mc, id = self._update(node.children[i], current_time=current_time, root=node.children[j])
                        if id is None:
                            id = self.pull_new_clusterId()
                            mc.id = id
                        merged[i] = True
                        merge_history[node.children[i].id] = id
                        break
                    elif node.children[i].is_centroid and not node.children[j].is_centroid:
                        mc, id = self._update(node.children[j], current_time=current_time, root=node.children[i])
                        if id is None:
                            id = self.pull_new_clusterId()
                            mc.id = id
                        merged[j] = True
                        merge_history[node.children[j].id] = id
                    else:
                        for c in node.children[j].children:
                            c.set_parent(node.children[i])
                        node.children[i].children.extend(node.children[j].children)
                        node.children[i].split_children(self.pull_new_clusterId(), self.pull_new_clusterId(), current_time)
                        node.children[i], node.children[j] = node.children[i].children[0], node.children[i].children[1]
                        node.children[i].set_parent(node)
                        node.children[j].set_parent(node)
                        merged[j] = True
        node.children = [child for m, child in zip(merged, node.children) if not m] 
        if len(node.children) == 1 and node.parent is not None:
            node.parent.children = [c for c in node.parent.children if c != node]
            node.parent.children.append(node.children[0])
            node.children[0].set_parent(node.parent)
        elif node.parent is not None and len(node.children) + len(node.parent.children) - 1 < self.max_children:
            # Merge with parent if the total number of children is less than max_children
            node.parent.children = [c for c in node.parent.children if c != node]
            node.parent.children.extend(node.children)
            for c in node.children:
                c.set_parent(node.parent)
        else:
            node.tf = node.get_tf_direct()
            node.idf = node.get_idf_direct()
            node.weight = node.get_weight_direct()
            
        return merge_history
            
    def _merge_clusters(self, current_time: float, *args, **kwargs) -> dict:    
        # Determine threshold
        threshold = self.dist_sum / self.num_merged_obs if self.num_merged_obs > 0 else 0.0
        
        merged_ids = {}
        if self.tree is not None:
            merged_ids = self._merge_children(self.tree, threshold, current_time)
        
        #self._rebuilt_tree(current_time)
        
        self.num_merged_obs = 0
        self.dist_sum = 0.0
        
        return merged_ids
        
    def _rebuilt_tree(self, current_time: float, microclusters: Optional[List[TfIdfMicrocluster]] = None):
        def _rebuild_tree_helper(node: TreeMicrocluster):
            if len(node.children) > self.max_children:
                node.split_children(self.pull_new_clusterId(), self.pull_new_clusterId(), current_time)
                for c in node.children:
                    _rebuild_tree_helper(c)
            else:
                return 
        
        clusters = self.get_clusters() if microclusters is None else microclusters
        self.tree = TreeMicrocluster(self.pull_new_clusterId(), {}, current_time, 0, None, 0)
        self.tree.is_centroid = True
        self.tree.children = clusters  # type: ignore
        tf, idf = combine_children(clusters)
        self.tree.tf = tf
        self.tree.idf = idf 
        self.tree.n = sum(c.n for c in clusters)
        self.tree.weight = sum(c.weight for c in clusters)
        for c in clusters:
            if isinstance(c, TreeMicrocluster):
                c.set_parent(self.tree)         
        _rebuild_tree_helper(node=self.tree)
    
    def _clean_up_clusters(self, current_time: float, node: Optional[TreeMicrocluster] = None):
        node = self.tree if node is None else node
        expired_ids = []
        if node is not None:  
            _, expired_ids = self._clean_up_clusters_rec(current_time, node)
        return expired_ids
    
    def _clean_up_clusters_rec(self, current_time: float, node: TreeMicrocluster) -> tuple[bool, list]:
        expired_ids = []
        if len(node.children) == 0:
            node, is_expired = self.fader(node, current_time)
            if is_expired:
                return True, [node.id]
            else:
                return False, []
        else:
            expired_children = np.zeros(len(node.children), dtype=bool)
            for i, c in enumerate(node.children):
                is_expired, expired_ids = self._clean_up_clusters_rec(current_time, c)
                expired_children[i] = is_expired
                if is_expired:
                    expired_ids.extend(expired_ids)
        
            if np.all(expired_children):
                return True, expired_ids
            
            node.children = [c for expired, c in zip(expired_children, node.children) if not expired]
            node.tf = node.get_tf_direct()
            node.idf = node.get_idf_direct()
            node.weight = node.get_weight_direct()
            
            return False, expired_ids
        
    def update_with_merge_history(self, merge_history: Optional[dict[int, int]], current_time: float, *args, **kwargs):
        if merge_history is None or len(merge_history) == 0:
            return
        microclusters = self.get_clusters()
        simple_merger = SumMerger()
        mc_dict = {mc.id: mc for mc in microclusters}
        for dest_id in set(merge_history.values()):
            for c_id in [mid for mid, dest in merge_history.items() if dest == dest_id]:
                if dest_id == c_id:
                    continue
                mc_dict[dest_id] = simple_merger(mc_dict[dest_id], mc_dict[c_id], current_time=current_time, fader=self.fader)
                del mc_dict[c_id]
        microclusters = list(mc_dict.values())
        
        self._rebuilt_tree(current_time, microclusters)
        
    