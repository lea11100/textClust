# from textClust.online.cluster import Microcluster

# import math
# import numpy as np
# from typing import List, Optional
# import deepcopy
# import warnings
# from sklearn.decomposition import PCA

# class MicroclusterNorm: 
#     def __init__(self, tf, idf):
        
#         self.tf_squared_sum = 0
#         self.tf_squared_log_c_sum = 0
#         self.tf_log_c_sum = 0
#         self.tf_sqared_log_c_sqared_sum = 0
        
#         for k, v in tf.items():
#             c = idf.get(k, 1)
#             log_c = math.log(c) if c > 0 else 0        
#             self.tf_squared_sum += v ** 2
#             self.tf_squared_log_c_sum += (v ** 2) * log_c
#             self.tf_log_c_sum += v * log_c
#             self.tf_sqared_log_c_sqared_sum += (v ** 2) * (log_c ** 2)

#     def merge(self, other: 'MicroclusterNorm', tf1, tf2, idf):
#         intersection = set(tf1.keys()) & set(tf2.keys())
#         for term in intersection:
#             self.remove_entry(tf1[term], idf.get(term, 1))
#             other.remove_entry(tf2[term], idf.get(term, 1))
#             self.add_entry(tf1[term] + tf2[term], idf.get(term, 1))
            
#         self.tf_squared_sum += other.tf_squared_sum
#         self.tf_squared_log_c_sum += other.tf_squared_log_c_sum
#         self.tf_log_c_sum += other.tf_log_c_sum
#         self.tf_sqared_log_c_sqared_sum += other.tf_sqared_log_c_sqared_sum
        
#     def fade(self, factor):
#         self.tf_squared_sum *= factor ** 2
#         self.tf_squared_log_c_sum *= factor ** 2
#         self.tf_log_c_sum *= factor
#         self.tf_sqared_log_c_sqared_sum *= factor ** 2

#     def get_norm(self, n):
#         log_squared_n = math.log(n) ** 2
#         log_n_squared = math.log(n ** 2)
        
#         norm = log_squared_n * self.tf_squared_sum
#         norm += log_n_squared * self.tf_squared_sum - log_n_squared * self.tf_squared_log_c_sum
#         norm += self.tf_squared_sum - 2 * self.tf_log_c_sum + self.tf_sqared_log_c_sqared_sum
        
#         return math.sqrt(np.abs(norm)) 
    
#     def remove_entry(self, tf, c, log_c = None):
#         log_c = log_c if log_c is not None else math.log(c) if c > 0 else 0
        
#         self.tf_squared_sum -= tf ** 2
#         if log_c is not None:
#             self.tf_squared_log_c_sum -= (tf ** 2) * log_c
#             self.tf_log_c_sum -= tf * log_c
#             self.tf_sqared_log_c_sqared_sum -= (tf ** 2) * (log_c ** 2)
            
#     def add_entry(self, tf, c, log_c = None):
#         log_c = log_c if log_c is not None else math.log(c) if c > 0 else 0
        
#         self.tf_squared_sum += tf ** 2
#         if log_c is not None:
#             self.tf_squared_log_c_sum += (tf ** 2) * log_c
#             self.tf_log_c_sum += tf * log_c
#             self.tf_sqared_log_c_sqared_sum += (tf ** 2) * (log_c ** 2)
            
#     def update_tf(self, old_tf, new_tf, c, log_c = None):
#         log_c = log_c if log_c is not None else math.log(c) if c > 0 else 0
        
#         self.remove_entry(old_tf, c, log_c)
#         self.add_entry(new_tf, c, log_c)
        
#     def update_log(self, tf, old_c, new_c, old_log_c = None, new_log_c = None):
#         old_log_c = old_log_c if old_log_c is not None else math.log(old_c) if old_c > 0 else 0
#         new_log_c = new_log_c if new_log_c is not None else math.log(new_c) if new_c > 0 else 0
        
#         self.remove_entry(tf, old_c, old_log_c)
#         self.add_entry(tf, new_c, new_log_c)
    
#     def update_all(self, old_tf, new_tf, old_c, new_c, old_log_c = None, new_log_c = None):
#         old_log_c = old_log_c if old_log_c is not None else math.log(old_c) if old_c > 0 else 0
#         new_log_c = new_log_c if new_log_c is not None else math.log(new_c) if new_c > 0 else 0
        
#         self.remove_entry(old_tf, old_c, old_log_c)
#         self.add_entry(new_tf, new_c, new_log_c)

# class TreeMicrocluster(Microcluster):
#     def __init__(self, id, tf:dict[str, float], current_time: float, n: int, parent: Optional['TreeMicrocluster']=None, embedding=None, weight = 1.0):
#         super().__init__(id, tf, current_time, n, embedding, weight)
#         self.children: list['TreeMicrocluster'] = []
#         self.is_centroid = False
#         self.parent = parent
        
#         self.idf = {}
#         self.norm = MicroclusterNorm(tf, self.parent.idf) if self.parent is not None else None  # Only compute norm for leaf nodes
        
#     def set_parent(self, parent: 'TreeMicrocluster'):
#         self.parent = parent
#         if parent is not None:
#             self.norm = MicroclusterNorm(self.tf, self.parent.idf)
#         else:
#             self.norm = None
        
#     def set_tf(self,term, tf):
#         old_tf = self.tf.get(term, 0)
#         self.tf[term] = tf
        
#         if self.norm is not None and self.parent is not None:
#             self.norm.update_tf(old_tf, tf, self.parent.idf.get(term, 1) if self.parent is not None else 1)
            
#         if tf == 0:
#             del self.tf[term]
            
#     def set_idf(self, term, idf):
#         old_idf = self.idf.get(term, 1)
#         self.idf[term] = idf
        
#         if idf == 0:
#             del self.idf[term]
            
#         for c in self.children:
#             if c.norm is not None:
#                 c.norm.update_log(c.tf.get(term, 0), old_idf, idf)
        
#     def add_tf(self, other: 'TreeMicrocluster'):
#         for term, tf in other.tf.items():
#             self.set_tf(term, self.tf.get(term, 0) + tf)
            
#     def add_new_idf(self, other: 'TreeMicrocluster'):
#         diff = set(other.tf.keys()) - set(self.idf.keys())
#         for term in diff:
#             self.set_idf(term, self.idf.get(term, 0) + 1)
            
#     def add_to_existing_idf(self, other: 'TreeMicrocluster'):
#         intersection = set(other.tf.keys()) & set(self.idf.keys())
#         for term in intersection:
#             self.set_idf(term, self.idf.get(term, 0) + 1)
    
#     def get_idf_direct(self):
#         result = {}
#         for c in self.children:
#             for term in c.tf.keys():
#                 result[term] = result.get(term, 0) + 1
#         return result
    
#     def get_tf_direct(self):
#         result = {}
#         for c in self.children:
#             for term, tf in c.tf.items():
#                 result[term] = result.get(term, 0) + tf
#         return result
    
#     def get_n_direct(self):
#         return sum(c.n for c in self.children)
    
#     def get_weight_direct(self):
#         return sum(c.weight for c in self.children)
    
#     def get_tfidf_matrix_of_children(self):
#         all_terms = set([term for c in self.children for term in c.tf.keys()])
#         term_index = {term: i for i, term in enumerate(all_terms)}
#         n = len(self.children)
#         matrix = np.zeros((len(self.children), len(all_terms)))
#         for i, c in enumerate(self.children):
#             for term, info in c.tf.items():
#                 matrix[i, term_index[term]] = info * (1 + math.log(n/self.idf.get(term, 1)))
                
#         return matrix, all_terms    
            
#     def split_children(self, id1, id2):
#         data_matrix, _ = self.get_tfidf_matrix_of_children()
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             pca = PCA(n_components=1).fit_transform(data_matrix)
#         threshold = np.median(pca)  # Use mean of projections as threshold
#         above_the_threshold = pca > threshold
#         left_children = [child for child, t in zip(self.children, above_the_threshold) if not t]
#         right_children = [child for child, t in zip(self.children, above_the_threshold) if t]
#         self.is_centroid = True # Should be ususally the case except for the first split
        
#         left_mc = TreeMicrocluster(id1, {}, Clock().get_time(), None, parent=self, embedding=None, weight=0)
#         tf, idf = combine_children(left_children)
#         left_mc.tf = tf
#         left_mc.idf = idf
#         left_mc.children = left_children
#         left_mc.weight = left_mc.get_weight_direct()
#         left_mc.n = left_mc.get_n_direct()
#         for c in left_children:
#             c.set_parent(left_mc)
            
#         right_mc = TreeMicrocluster(id2, {}, Clock().get_time(), None, parent=self, embedding=None, weight=0)
#         tf, idf, info = TreeMicrocluster.combine_children(right_children)
#         right_mc.tf = tf
#         right_mc.idf = idf
#         right_mc.info = info
#         right_mc.children = right_children
#         right_mc.weight = right_mc.get_weight_direct()
#         right_mc.n = right_mc.get_n_direct()
#         for c in right_children:
#             c.set_parent(right_mc)
            
#         self.children = [left_mc, right_mc]
#         self.idf = self.get_idf_direct()
#         self.tf = self.get_tf_direct()
#         self.weight = self.get_weight_direct()
#         for c in self.children:
#             c.set_parent(self)
    
#     def visualize_tree(self, output_path: str = "tree_visualization", format: str = "png", top_k_terms: int = 5):
#         """
#         Visualize the tree structure using graphviz.
        
#         Args:
#             output_path (str): Path where the visualization will be saved (without extension)
#             format (str): Output format (png, pdf, svg, etc.)
#             top_k_terms (int): Number of top terms to display for each node
            
#         Returns:
#             graphviz.Digraph: The graph object (also saved to file)
#         """
#         try:
#             from graphviz import Digraph
#         except ImportError:
#             raise ImportError("graphviz package is required for tree visualization. Install it with: pip install graphviz")
        
#         # Create a new directed graph
#         dot = Digraph(comment='TreeMicrocluster Visualization')
#         dot.attr(rankdir='TB')  # Top to bottom layout
#         dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
        
#         def add_node_to_graph(node: 'TreeMicrocluster', parent_id: str = None):
#             """Recursively add nodes to the graph"""
#             node_id = f"node_{node.id}"
            
#             # Get top k terms by frequency
#             if node.tf:
#                 sorted_terms = sorted(node.tf.items(), key=lambda x: x[1], reverse=True)[:top_k_terms]
#                 top_terms = ",\\n".join([f"{term}({tf:.2f})" for term, tf in sorted_terms])
#             else:
#                 top_terms = "No terms"
            
#             # Create node label with weight, n, and top terms
#             label = f"ID: {node.id}\\n"
#             label += f"Weight: {node.weight:.2f}\\n"
#             label += f"N: {node.n}\\n"
#             label += f"Top Terms:\\n{top_terms}"
            
#             # Different color for centroid nodes vs leaf nodes
#             fillcolor = 'lightcoral' if node.is_centroid else 'lightgreen'
#             if not node.children:  # Leaf node
#                 fillcolor = 'lightyellow'
            
#             dot.node(node_id, label, fillcolor=fillcolor)
            
#             # Add edge from parent if exists
#             if parent_id is not None:
#                 dot.edge(parent_id, node_id)
            
#             # Recursively add children
#             for child in node.children:
#                 add_node_to_graph(child, node_id)
        
#         # Start visualization from root (self)
#         add_node_to_graph(self)
        
#         # Render the graph
#         dot.render(output_path, format=format, cleanup=True)
        
#         return dot
    
# def mc_mean(microclusters: list['TreeMicrocluster']):
#     result = deepcopy(microclusters[0])
#     for mc in microclusters[1:]:
#         result.merge(mc)
#     return result

# def combine_children(children: list['TreeMicrocluster']):
#     terms = set([term for c in children for term in c.tf.keys()])
#     result_tf = {}
#     result_idf = {}
#     for term in terms:
#         for c in children:
#             result_tf[term] = result_tf.get(term, 0) + c.tf.get(term, 0)
#             result_idf[term] = result_idf.get(term, 0) + (1 if term in c.tf else 0)
#     return result_tf, result_idf
        
    