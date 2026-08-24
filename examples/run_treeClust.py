from sentence_transformers import SentenceTransformer

from textClust.core.textClust import textClust
from textClust.data.wrapper import StaticJsonWrapper
from textClust.preprocessing.preprocessor import TFIDFPreprocessor
from textClust.online.tree_clusterer import TreeFader, TreeSumMerger, TreeClusterer, TfidfCosineDistanceTree
from textClust.online.merge import SumMerger
from textClust.time.manager import FixedIntervalTimeManager
from textClust.history.manager import HistoryManager
from textClust.offline.clusterer import SimpleMacroClusterer, TfidfToEmbeddingMacroClusterer

if __name__ == "__main__":
    wrapper = StaticJsonWrapper("examples/data/raw_data/Semantic-T_peaked.json")
    preprocessor = TFIDFPreprocessor()
    
    time_manager = FixedIntervalTimeManager(clean_interval=100, cluster_interval=1000, log_interval=50)
    
    metric = TfidfCosineDistanceTree()
    fader = TreeFader(decay_rate=0.005, term_fading=False, min_weight=2 ** (-1 * 0.005 * 1100))
    merger = TreeSumMerger()
    micro_clusterer = TreeClusterer(max_depth=10, max_children=10, sigma=1.5,
                                    distance_metric=metric, fader=fader, merger=merger)
    history = HistoryManager()
    
    macro_clusterer = TfidfToEmbeddingMacroClusterer(SentenceTransformer("all-MiniLM-L6-v2", device="mps"), epsilon=0.15)
    
    clusterer = textClust(wrapper, preprocessor, time_manager, micro_clusterer, macro_clusterer, history)
    clusterer.start()