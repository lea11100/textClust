from textClust.core.textClust import textClust
from textClust.data.wrapper import StaticJsonWrapper
from textClust.preprocessing.preprocessor import TFIDFPreprocessor
from textClust.online.clusterer import AdaptiveThresholdMicroclusterer
from textClust.online.metrics import TfidfCosineDistance
from textClust.online.tree_clusterer import TreeFader, TreeSumMerger, TreeClusterer, TfidfCosineDistanceTree
from textClust.online.merge import SumMerger
from textClust.time.manager import FixedIntervalTimeManager
from textClust.history.manager import HistoryManager
from textClust.offline.clusterer import SimpleTfidfMacroClusterer

if __name__ == "__main__":
    wrapper = StaticJsonWrapper("examples/data/News-T.json")
    preprocessor = TFIDFPreprocessor()
    
    time_manager = FixedIntervalTimeManager(clean_interval=200, cluster_interval=1000)
    
    metric = TfidfCosineDistanceTree()
    fader = TreeFader(decay_rate=0.002, term_fading=True, min_weight=2 ** (-1 * 0.002 * 100))
    merger = TreeSumMerger()
    micro_clusterer = TreeClusterer(max_depth=10, max_children=10, sigma=2,
                                    distance_metric=metric, fader=fader, merger=merger)
    history = HistoryManager()
    
    macro_clusterer = SimpleTfidfMacroClusterer()
    
    clusterer = textClust(wrapper, preprocessor, time_manager, micro_clusterer, macro_clusterer, history)
    clusterer.start()