from textClust.core.textClust import textClust
from textClust.data.wrapper import StaticJsonWrapper
from textClust.preprocessing.preprocessor import EmbeddingPreprocessor
from textClust.online.clusterer import EmbeddingAdaptiveThresholdClusterer
from textClust.online.metrics import EmbeddingCosineDistance
from textClust.online.fade import EmbeddingExpFader
from textClust.online.merge import WeightedSumMerger
from textClust.time.manager import FixedIntervalTimeManager
from textClust.history.manager import HistoryManager
from textClust.offline.clusterer import SimpleMacroClusterer

from sentence_transformers import SentenceTransformer
import time

if __name__ == "__main__":
    start_time = time.time()
    wrapper = StaticJsonWrapper("examples/data/News-T.json")
    preprocessor = EmbeddingPreprocessor(SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2', device="mps"))
    
    time_manager = FixedIntervalTimeManager(clean_interval=100, cluster_interval=1000, log_interval=100)
    
    metric = EmbeddingCosineDistance()
    fader = EmbeddingExpFader(decay_rate=0.005, min_weight=2 ** (-1 * 0.005 * 1000))
    merger = WeightedSumMerger()
    micro_clusterer = EmbeddingAdaptiveThresholdClusterer(sigma=1, distance_metric=metric, fader=fader, merger=merger, min_num_mc=5, 
                                                          include_min_for_threshold=False, threshold_momentum=None)
    history = HistoryManager()
    
    macro_clusterer = SimpleMacroClusterer()
    
    clusterer = textClust(wrapper, preprocessor, time_manager, micro_clusterer, macro_clusterer, history)
    clusterer.start()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time} seconds")