from textClust.core.textClust import textClust
from textClust.data.wrapper import StaticJsonWrapper
from textClust.preprocessing.preprocessor import EmbeddingPreprocessor
from textClust.online.clusterer import EmbeddingAdaptiveThresholdClusterer
from textClust.online.metrics import EmbeddingCosineDistance
from textClust.online.fade import EmbeddingExpFader
from textClust.online.merge import WeightedSumMerger
from textClust.time.manager import FixedIntervalTimeManager
from textClust.history.manager import HistoryManager
from textClust.offline.clusterer import SimpleEmbeddingMacroClusterer

from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    wrapper = StaticJsonWrapper("examples/data/Trends-T_modified.json", max_samples=20000)
    preprocessor = EmbeddingPreprocessor(SentenceTransformer("all-MiniLM-L6-v2"))
    
    time_manager = FixedIntervalTimeManager(clean_interval=100, cluster_interval=1000, log_interval=100)
    
    metric = EmbeddingCosineDistance()
    fader = EmbeddingExpFader(decay_rate=0.005, min_weight=2 ** (-1 * 0.005 * 40))
    merger = WeightedSumMerger()
    micro_clusterer = EmbeddingAdaptiveThresholdClusterer(sigma=1.2, distance_metric=metric, fader=fader, merger=merger)
    history = HistoryManager()
    
    macro_clusterer = SimpleEmbeddingMacroClusterer()
    
    clusterer = textClust(wrapper, preprocessor, time_manager, micro_clusterer, macro_clusterer, history)
    clusterer.start()