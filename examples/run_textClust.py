from textClust.core.textClust import textClust
from textClust.data.wrapper import StaticJsonWrapper
from textClust.preprocessing.preprocessor import TFIDFPreprocessor
from textClust.online.clusterer import TfIdfAdaptiveThresholdMicroclusterer
from textClust.online.metrics import TfidfCosineDistance
from textClust.online.fade import TfIdfExpFade
from textClust.online.merge import SumMerger
from textClust.time.manager import FixedIntervalTimeManager
from textClust.history.manager import HistoryManager
from textClust.offline.clusterer import SimpleMacroClusterer

if __name__ == "__main__":
    wrapper = StaticJsonWrapper("examples/data/Trends-T_modified.json", max_samples=20000)
    preprocessor = TFIDFPreprocessor()
    
    time_manager = FixedIntervalTimeManager(clean_interval=100, cluster_interval=1000, log_interval=100)
    
    metric = TfidfCosineDistance()
    fader = TfIdfExpFade(decay_rate=0.002, term_fading=True, min_weight=2 ** (-1 * 0.002 * 100))
    merger = SumMerger()
    micro_clusterer = TfIdfAdaptiveThresholdMicroclusterer(sigma=1, distance_metric=metric, fader=fader, merger=merger)
    history = HistoryManager()
    
    macro_clusterer = SimpleMacroClusterer()
    
    clusterer = textClust(wrapper, preprocessor, time_manager, micro_clusterer, macro_clusterer, history)
    clusterer.start()