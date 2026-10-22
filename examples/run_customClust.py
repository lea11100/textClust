from textClust.core.textClust import textClust
from textClust.data.wrapper import StaticJsonWrapper
from textClust.preprocessing.preprocessor import TFIDFPreprocessor
from textClust.online.fast_sqrt_clusterer import FastSqrtClusterer, FastSumMerger, FastSqrtFader, FastSqrtCosineDistance
from textClust.time.manager import FixedIntervalTimeManager
from textClust.history.manager import HistoryManager
from textClust.offline.clusterer import SimpleMacroClusterer

if __name__ == "__main__":
    
    wrapper = StaticJsonWrapper("examples/data/raw_data/News-T.json")
    preprocessor = TFIDFPreprocessor()  
    time_manager = FixedIntervalTimeManager(clean_interval=100, cluster_interval=1000, log_interval=100)
    metric = FastSqrtCosineDistance()
    fader = FastSqrtFader(decay_rate=0.005, term_fading=True, min_weight=2 ** (-1 * 0.005 * 1050))
    merger = FastSumMerger()
    micro_clusterer = FastSqrtClusterer(distance_metric=metric, fader=fader, merger=merger, sigma=1, min_num_mc=5, threshold_momentum=None)
    history = HistoryManager()
    macro_clusterer = SimpleMacroClusterer()

    clusterer = textClust(wrapper, preprocessor, time_manager, micro_clusterer, macro_clusterer, history)
    clusterer.start()