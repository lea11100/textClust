from textClust.core.textClust import textClust
from textClust.data.wrapper import StaticJsonWrapper
from textClust.preprocessing.preprocessor import TFIDFPreprocessor
from textClust.online.fast_clusterer import FastClusterer, FastSumMerger, FastFader, FastCosineDistance
from textClust.time.manager import FixedIntervalTimeManager
from textClust.history.manager import HistoryManager
from textClust.offline.clusterer import LlmMacroClusterer

if __name__ == "__main__":
    
    wrapper = StaticJsonWrapper("examples/data/News-T.json")
    preprocessor = TFIDFPreprocessor()  
    time_manager = FixedIntervalTimeManager(clean_interval=100, cluster_interval=1000, log_interval=100)
    metric = FastCosineDistance()
    fader = FastFader(decay_rate=0.005, term_fading=True, min_weight=2 ** (-1 * 0.005 * 1050))
    merger = FastSumMerger()
    micro_clusterer = FastClusterer(distance_metric=metric, fader=fader, merger=merger, sigma=1, min_num_mc=5, threshold_momentum=None)
    history = HistoryManager()
    macro_clusterer = LlmMacroClusterer(model_name="Qwen/Qwen2.5-32B-Instruct:featherless-ai", use_hf=True)

    clusterer = textClust(wrapper, preprocessor, time_manager, micro_clusterer, macro_clusterer, history)
    clusterer.start()