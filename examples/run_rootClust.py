import time

from sentence_transformers import SentenceTransformer

from textClust.core.textClust import textClust
from textClust.data.wrapper import StaticJsonWrapper
from textClust.preprocessing.preprocessor import TFIDFPreprocessor
from textClust.online.clusterer import TfIdfAdaptiveThresholdMicroclusterer
from textClust.online.root_clusterer import RootClusterer, RootSumMerger, RootFader, RootCosineDistance
from textClust.online.fade import TfIdfExpFade
from textClust.online.merge import SumMerger
from textClust.time.manager import FixedIntervalTimeManager
from textClust.history.manager import HistoryManager
from textClust.offline.clusterer import SimpleMacroClusterer, TfidfToEmbeddingMacroClusterer, LlmMacroClusterer

if __name__ == "__main__":
    start_time = time.time()
    wrapper = StaticJsonWrapper("examples/data/Semantic-T_peaked.json")
    preprocessor = TFIDFPreprocessor()
    
    time_manager = FixedIntervalTimeManager(clean_interval=100, cluster_interval=2000, log_interval=100)
    
    metric = RootCosineDistance()
    fader = RootFader(decay_rate=0.005, term_fading=True, min_weight=2 ** (-1 * 0.005 * 1050))
    merger = RootSumMerger()
    micro_clusterer = RootClusterer(distance_metric=metric, fader=fader, merger=merger, sigma=2.5, min_num_mc=5, threshold_momentum=None)
    history = HistoryManager()
    
    macro_clusterer = LlmMacroClusterer(model_name="Qwen/Qwen2.5-32B-Instruct:featherless-ai", use_hf=True)
    #macro_clusterer = LlmMacroClusterer(model_name="Qwen/Qwen2.5-32B-Instruct:featherless-ai")
    #macro_clusterer = TfidfToEmbeddingMacroClusterer(SentenceTransformer("all-MiniLM-L6-v2", device="mps"), epsilon=0.3, min_samples=5)
    #macro_clusterer = SimpleMacroClusterer()
    
    clusterer = textClust(wrapper, preprocessor, time_manager, micro_clusterer, macro_clusterer, history)
    clusterer.start()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time} seconds")