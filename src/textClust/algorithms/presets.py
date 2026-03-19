import time

from sentence_transformers import SentenceTransformer

from textClust.core.textClust import textClust
from textClust.preprocessing.preprocessor import TFIDFPreprocessor, EmbeddingPreprocessor
from textClust.online.clusterer import EmbeddingAdaptiveThresholdClusterer
from textClust.online.metrics import EmbeddingCosineDistance
from textClust.online.fade import EmbeddingExpFader
from textClust.online.merge import WeightedSumMerger
from textClust.online.clusterer import TfIdfAdaptiveThresholdMicroclusterer
from textClust.online.metrics import TfidfCosineDistance
from textClust.online.fade import TfIdfExpFade
from textClust.online.merge import SumMerger
from textClust.time.manager import FixedIntervalTimeManager
from textClust.history.manager import HistoryManager
from textClust.offline.clusterer import SimpleMacroClusterer, TfidfToEmbeddingMacroClusterer
from textClust.data.stream import Stream

from textClust.online.fast_clusterer import FastClusterer, FastSumMerger, FastFader, FastCosineDistance

def get_textClust(data_loader: Stream, clean_interval: int, cluster_interval: int, log_interval: int, decay_rate: float, sigma: float):
    return textClust(
        data_loader=data_loader,
        preprocessor=TFIDFPreprocessor(),
        time_manager=FixedIntervalTimeManager(clean_interval=clean_interval, cluster_interval=cluster_interval, log_interval=log_interval),
        micro_clusterer=TfIdfAdaptiveThresholdMicroclusterer(
            sigma=sigma, 
            distance_metric=TfidfCosineDistance(), 
            fader=TfIdfExpFade(decay_rate=decay_rate, term_fading=False, min_weight=2 ** (-1 * decay_rate * cluster_interval)), 
            merger=SumMerger(), 
            min_num_mc=1, 
            threshold_momentum=None),
        macro_clusterer=SimpleMacroClusterer(),
        history_manager=HistoryManager()
    )
    
def get_embClust(data_loader: Stream, embedding_model_name: str, clean_interval: int, cluster_interval: int, log_interval: int, decay_rate: float, sigma: float):
    return textClust(
        data_loader=data_loader,
        preprocessor=EmbeddingPreprocessor(SentenceTransformer(embedding_model_name, device="auto")),
        time_manager=FixedIntervalTimeManager(clean_interval=clean_interval, cluster_interval=cluster_interval, log_interval=log_interval),
        micro_clusterer=EmbeddingAdaptiveThresholdClusterer(
            sigma=sigma, 
            distance_metric=EmbeddingCosineDistance(), 
            fader=EmbeddingExpFader(decay_rate=decay_rate, min_weight=2 ** (-1 * decay_rate * cluster_interval)), 
            merger=WeightedSumMerger(), 
            min_num_mc=5,
            threshold_momentum=None),
        macro_clusterer=SimpleMacroClusterer(),
        history_manager=HistoryManager()
    )
    
def get_fastClust(data_loader: Stream, clean_interval: int, cluster_interval: int, log_interval: int, decay_rate: float, sigma: float):
    return textClust(
        data_loader=data_loader,
        preprocessor=TFIDFPreprocessor(),
        time_manager=FixedIntervalTimeManager(clean_interval=clean_interval, cluster_interval=cluster_interval, log_interval=log_interval),
        micro_clusterer=FastClusterer(
            sigma=sigma, 
            distance_metric=FastCosineDistance(), 
            fader=FastFader(decay_rate=decay_rate, term_fading=False, min_weight=2 ** (-1 * decay_rate * cluster_interval)), 
            merger=FastSumMerger(), 
            min_num_mc=1, 
            threshold_momentum=None),
        macro_clusterer=SimpleMacroClusterer(),
        history_manager=HistoryManager()
    )