import time
from typing import Optional

from sentence_transformers import SentenceTransformer

from textClust.core.textClust import textClust
from textClust.preprocessing.preprocessor import TFIDFPreprocessor, EmbeddingPreprocessor
from textClust.online.clusterer import EmbeddingThresholdClusterer
from textClust.online.metrics import EmbeddingCosineDistance
from textClust.online.fade import EmbeddingExpFader
from textClust.online.merge import WeightedSumMerger
from textClust.online.clusterer import TfIdfThresholdMicroclusterer
from textClust.online.metrics import TfidfCosineDistance
from textClust.online.fade import TfIdfExpFade
from textClust.online.merge import SumMerger
from textClust.time.manager import FixedIntervalTimeManager
from textClust.history.manager import HistoryManager
from textClust.offline.clusterer import SimpleMacroClusterer, TfidfToEmbeddingMacroClusterer
from textClust.data.stream import Stream

from textClust.online.fast_clusterer import FastClusterer, FastSumMerger, FastFader, FastCosineDistance

from textClust.online.fast_log_clusterer import FastClusterer as FastLogCluster, FastSumMerger as FastLogSumMerger, FastFader as FastLogMerger, FastCosineDistance as FastLogCosineDistance 

def get_textClust(
    data_loader: Stream, clean_interval: int, cluster_interval: int,
    log_interval: int, decay_rate: float, adaptive_threshold: bool = True,
    sigma: Optional[float] = 1, radius: Optional[float] = None,
    threshold_momentum: Optional[float] = None, use_heuristic: bool = True,
    min_weight: Optional[float] = None, time_gap: Optional[float] = None,
    min_num_mc: int = 1,
):
    return textClust(
        data_loader=data_loader,
        preprocessor=TFIDFPreprocessor(),
        time_manager=FixedIntervalTimeManager(clean_interval=clean_interval, cluster_interval=cluster_interval, log_interval=log_interval),
        micro_clusterer=TfIdfThresholdMicroclusterer(
            adaptive_threshold=adaptive_threshold,
            sigma=sigma, 
            radius=radius,
            distance_metric=TfidfCosineDistance(), 
            fader=TfIdfExpFade(
                decay_rate=decay_rate,
                term_fading=False,
                use_heuristic=use_heuristic,
                time_gap=time_gap,
                min_weight=min_weight,
            ),
            merger=SumMerger(), 
            min_num_mc=min_num_mc,
            threshold_momentum=threshold_momentum),
        macro_clusterer=SimpleMacroClusterer(),
        history_manager=HistoryManager()
    )
    
def get_embClust(
    data_loader: Stream, clean_interval: int, 
    cluster_interval: int, log_interval: int, decay_rate: float, 
    embedding_model_name: Optional[str] = None, embedding_model:Optional[SentenceTransformer] = None,
    adaptive_threshold = True, sigma: Optional[float] = 1, radius: Optional[float] = None,
    threshold_momentum: Optional[float] = None, use_heuristic: bool = True, min_weight: Optional[float] = None, time_gap: Optional[float] = None,
    min_num_mc: int = 5, device: str = "cpu", **kwargs):
    assert embedding_model_name is not None or embedding_model is not None, "Either embedding_model_name or embedding_model must be provided"
    if embedding_model is None:
        embedding_model = SentenceTransformer(embedding_model_name, device=device)
    return textClust(
        data_loader=data_loader,
        preprocessor=EmbeddingPreprocessor(embedding_model=embedding_model),
        time_manager=FixedIntervalTimeManager(clean_interval=clean_interval, cluster_interval=cluster_interval, log_interval=log_interval),
        micro_clusterer=EmbeddingThresholdClusterer(
            adaptive_threshold=adaptive_threshold,
            sigma=sigma, 
            radius=radius,
            distance_metric=EmbeddingCosineDistance(), 
            fader=EmbeddingExpFader(decay_rate=decay_rate, use_heuristic=use_heuristic, time_gap=time_gap, min_weight=min_weight), 
            merger=WeightedSumMerger(), 
            min_num_mc=min_num_mc,
            threshold_momentum=threshold_momentum),
        macro_clusterer=SimpleMacroClusterer(),
        history_manager=HistoryManager()
    )
    
def get_fastClust(
    data_loader: Stream, clean_interval: int, cluster_interval: int,
    log_interval: int, decay_rate: float, adaptive_threshold: bool = True,
    sigma: Optional[float] = 1, radius: Optional[float] = None,
    threshold_momentum: Optional[float] = None, use_heuristic: bool = True,
    min_weight: Optional[float] = None, time_gap: Optional[float] = None,
    min_num_mc: int = 1,
):
    return textClust(
        data_loader=data_loader,
        preprocessor=TFIDFPreprocessor(),
        time_manager=FixedIntervalTimeManager(clean_interval=clean_interval, cluster_interval=cluster_interval, log_interval=log_interval),
        micro_clusterer=FastClusterer(
            adaptive_threshold=adaptive_threshold,
            sigma=sigma, 
            radius=radius,
            distance_metric=FastCosineDistance(), 
            fader=FastFader(
                decay_rate=decay_rate,
                term_fading=False,
                use_heuristic=use_heuristic,
                time_gap=time_gap,
                min_weight=min_weight,
            ),
            merger=FastSumMerger(), 
            min_num_mc=min_num_mc,
            threshold_momentum=threshold_momentum),
        macro_clusterer=SimpleMacroClusterer(),
        history_manager=HistoryManager()
    )
    
def get_fastLogClust(
    data_loader: Stream, clean_interval: int, cluster_interval: int,
    log_interval: int, decay_rate: float, adaptive_threshold: bool = True,
    sigma: Optional[float] = 1, radius: Optional[float] = None,
    threshold_momentum: Optional[float] = None, use_heuristic: bool = True,
    min_weight: Optional[float] = None, time_gap: Optional[float] = None,
    min_num_mc: int = 1,
):
    return textClust(
        data_loader=data_loader,
        preprocessor=TFIDFPreprocessor(),
        time_manager=FixedIntervalTimeManager(clean_interval=clean_interval, cluster_interval=cluster_interval, log_interval=log_interval),
        micro_clusterer=FastLogCluster(
            adaptive_threshold=adaptive_threshold,
            sigma=sigma, 
            radius=radius,
            distance_metric=FastLogCosineDistance(), 
            fader=FastLogMerger(
                decay_rate=decay_rate,
                term_fading=False,
                use_heuristic=use_heuristic,
                time_gap=time_gap,
                min_weight=min_weight,
            ),
            merger=FastLogSumMerger(), 
            min_num_mc=min_num_mc,
            threshold_momentum=threshold_momentum),
        macro_clusterer=SimpleMacroClusterer(),
        history_manager=HistoryManager()
    )