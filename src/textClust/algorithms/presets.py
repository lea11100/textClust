import time
from typing import Optional

from sentence_transformers import SentenceTransformer

from textClust.core.textClust import textClust
from textClust.preprocessing.preprocessor import Preprocessor, TFIDFPreprocessor, EmbeddingPreprocessor
from textClust.online.clusterer import EmbeddingThresholdClusterer
from textClust.online.metrics import CosineDistance
from textClust.online.fade import ExpFade
from textClust.online.merge import WeightedSumMerger
from textClust.online.clusterer import TfIdfThresholdMicroclusterer
from textClust.online.merge import SumMerger
from textClust.time.manager import FixedIntervalTimeManager
from textClust.history.manager import HistoryManager
from textClust.offline.clusterer import SimpleMacroClusterer
from textClust.data.stream import Stream

from textClust.online.fast_sqrt_clusterer import FastSqrtClusterer, FastSqrtFader, FastSqrtCosineDistance

from textClust.online.fast_log_clusterer import FastLogClusterer, FastSumMerger, FastLogFader, FastLogCosineDistance 

def get_textClust(
    data_loader: Stream, clean_interval: int, cluster_interval: int,
    log_interval: int, decay_rate: float, adaptive_threshold: bool = True, term_fading: bool = True,
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
            distance_metric=CosineDistance(), 
            fader=ExpFade(
                decay_rate=decay_rate,
                term_fading=term_fading,
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
    min_num_mc: int = 5, normalize: bool = True, device: str = "cpu", **kwargs):
    if embedding_model is None and embedding_model_name is not None:
        embedding_model = SentenceTransformer(embedding_model_name, device=device)
    return textClust(
        data_loader=data_loader,
        preprocessor=EmbeddingPreprocessor(embedding_model=embedding_model, normalize=normalize) if embedding_model is not None else Preprocessor(),
        time_manager=FixedIntervalTimeManager(clean_interval=clean_interval, cluster_interval=cluster_interval, log_interval=log_interval),
        micro_clusterer=EmbeddingThresholdClusterer(
            adaptive_threshold=adaptive_threshold,
            sigma=sigma, 
            radius=radius,
            distance_metric=CosineDistance(assume_normalized=normalize), 
            fader=ExpFade(decay_rate=decay_rate, use_heuristic=use_heuristic, time_gap=time_gap, min_weight=min_weight), 
            merger=WeightedSumMerger(normalize=normalize), 
            min_num_mc=min_num_mc,
            threshold_momentum=threshold_momentum),
        macro_clusterer=SimpleMacroClusterer(),
        history_manager=HistoryManager()
    )
    
def get_fastSqrtClust(
    data_loader: Stream, clean_interval: int, cluster_interval: int,
    log_interval: int, decay_rate: float, adaptive_threshold: bool = True, term_fading: bool = True,
    sigma: Optional[float] = 1, radius: Optional[float] = None,
    threshold_momentum: Optional[float] = None, use_heuristic: bool = True,
    min_weight: Optional[float] = None, time_gap: Optional[float] = None,
    min_num_mc: int = 1,
):
    return textClust(
        data_loader=data_loader,
        preprocessor=TFIDFPreprocessor(),
        time_manager=FixedIntervalTimeManager(clean_interval=clean_interval, cluster_interval=cluster_interval, log_interval=log_interval),
        micro_clusterer=FastSqrtClusterer(
            adaptive_threshold=adaptive_threshold,
            sigma=sigma, 
            radius=radius,
            distance_metric=FastSqrtCosineDistance(), 
            fader=FastSqrtFader(
                decay_rate=decay_rate,
                term_fading=term_fading,
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
    log_interval: int, decay_rate: float, adaptive_threshold: bool = True, term_fading: bool = True,
    sigma: Optional[float] = 1, radius: Optional[float] = None,
    threshold_momentum: Optional[float] = None, use_heuristic: bool = True,
    min_weight: Optional[float] = None, time_gap: Optional[float] = None,
    min_num_mc: int = 1,
):
    return textClust(
        data_loader=data_loader,
        preprocessor=TFIDFPreprocessor(),
        time_manager=FixedIntervalTimeManager(clean_interval=clean_interval, cluster_interval=cluster_interval, log_interval=log_interval),
        micro_clusterer=FastLogClusterer(
            adaptive_threshold=adaptive_threshold,
            sigma=sigma, 
            radius=radius,
            distance_metric=FastLogCosineDistance(), 
            fader=FastLogFader(
                decay_rate=decay_rate,
                term_fading=term_fading,
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