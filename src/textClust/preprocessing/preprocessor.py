"""Text preprocessing components used before micro-clustering."""

import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

from textClust.data.point import RawPoint, Point, point_from_raw_point
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from typing import List
import numpy as np
import torch

class Preprocessor:
    """Base callable preprocessor interface."""

    def __init__(self):
        """Initialize the preprocessor."""
        pass
    
    def __call__(self, point: RawPoint) -> Point:
        """Transform a raw point into a processed point."""
        raise NotImplementedError("preprocess() must be implemented by subclass")

class TFIDFPreprocessor(Preprocessor):
    """Rule-based text preprocessor for token-based TF-IDF workflows."""

    def __init__(self, 
                 language='english',
                 stopword_removal=True,
                 stemming=False,
                 punctuation=True,
                 hashtag=True,
                 username=True,
                 url=True,
                 exclude_tokens=None):
        """Initialize preprocessing options.

        Args:
            language: Stopword language for NLTK stopword removal.
            stopword_removal: Whether to remove stopwords.
            stemming: Whether to apply Porter stemming.
            punctuation: Whether to strip punctuation.
            hashtag: Keep hashtag terms while removing `#` marker if true.
            username: Whether to remove @mentions.
            url: Whether to remove URLs.
            exclude_tokens: Optional custom token blocklist.
        """
        
        self.language = language
        self.stopword_removal = stopword_removal
        self.stemming = stemming
        self.remove_punctuation = punctuation
        self.keep_hashtag = hashtag
        self.remove_username = username
        self.remove_url = url
        self.exclude_tokens = exclude_tokens or []
        self.stopwords = set(stopwords.words(self.language)) if self.stopword_removal else set()
        self.stemmer = PorterStemmer() if self.stemming else None
    
    def __call__(self, point: RawPoint) -> Point:
        """Preprocess the input point text and return a `Point`."""
        processed_text_tokens = self.preprocess_text(point.text)
        return point_from_raw_point(point, processed_text_tokens=processed_text_tokens)
    
    def preprocess_text(self, text: str) -> List[str]:
        """Normalize and tokenize text according to configured options."""
        
        text = re.sub(r'https?://\S+|www\.\S+', '', text) if self.remove_url else text # Remove URLs
        text = re.sub(r'#(\w+)', r'\1', text) if self.keep_hashtag else re.sub(r'#\w+', '', text) # Handle hashtags
        text = re.sub(r'@\w+', '', text) if self.remove_username else text # Remove usernames
        text = text.lower() # Lowercase  
        if self.remove_punctuation: # Remove punctuation
            text = re.sub(r'[^\w\s\']|_', ' ', text)
            text = re.sub(r'\s\'\s|\s\'|^\s', ' ', text)
        
        tokens = text.split() # Tokenize
        tokens = [t for t in tokens if t not in self.stopwords] if self.stopwords else tokens # Remove stopwords
        
        if self.exclude_tokens: # Remove excluded tokens
            exclude_set = set([t.lower() for t in self.exclude_tokens])
            tokens = [t for t in tokens if t not in exclude_set]
        
        tokens = [self.stemmer.stem(t) for t in tokens] if self.stemming and self.stemmer else tokens # Apply stemming
        tokens = [t for t in tokens if len(t) >= 2 and not t.isdigit()] # Remove short tokens and numbers
        
        return tokens
    
class TFIDFPreprocessorWithTranslation(TFIDFPreprocessor):
    """TFIDFPreprocessor that translates text to English before preprocessing."""

    def __init__(self, dataset: str, folder: str = "data/translated_no_emojis", *args, **kwargs):
        """Initialize with a dataset name and folder for translation."""
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.folder = folder
        
        eng_data = np.load(f"{self.folder}/{self.dataset}.npy", allow_pickle=True).item()
        for k in ["tweet_ids", "tweets_eng"]:
            assert k in eng_data, f"Key '{k}' not found in translation data."
        self.tweet_id_to_eng = dict(zip(eng_data["tweet_ids"], eng_data["tweets_eng"]))
        
    
    def __call__(self, point: RawPoint) -> Point:
        """Translate the input point text to English, preprocess it, and return a `Point`."""
        translated_text = self.tweet_id_to_eng.get(point.text_id)
        if not translated_text:
            raise ValueError(f"Translated text for ID {point.text_id} not found.")
        processed_text_tokens = self.preprocess_text(translated_text)
        return point_from_raw_point(point, processed_text_tokens=processed_text_tokens)
    

class EmbeddingPreprocessor(Preprocessor):
    """Preprocessor that generates vector embeddings for input text."""

    def __init__(self, embedding_model: SentenceTransformer):
        """Initialize with a callable embedding model."""
        self.embedding_model = embedding_model
    
    def __call__(self, point: RawPoint) -> Point:
        """Generate an embedding for the input point text and return a `Point`."""
        embedding = self.embedding_model.encode(point.text, convert_to_tensor=True, show_progress_bar=False)
        return point_from_raw_point(point, embedding=embedding)
    
class SimulatedEmbeddingPreprocessor(Preprocessor):
    """Preprocessor that loads precomputed embeddings for testing purposes.

    Embeddings are stored at ``embedded/<model>/<dataset>.npy`` below the
    configured data folder.
    """

    def __init__(self, model: str, dataset: str, folder: str = "data/embedded"):
        """Initialize with a specified embedding dimension."""
        self.model = model
        self.dataset = dataset
        self.folder = folder

        data = np.load(f"{self.folder}/{self.model}/{self.dataset}.npy", allow_pickle=True).item()
        for k in ["tweet_ids", "embeddings", "times"]:
            assert k in data, f"Key '{k}' not found in simulated embedding data."
        
        if "dataset_name" in data:
            assert data["dataset_name"] == self.dataset, f"Dataset name mismatch: expected {self.dataset}, found {data['dataset_name']}."
        if "model_name" in data:
            assert data["model_name"] == self.model, f"Model name mismatch: expected {self.model}, found {data['model_name']}."
        
        self.tweet_id_to_embedding = dict(zip(data["tweet_ids"], data["embeddings"]))
        self.tweet_id_to_time = dict(zip(data["tweet_ids"], data["times"]))
        
        self.saved_time = 0.0
    
    def __call__(self, point: RawPoint) -> Point:
        """Generate a simulated embedding for the input point text and return a `Point`."""
        assert point.text_id is not None, "RawPoint must have a text_id for simulated embedding lookup."
        assert point.text_id in self.tweet_id_to_embedding, f"Text ID {point.text_id} not found in simulated embedding dataset."
        assert point.text_id in self.tweet_id_to_time, f"Text ID {point.text_id} not found in simulated time dataset."
        
        simulated_embedding = self.tweet_id_to_embedding[point.text_id]
        time_to_wait = self.tweet_id_to_time[point.text_id]
        self.saved_time += time_to_wait
        return point_from_raw_point(point, embedding=simulated_embedding)