"""
Translate and embed tweets from multiple datasets using several
sentence-transformer models, timing each individual encode call, and store
the results per dataset and model.

Expected input format
----------------------
Each dataset record is expected to behave like a sequence where:
    record[0]  ->  the tweet id
    record[1]  ->  the tweet text (str)

Datasets can be either .npy files (loaded with np.load) or .json files
(loaded with json.load) -- see load_dataset(). Tweets are translated to
English after emojis are converted to textual names.

Output format
-------------
For each dataset/model pair, we save OUTPUT_DIR/<dataset>/<model>.npy
containing a dictionary with "dataset_name", "model_name", "tweet_ids",
"embeddings", and "times".

Memory note
-----------
Only ONE model is ever loaded into memory at a time: main() loops over
models on the outside, encodes every dataset with the currently loaded
model, then frees it before loading the next one. This matters even on CPU
since these models can add up, and is essential if DEVICE is switched to a
GPU.

Usage
-----
Edit DATASETS and MODEL_IDS below, then run:
    python src/textClust/preprocessing/offline_preprocessing.py
"""

import gc
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import emoji
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer

from textClust.data.wrapper import StaticJsonWrapper, SocialEDWrapper
from textClust.data.point import RawPoint

try:
    import torch
except ImportError:
    torch = None

# ---------------------------------------------------------------------------
# Configuration -- edit these for your setup
# ---------------------------------------------------------------------------

# Datasets: friendly name -> path. Supports both .npy (loaded via np.load)
# and .json (loaded via json.load) files. Each record is still expected to
# provide the tweet id at index 0 and the tweet text at index 1.
DATASETS = {
    "Arabic_7": "examples/social_ed_data/Arabic_7.npy",
    "Arabic_Twitter": "examples/social_ed_data/Arabic_Twitter.npy",
    "CrisisLexT6": "examples/social_ed_data/CrisisLexT6.npy",
    "CrisisLexT7": "examples/social_ed_data/CrisisLexT7.npy",
    "CrisisLexT26": "examples/social_ed_data/CrisisLexT26.npy",
    "CrisisMMD": "examples/social_ed_data/CrisisMMD.npy",
    "CrisisNLP": "examples/social_ed_data/CrisisNLP.npy",
    "Event2012": "examples/social_ed_data/Event2012.npy",
    "Event2018": "examples/social_ed_data/Event2018.npy",
    "HumAID": "examples/social_ed_data/HumAID.npy",
    "KBP": "examples/social_ed_data/KBP.npy",
    "MAVEN": "examples/social_ed_data/MAVEN.npy",
    "Mix_Data": "examples/social_ed_data/Mix_Data.npy",
    "Semantic5-T": "examples/data/Semantic5-T.json",
    "Semantic10-T": "examples/data/Semantic10-T.json",
    "Semantic40-T": "examples/data/Semantic40-T.json",
}

# Models: friendly name -> HuggingFace model id. Only one embedding model is ever
# instantiated at a time (see main()), so this stays a plain id mapping
# rather than pre-built SentenceTransformer objects.
MODEL_IDS = {
    "NeuML": "NeuML/pubmedbert-base-embeddings-2M",
    "minishlab_2M": "minishlab/potion-base-2M",
    "minishlab_4M": "minishlab/potion-base-4M",
    "minishlab_8M": "minishlab/potion-base-8M",
    "minishlab_32M": "minishlab/potion-retrieval-32M",
    "stephantulkens": "stephantulkens/NIFE-mxbai-embed-large-v1_as_router",
    "Mihaiii_Bulbasaur": "Mihaiii/Bulbasaur",
    "Mihaiii_gte_micro_v4": "Mihaiii/gte-micro-v4",
    "sentence_transformers_all_MiniLM_L6_v2": "sentence-transformers/all-MiniLM-L6-v2",
    "Snowflake": "Snowflake/snowflake-arctic-embed-xs",
    "MongoDB": "MongoDB/mdbr-leaf-ir",
    "intfloat": "intfloat/e5-base-v2",
    "Alibaba_NLP": "Alibaba-NLP/gte-modernbert-base",
    "microsoft": "microsoft/harrier-oss-v1-270m",
}

# All models here are run on CPU. Change to "cuda" if/when running on a GPU.
DEVICE = "cuda"

# Hugging Face models used to identify each tweet's source language and to
# translate it to English. They are loaded once and released before embedding.
LANGUAGE_ID_MODEL = "facebook/fasttext-language-identification"
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"

# Text preprocessing is performed in batches. Increase this if the selected
# device has enough memory.
PREPROCESS_BATCH_SIZE = 32

# Index within each record where the tweet id and tweet text are stored.
ID_INDEX = 0
TEXT_INDEX = 1

# Where to write the per-dataset result files.
OUTPUT_DIR = Path("results")

# One checkpoint per input dataset. Embedding models can then be changed or
# rerun without downloading/translating the datasets again.
PREPROCESSED_DIR = Path("translated/to_en")

# Remove existing result files before starting. When false, completed model /
# dataset pairs are resumed from their existing result file.
CLEAR_OUTPUT = False


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _normalise_language_label(label: str) -> str:
    """Convert fastText labels to the NLLB language-code format."""
    return label.removeprefix("__label__")


def _language_code(
    label: str, translation_tokenizer: AutoTokenizer
) -> Union[str, None]:
    """Return an NLLB source code, or ``None`` for unsupported languages."""
    code = _normalise_language_label(label)
    if code in translation_tokenizer.lang_code_to_id:
        return code

    # Some versions of the fastText checkpoint emit ISO-639-1/3 labels while
    # NLLB uses ISO-639-3 plus a script. Prefer the tokenizer's matching code.
    candidates = [
        nllb_code for nllb_code in translation_tokenizer.lang_code_to_id
        if nllb_code.startswith(f"{code}_")
    ]
    if candidates:
        return candidates[0]
    iso1_to_iso3 = {"en": "eng", "de": "deu", "fr": "fra", "es": "spa", "it": "ita", "pt": "por", "nl": "nld", "ar": "arb", "ru": "rus", "uk": "ukr", "tr": "tur", "pl": "pol"}
    iso3 = iso1_to_iso3.get(code, code)
    candidates = [
        nllb_code for nllb_code in translation_tokenizer.lang_code_to_id
        if nllb_code.startswith(f"{iso3}_")
    ]
    return candidates[0] if candidates else None


def _predict_languages(
    texts: List[str], language_tokenizer: AutoTokenizer,
    language_model: AutoModelForSequenceClassification, batch_size: int,
) -> List[str]:
    """Identify tweet languages in batches using the fastText HF checkpoint."""
    labels: List[str] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = language_tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
        ).to(DEVICE)
        with torch.inference_mode():
            predictions = language_model(**inputs).logits.argmax(dim=-1)
        labels.extend(language_model.config.id2label[int(index)] for index in predictions)
    return labels


def preprocess_tweets(
    texts: List[str], language_tokenizer: AutoTokenizer,
    language_model: AutoModelForSequenceClassification,
    translation_tokenizer: AutoTokenizer,
    translation_model: AutoModelForSeq2SeqLM,
    batch_size: int,
) -> Tuple[List[str], List[str]]:
    """Demojify, identify languages, and translate tweets in batches."""
    demojified = [emoji.demojize("" if text is None else str(text)) for text in texts]
    initial_language = _predict_languages(
        demojified, language_tokenizer, language_model, batch_size,
    )
    translated = list(demojified)
    grouped: Dict[str, List[int]] = {}
    for index, label in enumerate(initial_language):
        source_code = _language_code(label, translation_tokenizer)
        if source_code is not None and source_code != "eng_Latn":
            grouped.setdefault(source_code, []).append(index)

    for source_code, indices in grouped.items():
        translation_tokenizer.src_lang = source_code
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            inputs = translation_tokenizer(
                [demojified[index] for index in batch_indices],
                return_tensors="pt", padding=True, truncation=True,
            ).to(DEVICE)
            with torch.inference_mode():
                output = translation_model.generate(
                    **inputs,
                    forced_bos_token_id=translation_tokenizer.lang_code_to_id["eng_Latn"],
                    max_new_tokens=256,
                )
            decoded = translation_tokenizer.batch_decode(output, skip_special_tokens=True)
            for index, text in zip(batch_indices, decoded):
                translated[index] = text
    return translated, initial_language

def load_dataset(
    dataset_path: Union[str, Path]
) -> Tuple[np.ndarray, List[str]]:
    """
    Load a dataset's tweet ids and tweet texts (no embeddings yet).

    Supports .npy files (loaded with np.load) and .json files (loaded with
    json.load). Either way, each record is expected to behave like a
    sequence where record[ID_INDEX] is the tweet id and record[TEXT_INDEX]
    is the tweet text.
    """
    suffix = Path(dataset_path).suffix.lower()

    if suffix == ".npy":
        stream = SocialEDWrapper(str(dataset_path), max_samples=None, random_start_seed=None).load()
    elif suffix == ".json":
        stream = StaticJsonWrapper(str(dataset_path), max_samples=None, random_start_seed=None).load()
    else:
        raise ValueError(f"Unsupported dataset file type: {dataset_path}")

    data: List[RawPoint] = []
    while True:
        point = next(stream)
        if point is None:
            break
        data.append(point)

    tweet_ids = np.array([point.text_id for point in data])
    tweets = [point.text for point in data]
    return tweet_ids, tweets


def preprocessed_path(dataset_name: str) -> Path:
    """Return the checkpoint path for a translated dataset."""
    return PREPROCESSED_DIR / f"{dataset_name}.npy"


def save_preprocessed_dataset(
    path: Path, tweet_ids: np.ndarray, tweets: List[str], initial_language: List[str],
) -> None:
    """Persist preprocessing output so it can be reused by all embeddings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tweet_ids": tweet_ids,
        "tweets": np.asarray(tweets, dtype=object),
        "initial_language": np.asarray(initial_language, dtype=object),
    }
    np.save(path, np.array(payload, dtype=object), allow_pickle=True)


def load_preprocessed_dataset(path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load a preprocessing checkpoint."""
    payload = np.load(path, allow_pickle=True).item()
    return payload["tweet_ids"], payload["tweets"].tolist()


def embed_tweets_with_model(
    tweets: List[str], model: SentenceTransformer, model_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode every tweet individually with a single loaded model, timing each
    encode call.

    Returns
    -------
    embeddings : object ndarray, shape (n_tweets,)
    times      : float64 ndarray, shape (n_tweets,)
    """
    n_tweets = len(tweets)
    embeddings = np.empty(n_tweets, dtype=object)
    times = np.zeros(n_tweets, dtype=np.float64)

    for t_idx, tweet in enumerate(tweets):
        # Guard against missing/non-string tweet text
        text = "" if tweet is None else str(tweet)

        start = time.perf_counter()
        vec = model.encode(text)
        elapsed = time.perf_counter() - start

        embeddings[t_idx] = np.asarray(vec, dtype=np.float32)
        times[t_idx] = elapsed

    print(f"  finished model '{model_name}': "
          f"mean encode time {times.mean():.4f}s over {n_tweets} tweets")
    return embeddings, times





def result_path(dataset_name: str, model_name: str) -> Path:
    """Return the checkpoint path for one dataset/model pair."""
    return OUTPUT_DIR / dataset_name / f"{model_name}.npy"


def result_has_model(results_path: Path) -> bool:
    """Check whether a dataset/model checkpoint already exists."""
    return results_path.exists()


def save_model_results(
    results_path: Path,
    tweet_ids: np.ndarray,
    embeddings: np.ndarray,
    times: np.ndarray,
    dataset_name: str,
    model_name: str,
) -> None:
    """Save one model's embeddings and metadata for a dataset."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tweet_ids": tweet_ids,
        "embeddings": embeddings,
        "times": times,
        "dataset_name": dataset_name,
        "model_name": model_name,
    }
    np.save(results_path, np.array(payload, dtype=object), allow_pickle=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    model_names = list(MODEL_IDS.keys())
    dataset_names = list(DATASETS.keys())

    if CLEAR_OUTPUT:
        for dataset_name in dataset_names:
            dataset_output_dir = OUTPUT_DIR / dataset_name
            if dataset_output_dir.exists():
                shutil.rmtree(dataset_output_dir)
                print(f"Removed existing results: {dataset_output_dir}")
        if PREPROCESSED_DIR.exists():
            shutil.rmtree(PREPROCESSED_DIR)
            print(f"Removed existing translated checkpoints: {PREPROCESSED_DIR}")

    print("=" * 72)
    print("Starting embedding pipeline")
    print(f"- device: {DEVICE}")
    print(f"- datasets: {len(dataset_names)}")
    print(f"- models: {len(model_names)}")
    print(f"- output directory: {OUTPUT_DIR}")
    print(f"- preprocessed data: {PREPROCESSED_DIR}")
    print("=" * 72)

    if torch is None:
        raise RuntimeError("PyTorch is required for Hugging Face preprocessing models")

    # Preprocess each dataset once, independently of the embedding models.
    language_tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_ID_MODEL)
    language_model = AutoModelForSequenceClassification.from_pretrained(
        LANGUAGE_ID_MODEL,
    ).to(DEVICE)
    translation_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
    translation_model = AutoModelForSeq2SeqLM.from_pretrained(
        TRANSLATION_MODEL,
    ).to(DEVICE)
    try:
        for d_idx, (dataset_name, dataset_path) in enumerate(DATASETS.items()):
            preprocessing_path = preprocessed_path(dataset_name)
            if preprocessing_path.exists():
                print(f"Skipping preprocessed dataset: {dataset_name}")
                continue
            print(
                f"Preprocessing dataset [{d_idx + 1}/{len(dataset_names)}]: "
                f"{dataset_name} ({dataset_path})"
            )
            tweet_ids, raw_tweets = load_dataset(dataset_path)
            tweets, initial_language = preprocess_tweets(
                raw_tweets,
                language_tokenizer,
                language_model,
                translation_tokenizer,
                translation_model,
                PREPROCESS_BATCH_SIZE,
            )
            save_preprocessed_dataset(
                preprocessing_path, tweet_ids, tweets, initial_language,
            )
            print(f"  -> saved {preprocessing_path}")
            del tweet_ids, raw_tweets, tweets, initial_language
            gc.collect()
    finally:
        language_model.to("cpu")
        translation_model.to("cpu")
        del language_model, translation_model, language_tokenizer, translation_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Process one model and one preprocessed dataset at a time. Dataset data
    # is released after its checkpoint is written, and completed model rows
    # are skipped.
    for m_idx, name in enumerate(model_names):
        model_id = MODEL_IDS[name]
        print("\n" + "-" * 72)
        print(f"Model [{m_idx + 1}/{len(model_names)}]: {name} ({model_id})")
        model = SentenceTransformer(model_id, device=DEVICE)
        try:
            for d_idx, (dataset_name, dataset_path) in enumerate(DATASETS.items()):
                output_path = result_path(dataset_name, name)
                if result_has_model(output_path):
                    print(f"  skipping completed dataset: {dataset_name}")
                    continue

                preprocessing_path = preprocessed_path(dataset_name)
                tweet_ids, tweets = load_preprocessed_dataset(preprocessing_path)
                print(f"  loading preprocessed dataset: {dataset_name}")
                print(f"  encoding dataset: {dataset_name} ({len(tweets)} tweets)")
                embeddings, times = embed_tweets_with_model(tweets, model, name)
                save_model_results(
                    output_path,
                    tweet_ids,
                    embeddings,
                    times,
                    dataset_name,
                    name,
                )
                del tweet_ids, tweets, embeddings, times
                gc.collect()
                print(f"    -> saved {output_path}")
        finally:
            model.to("cpu")
            del model
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"Finished model {name}")

    print("Embedding pipeline completed")
    print("=" * 72)


if __name__ == "__main__":
    main()