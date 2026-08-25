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
English through an OpenAI-compatible Chat Completions API.

Output format
-------------
For each dataset/model pair, we save
examples/data/embedded/<model>/<dataset>.npy
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

import ast
import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from textClust.data.wrapper import StaticJsonWrapper, SocialEDWrapper
from textClust.data.point import RawPoint

try:
    import torch
except ImportError:
    torch = None

# ---------------------------------------------------------------------------
# Configuration -- edit these for your setup
# ---------------------------------------------------------------------------

# All dataset artefacts live below examples/data. Raw input files are kept in
# raw_data, translated files in translated_no_emojis, and embedding
# outputs below embedded/<model_name>.
DATA_DIR = Path("examples/data")
RAW_DATA_DIR = DATA_DIR / "raw_data"
PREPROCESSED_DIR = DATA_DIR / "translated_no_emojis"
EMBEDDED_DIR = DATA_DIR / "embedded"

# Datasets: friendly name -> raw input path. Supports both .npy (loaded via
# np.load) and .json (loaded via json.load) files. Each record is still
# expected to provide the tweet id at index 0 and the tweet text at index 1.
DATASETS = {
    "Arabic_7": RAW_DATA_DIR / "Arabic_7.npy",
    "Arabic_Twitter": RAW_DATA_DIR / "Arabic_Twitter.npy",
    "CrisisLexT6": RAW_DATA_DIR / "CrisisLexT6.npy",
    "CrisisLexT7": RAW_DATA_DIR / "CrisisLexT7.npy",
    "CrisisLexT26": RAW_DATA_DIR / "CrisisLexT26.npy",
    "CrisisMMD": RAW_DATA_DIR / "CrisisMMD.npy",
    "CrisisNLP": RAW_DATA_DIR / "CrisisNLP.npy",
    "Event2012": RAW_DATA_DIR / "Event2012.npy",
    "Event2012_100": RAW_DATA_DIR / "Event2012_100.npy",
    "Event2018": RAW_DATA_DIR / "Event2018.npy",
    "Event2018_100": RAW_DATA_DIR / "Event2018_100.npy",
    "HumAID": RAW_DATA_DIR / "HumAID.npy",
    "KBP": RAW_DATA_DIR / "KBP.npy",
    "MAVEN": RAW_DATA_DIR / "MAVEN.npy",
    "Mix_Data": RAW_DATA_DIR / "Mix_Data.npy",
    "Semantic5-T": RAW_DATA_DIR / "Semantic5-T.json",
    "Semantic5-T_sorted": RAW_DATA_DIR / "Semantic5-T_sorted.json",
    "Semantic10-T": RAW_DATA_DIR / "Semantic10-T.json",
    "Semantic40-T": RAW_DATA_DIR / "Semantic40-T.json",
    # Do not use these, they are partly already processed
    # "NT": RAW_DATA_DIR / "NT.json",
    # "NTS": RAW_DATA_DIR / "NTS.json",
    # "NTSO-T": RAW_DATA_DIR / "NTSO-T.json",
    # "SO-T": RAW_DATA_DIR / "SO-T.json",
    # "News-T": RAW_DATA_DIR / "News-T.json",
    # "Trends-T": RAW_DATA_DIR / "Trends-T.json",
    # "Trends-T_modified": RAW_DATA_DIR / "Trends-T_modified.json",
    # "Tweets-T": RAW_DATA_DIR / "Tweets-T.json",
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

# OpenAI-compatible translation service. Store the key in the environment,
# never in this file. These settings can be overridden without code changes.
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL", "https://ai-gateway.uni-paderborn.de/v1"
)
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gwdg.qwen3.5-397b-a17b-no-think")
# Text preprocessing is performed in batches. Increase this if the selected
# device has enough memory.
PREPROCESS_BATCH_SIZE = 256

# Number of raw/translated tweet pairs to print after each dataset is
# translated. Set to 0 to disable sample output.
SAMPLE_COUNT = 3

# Index within each record where the tweet id and tweet text are stored.
ID_INDEX = 0
TEXT_INDEX = 1

# Remove existing result files before starting. When false, completed model /
# dataset pairs are resumed from their existing result file.
CLEAR_OUTPUT = False


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse the datasets to process from the command line."""
    parser = argparse.ArgumentParser(
        description="Translate and embed selected text datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        metavar="DATASET",
        help=(
            "Dataset names to process, or 'all' (default) to process every "
            "configured dataset."
        ),
    )
    args = parser.parse_args()

    if "all" in args.datasets and args.datasets != ["all"]:
        parser.error("'all' cannot be combined with individual dataset names")

    unknown = sorted(set(args.datasets) - set(DATASETS))
    if unknown:
        parser.error(
            "unknown dataset(s): "
            + ", ".join(unknown)
            + ". Choose from: "
            + ", ".join(DATASETS)
        )

    return args


def selected_datasets(dataset_names: List[str]) -> dict:
    """Return the configured dataset mapping selected by the CLI."""
    if dataset_names == ["all"]:
        return DATASETS.copy()
    return {name: DATASETS[name] for name in dataset_names}

RESPONSE_SCHEMA = {
    "name": "translation_batch",
    "schema": {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "detected_language": {"type": "string"},
                        "translation": {"type": "string"},
                    },
                    "required": ["id", "detected_language", "translation"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["translations"],
        "additionalProperties": False,
    },
    "strict": True,
}

SYSTEM_PROMPT = """You are a precise translation assistant.
You will receive a numbered list of tweets, each with a unique id.
For each tweet, produce an English translation and detect the source language.
 
Rules:
- Preserve the original meaning and tone as closely as possible.
- Do not alter URLs, @mentions, or hashtags -- copy them through unchanged.
- Replace emojis with a short textual description in parentheses, e.g. (smiling face).
- If a tweet mixes multiple languages, translate all of it to English.
- If a tweet is already in English, return it unchanged (still set detected_language: "en").
- You MUST return exactly one output item per input id, with matching ids.
- Do not skip, merge, or reorder items.
"""

def build_user_prompt(id_to_tweet: dict, extra_context: str = "") -> str:
    lines = [f"id={tid}: {text}" for tid, text in id_to_tweet.items()]
    prompt = "Translate the following tweets:\n\n" + "\n".join(lines)
    if extra_context:
        prompt += f"\n\nNote: {extra_context}"
    return prompt

def preprocess_tweets(
    texts: List[str], text_ids: List[str], client: OpenAI, batch_size: int,
    model_name: str = OPENAI_CHAT_MODEL,
) -> Tuple[List[str], List[str]]:
    """Translate tweet batches with an OpenAI-compatible Chat API."""
    translations: List[str] = []
    translated_ids: List[str] = []
    batch_starts = range(0, len(texts), batch_size)
    for start in tqdm(
        batch_starts,
        desc=f"Translating batches with {model_name}",
        unit="batch",
    ):
        batch = [
            "" if text is None else str(text)
            for text in texts[start:start + batch_size]
        ]
        batch_ids = text_ids[start:start + batch_size]
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt({tid: text for tid, text in zip(batch_ids, batch)})},
            ],
            response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
            temperature=0,
        )
        raw = completion.choices[0].message.content
        assert raw is not None, "Received empty response from translation API"
        data = json.loads(raw)
        
        translations.extend(data["translations"])
        translated_ids.extend(batch_ids)

    # Keep the existing metadata field for checkpoint compatibility.
    return translations, translated_ids

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


def create_translation_client() -> OpenAI:
    """Create the translation client without embedding credentials in code."""
    api_key = os.getenv(OPENAI_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Set {OPENAI_API_KEY_ENV} in the environment before running "
            "the preprocessing pipeline"
        )
    return OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)


def preprocessed_path(dataset_name: str) -> Path:
    """Return the checkpoint path for a translated dataset."""
    return PREPROCESSED_DIR / f"{dataset_name}.npy"


def save_preprocessed_dataset(
    path: Path, tweet_ids: List[str], tweets: List[str],
) -> None:
    """Persist preprocessing output so it can be reused by all embeddings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tweet_ids": np.asarray(tweet_ids, dtype=object),
        "tweets": np.asarray(tweets, dtype=object)
    }
    np.save(path, np.array(payload, dtype=object), allow_pickle=True)


def load_preprocessed_dataset(path: Path) -> Tuple[List[str], List[str]]:
    """Load a preprocessing checkpoint."""
    payload = np.load(path, allow_pickle=True).item()
    return payload["tweet_ids"].tolist(), payload["tweets"].tolist()


def print_translation_samples(
    dataset_name: str, raw_tweets: List[str], translated_tweets: List[str],
) -> None:
    """Print a few raw tweets alongside their translated output."""
    if SAMPLE_COUNT <= 0:
        return

    sample_count = min(SAMPLE_COUNT, len(raw_tweets), len(translated_tweets))
    print(f"  sample raw/translated tweets for '{dataset_name}':")
    if sample_count == 0:
        print("    (dataset is empty)")
        return

    for index in range(sample_count):
        raw = "" if raw_tweets[index] is None else str(raw_tweets[index])
        translated = (
            "" if translated_tweets[index] is None
            else str(translated_tweets[index])
        )
        print(f"    {index + 1}. raw: {raw!r}")
        print(f"       translated: {translated!r}")


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

    for t_idx, tweet in enumerate(
        tqdm(tweets, desc=f"Embedding {model_name}", unit="tweet")
    ):
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
    return EMBEDDED_DIR / model_name / f"{dataset_name}.npy"


def result_has_model(results_path: Path) -> bool:
    """Check whether a dataset/model checkpoint already exists."""
    return results_path.exists()


def save_model_results(
    results_path: Path,
    tweet_ids: List[str],
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
    args = parse_args()
    datasets = selected_datasets(args.datasets)

    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDED_DIR.mkdir(parents=True, exist_ok=True)

    model_names = list(MODEL_IDS.keys())
    dataset_names = list(datasets.keys())

    if CLEAR_OUTPUT:
        for dataset_name in dataset_names:
            for model_name in MODEL_IDS:
                output_path = result_path(dataset_name, model_name)
                if output_path.exists():
                    output_path.unlink()
                    print(f"Removed existing embedding: {output_path}")
        if PREPROCESSED_DIR.exists():
            shutil.rmtree(PREPROCESSED_DIR)
            print(f"Removed existing translated checkpoints: {PREPROCESSED_DIR}")

    print("=" * 72)
    print("Starting embedding pipeline")
    print(f"- device: {DEVICE}")
    print(f"- datasets: {len(dataset_names)}")
    print(f"- models: {len(model_names)}")
    print(f"- embedding directory: {EMBEDDED_DIR}")
    print(f"- preprocessed data: {PREPROCESSED_DIR}")
    print("=" * 72)

    # Preprocess each dataset once, independently of the embedding models.
    translation_client = create_translation_client()
    for d_idx, (dataset_name, dataset_path) in enumerate(
        tqdm(datasets.items(), desc="Preprocessing datasets", unit="dataset")
    ):
        preprocessing_path = preprocessed_path(dataset_name)
        print(
            f"Preprocessing dataset [{d_idx + 1}/{len(dataset_names)}]: "
            f"{dataset_name} ({dataset_path})"
        )
        tweet_ids, raw_tweets = load_dataset(dataset_path)

        if preprocessing_path.exists():
            print(f"  skipping translation; checkpoint exists: {preprocessing_path}")
            _, translated_tweets = load_preprocessed_dataset(preprocessing_path)
            print_translation_samples(dataset_name, raw_tweets, translated_tweets)
            del tweet_ids, raw_tweets, translated_tweets
            gc.collect()
            continue

        tweets, translated_tweet_ids = preprocess_tweets(
            raw_tweets, [str(id) for id in tweet_ids], translation_client, PREPROCESS_BATCH_SIZE,
            OPENAI_CHAT_MODEL,
        )
        save_preprocessed_dataset(
            preprocessing_path, translated_tweet_ids, tweets,
        )
        print(f"  -> saved {preprocessing_path}")
        print_translation_samples(dataset_name, raw_tweets, tweets)
        del tweet_ids, raw_tweets, tweets
        gc.collect()

    # Process one model and one preprocessed dataset at a time. Dataset data
    # is released after its checkpoint is written, and completed model rows
    # are skipped.
    for m_idx, name in enumerate(model_names):
        model_id = MODEL_IDS[name]
        print("\n" + "-" * 72)
        print(f"Model [{m_idx + 1}/{len(model_names)}]: {name} ({model_id})")
        model = SentenceTransformer(model_id, device=DEVICE)
        try:
            for d_idx, (dataset_name, dataset_path) in enumerate(
                tqdm(datasets.items(), desc=f"Datasets for {name}", unit="dataset")
            ):
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