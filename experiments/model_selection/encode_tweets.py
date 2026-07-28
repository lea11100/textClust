"""
Embed tweets from multiple datasets using several sentence-transformer models,
timing each individual encode call, and store the results per dataset.

Expected input format
----------------------
Each dataset record is expected to behave like a sequence where:
    record[0]  ->  the tweet id
    record[1]  ->  the tweet text (str)

Datasets can be either .npy files (loaded with np.load) or .json files
(loaded with json.load) -- see load_dataset().

Output format
-------------
For each dataset, we save a single .npz file (in OUTPUT_DIR) containing:
    - "tweet_ids":  array of shape (n_tweets,), the tweet id for each column
                     of the arrays below (so embeddings/times are identified
                     by id)
    - "embeddings": object array of shape (n_models, n_tweets), each entry is
      a 1D float32 numpy array (the embedding vector for that tweet/model)
    - "times":      float64 array of shape (n_models, n_tweets), seconds
                     taken to encode that single tweet with that model
    - "model_names": array of shape (n_models,), the friendly model names
                     used (same order as MODEL_IDS keys)

Embeddings are matched back to tweets via "tweet_ids": column j in
"embeddings" and "times" corresponds to tweet_ids[j], for every model row.

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
    python embed_and_time.py
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

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

# Models: friendly name -> HuggingFace model id. Only one of these is ever
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

# Index within each record where the tweet id and tweet text are stored.
ID_INDEX = 0
TEXT_INDEX = 1

# Where to write the per-dataset result files.
OUTPUT_DIR = Path("results")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_dataset(dataset_path):
    """
    Load a dataset's tweet ids and tweet texts (no embeddings yet).

    Supports .npy files (loaded with np.load) and .json files (loaded with
    json.load). Either way, each record is expected to behave like a
    sequence where record[ID_INDEX] is the tweet id and record[TEXT_INDEX]
    is the tweet text.
    """
    suffix = Path(dataset_path).suffix.lower()

    if suffix == ".npy":
        stream = SocialEDWrapper(dataset_path, max_samples=None, random_start_seed=None).load()
    elif suffix == ".json":
        stream = StaticJsonWrapper(dataset_path, max_samples=None, random_start_seed=None).load()
    else:
        raise ValueError(f"Unsupported dataset file type: {dataset_path}")

    data = []
    while True:
        point = next(stream)
        if point is None:
            break
        data.append(point)

    tweet_ids = np.array([point.id for point in data])
    tweets = [point.text for point in data]
    return tweet_ids, tweets


def embed_tweets_with_model(tweets, model, model_name):
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


def free_model(model):
    """Release a model from memory/GPU before loading the next one."""
    del model
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_names = list(MODEL_IDS.keys())
    dataset_names = list(DATASETS.keys())

    print("=" * 72)
    print("Starting embedding pipeline")
    print(f"- device: {DEVICE}")
    print(f"- datasets: {len(dataset_names)}")
    print(f"- models: {len(model_names)}")
    print(f"- output directory: {OUTPUT_DIR}")
    print("=" * 72)

    # Load only the lightweight tweet ids/texts for every dataset up front.
    # Embeddings are NOT computed yet -- results arrays are preallocated and
    # filled in one model-row at a time as each model is loaded below.
    datasets = {}
    for name, path in DATASETS.items():
        print(f"Loading dataset [{len(datasets) + 1}/{len(dataset_names)}]: {name} ({path})")
        tweet_ids, tweets = load_dataset(path)
        n_tweets = len(tweets)
        datasets[name] = {
            "tweet_ids": tweet_ids,
            "tweets": tweets,
            "embeddings": np.empty((len(model_names), n_tweets), dtype=object),
            "times": np.zeros((len(model_names), n_tweets), dtype=np.float64),
        }
        print(f"  -> loaded {n_tweets} tweets and {len(tweet_ids)} ids")

    # Process one model at a time so only a single model ever sits in
    # memory/on the GPU. For each model, run it over every dataset, then
    # free it before moving to the next model.
    for m_idx, name in enumerate(model_names):
        model_id = MODEL_IDS[name]
        print("\n" + "-" * 72)
        print(f"Model [{m_idx + 1}/{len(model_names)}]: {name} ({model_id})")
        model = SentenceTransformer(model_id, device=DEVICE)

        for dataset_name, d in datasets.items():
            print(f"  encoding dataset: {dataset_name} ({len(d['tweets'])} tweets)")
            embeddings, times = embed_tweets_with_model(d["tweets"], model, name)
            d["embeddings"][m_idx, :] = embeddings
            d["times"][m_idx, :] = times
            print(f"    -> completed {dataset_name}")

        free_model(model)
        print(f"Finished model {name}")

    # All models processed -- now write out one results file per dataset.
    print("\n" + "=" * 72)
    print("Writing results files")
    for dataset_name, d in datasets.items():
        out_path = OUTPUT_DIR / f"{dataset_name}_results.npz"
        np.savez(
            out_path,
            tweet_ids=d["tweet_ids"],
            embeddings=d["embeddings"],
            times=d["times"],
            model_names=np.array(model_names),
        )
        print(f"saved results to {out_path}")
    print("Embedding pipeline completed")
    print("=" * 72)


def load_embedding(results_npz_path, tweet_id, model_name):
    """
    Convenience lookup: given a saved results file, a tweet id, and a model
    name, return that tweet's embedding (and encode time) for that model.
    """
    r = np.load(results_npz_path, allow_pickle=True)
    tweet_ids = r["tweet_ids"]
    model_names = list(r["model_names"])

    id_matches = np.where(tweet_ids == tweet_id)[0]
    if len(id_matches) == 0:
        raise KeyError(f"tweet_id {tweet_id!r} not found in {results_npz_path}")
    t_idx = id_matches[0]

    if model_name not in model_names:
        raise KeyError(f"model {model_name!r} not found in {results_npz_path}")
    m_idx = model_names.index(model_name)

    embedding = r["embeddings"][m_idx, t_idx]
    encode_time = r["times"][m_idx, t_idx]
    return embedding, encode_time


if __name__ == "__main__":
    main()