# textClust

textClust is a Python implementation of a stream-based text clustering
framework. It processes text observations as they arrive, maintains evolving
micro-clusters, periodically merges and fades clusters, and supports either
token/TF-IDF representations or dense sentence embeddings. The repository
contains online and offline clustering components, preprocessing utilities,
stream loaders, history tracking, and runnable examples.

## Installation

The project requires Python 3.11 or newer.

```bash
git clone https://github.com/lea11100/textClust.git
cd textClust
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For the optional offline translation and embedding pipeline, install its
extra dependencies:

```bash
python -m pip install -e ".[offline-preprocessing]"
```

This extra adds the OpenAI client, emoji utilities, and progress bars. Core
sentence-transformer dependencies are already included in the main package.

## Toy example

The following example creates an in-memory text stream and runs textClust with
TF-IDF preprocessing. It does not require a dataset file or a downloaded
embedding model.

```python
from textClust.algorithms.presets import get_textClust
from textClust.data.point import RawPoint
from textClust.data.stream import Stream


class ToyStream(Stream):
    def load(self):
        texts = [
            "football team wins the match",
            "the team celebrates the football victory",
            "new python release improves package tools",
            "developers publish a new python library",
        ]
        for index, text in enumerate(texts):
            yield RawPoint(text=text, text_id=str(index), timestamp=index)
        yield None


clusterer = get_textClust(
    data_loader=ToyStream(),
    clean_interval=100,
    cluster_interval=10,
    log_interval=100,
    decay_rate=0.01,
    sigma=1,
)
clusterer.start()
print("Assigned clusters:", clusterer.history_manager.get_final_assignments())
```

Run it with:

```bash
python toy_example.py
```

The preset uses `TFIDFPreprocessor`, stopword removal, stemming, and fixed
interval time management. `nltk` may need its English stopword resource before
the first run:

```python
import nltk
nltk.download("stopwords")
```

More complete examples are in [`examples/`](examples/).

## Preprocessing

### Streaming TF-IDF preprocessing

`TFIDFPreprocessor` normalizes text and converts each observation into a
term-frequency dictionary:

```python
from textClust.preprocessing.preprocessor import TFIDFPreprocessor

preprocessor = TFIDFPreprocessor(
    stopword_removal=True,
    stemming=True,
    punctuation=True,
)
```

### Streaming embedding preprocessing

`EmbeddingPreprocessor` encodes text with a local
`sentence_transformers.SentenceTransformer` model:

```python
from sentence_transformers import SentenceTransformer
from textClust.preprocessing.preprocessor import EmbeddingPreprocessor

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
preprocessor = EmbeddingPreprocessor(model, normalize=True)
```

The `get_embClust` preset performs this setup for you:

```python
from textClust.algorithms.presets import get_embClust
from textClust.data.wrapper import StaticJsonWrapper

clusterer = get_embClust(
    data_loader=StaticJsonWrapper(
        "examples/data/raw_data/Semantic10-T.json"
    ),
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
)
clusterer.start()
```

## OpenAI-backed offline preprocessing

The script
[`src/textClust/preprocessing/offline_preprocessing.py`](src/textClust/preprocessing/offline_preprocessing.py)
prepares datasets for embedding:

1. loads tweet IDs and text from configured `.npy` or `.json` files;
2. converts emojis to readable text;
3. translates non-English text to English through an OpenAI-compatible
   Responses API;
4. writes a translation checkpoint once per dataset;
5. encodes the translated text with configured Hugging Face
   sentence-transformer models;
6. writes model-specific embedding checkpoints that can be reused by
   `SimulatedEmbeddingPreprocessor`.

Set the API key outside the source code:

```bash
export OPENAI_API_KEY="your-key"
```

The default base URL and model are configured for the Paderborn AI gateway,
but they can be overridden:

```bash
export OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"
export OPENAI_CHAT_MODEL="your-chat-model"
```

Do not commit credentials or pass API keys as command-line arguments. For HPC
jobs, use the scheduler or cluster secret store. A user-only environment file
outside the repository is another option:

```bash
chmod 600 ~/.config/textclust/openai.env
source ~/.config/textclust/openai.env
```

Run preprocessing for one or more configured datasets:

```bash
python src/textClust/preprocessing/offline_preprocessing.py \
  --datasets News-T Tweets-T
```

Use `--datasets all` (the default) to process every configured dataset. The
script has a `CUDA` device setting at the top of the file; change
`DEVICE = "cuda"` to `DEVICE = "cpu"` when CUDA is unavailable.

### Dataset layout

Raw and preprocessed example data can be downloaded from
[Sciebo](https://uni-paderborn.sciebo.de/s/LkxwYPFtiRkTBC6). Place downloaded
datasets in `examples/data/raw_data` or in the corresponding preprocessed
directories described below.

The pipeline uses this layout below `examples/data`:

```text
examples/data/
├── raw_data/
│   └── <dataset_name>.npy or .json
├── translated_no_emojis/
│   └── <dataset_name>.npy
└── embedded/
    └── <model_name>/<dataset_name>.npy
```

For JSON input, each record should contain `tweetID`, `tweets`, and
optionally `ts`/`timestamp`. `.npy` input is handled by `SocialEDWrapper`.

### Using translated and embedded checkpoints

This package provides helper preprocessors that utilize precomputed translated or embedded tweets. The main function is to speed-up test runs or hyperparameter optimization. The inteded use is not for real-world applications.

To use a translation checkpoint with streaming TF-IDF clustering:

```python
from textClust.preprocessing.preprocessor import (
    TFIDFPreprocessorWithTranslation,
)

preprocessor = TFIDFPreprocessorWithTranslation(
    dataset="News-T",
    folder="examples/data/translated_no_emojis",
)
```

To replay precomputed embeddings instead of encoding each tweet during runtime:

```python
from textClust.preprocessing.preprocessor import (
    SimulatedEmbeddingPreprocessor,
)

preprocessor = SimulatedEmbeddingPreprocessor(
    model="sentence_transformers_all_MiniLM_L6_v2",
    dataset="News-T",
    folder="examples/data/embedded",
)
```

In both cases, pass the preprocessor to `textClust` as part of a custom
pipeline and keep `RawPoint.text_id` unchanged. The offline script's model
folder name and dataset name must match the `model` and `dataset` arguments
used for the preprocessor.