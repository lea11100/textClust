# textClust

## Dataset layout

All example data is stored below `examples/data`:

```text
examples/data/
├── raw_data/
│   └── <dataset_name>.npy or .json
├── translated_no_emojis/
│   └── <dataset_name>.npy
└── embedded/
	└── <model_name>/<dataset_name>.npy
```

The offline preprocessing pipeline reads from `raw_data`, writes translated
and translated checkpoints to `translated_no_emojis`, and writes embeddings
to the model-specific directories under `embedded`.

## OpenAI translation setup

`src/textClust/preprocessing/offline_preprocessing.py` uses the configured
OpenAI-compatible Responses API to translate tweets. Set the API key outside
the source code before running it:

```bash
export OPENAI_API_KEY='your-key'
```

For HPC jobs, prefer the scheduler or cluster secret store. A practical
alternative is a user-only file outside the repository (for example,
`~/.config/textclust/openai.env` with permissions `600`) that the job script
sources at runtime. Do not commit that file, put the key in `.env` tracked by
Git, or pass it as a command-line argument. `OPENAI_BASE_URL` and
`OPENAI_CHAT_MODEL` may also be set to override the defaults.

The API key included in a chat message or otherwise exposed should be revoked
and replaced before use.
# textClust