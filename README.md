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

Run `examples/data/download_data.py` for the JSON datasets and
`examples/data/download_social_ed_data.py` for the SocialED NumPy datasets.
The offline preprocessing pipeline reads from `raw_data`, writes translated
and demojified checkpoints to `translated_no_emojis`, and writes embeddings
to the model-specific directories under `embedded`.
# textClust