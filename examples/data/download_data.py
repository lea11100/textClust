from pathlib import Path

import requests

datasets = {
    "Semantic10-T": "Semantic10-T.json",
    "Semantic40-T": "Semantic40-T.json",
    "Semantic5-T": "Semantic5-T.json",
    "Semantic5-T_sorted": "Semantic5-T_sorted.json",
    "NT": "NT.json",
    "NTS": "NTS.json",
    "NTSO-T": "NTSO-T.json",
    "SO-T": "SO-T.json",
    "News-T": "News-T.json",
    "Trends-T": "Trends-T.json",
    "Trends-T_modified": "Trends-T_modified.json",
    "Tweets-T": "Tweets-T.json",
}
    
url = "https://github.com/lea11100/twitter_stream_data/blob/main/data"
output_dir = Path(__file__).parent / "raw_data"
output_dir.mkdir(parents=True, exist_ok=True)

for d in datasets:
    file_name = datasets[d]
    file_url = f"{url}/{file_name}?raw=true"
    print(f"Downloading {file_name} from {file_url}...")
    
    response = requests.get(file_url)
    response.raise_for_status()

    with open(output_dir / file_name, "wb") as f:
        f.write(response.content)

    print(f"{file_name} downloaded.")