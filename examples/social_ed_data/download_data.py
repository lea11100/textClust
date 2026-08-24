from pathlib import Path

import requests

datasets = {
    "Arabic_7" : "Arabic_7.npy",
    "Arabic_Twitter" : "Arabic_Twitter.npy",
    "CrisisLexT6" : "CrisisLexT6.npy",
    "CrisisLexT7" : "CrisisLexT7.npy",
    "CrisisLexT26" : "CrisisLexT26.npy",
    "CrisisMMD" : "CrisisMMD.npy",
    "CrisisNLP" : "CrisisNLP.npy",
    "Event2012" : "Event2012.npy",
    "Event2012_100" : "Event2012_100.npy",
    "Event2018" : "Event2018.npy",
    "Event2018_100" : "Event2018_100.npy",
    "HumAID" : "HumAID.npy",
    "KBP" : "KBP.npy",
    "MAVEN" : "MAVEN.npy",
    "Mix_Data" : "Mix_Data.npy",
}
    
url = "https://github.com/lea11100/twitter_stream_data/blob/main/social_ed_data"
output_dir = Path(__file__).parent.parent / "data" / "raw_data"
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