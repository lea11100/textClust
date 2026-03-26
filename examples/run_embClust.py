from textClust.data.wrapper import StaticJsonWrapper

from textClust.algorithms.presets import get_embClust

if __name__ == "__main__":
    clusterer = get_embClust(
        data_loader=StaticJsonWrapper("examples/data/Trends-T_modified.json"),
        embedding_model_name='sentence-transformers/paraphrase-MiniLM-L3-v2',
        clean_interval=100,
        cluster_interval=1000,
        log_interval=100,
        decay_rate=0.005,
        sigma=1
    )
    clusterer.start()