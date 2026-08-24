from textClust.data.wrapper import StaticJsonWrapper

from textClust.algorithms.presets import get_textClust

if __name__ == "__main__":
    clusterer = get_textClust(
        data_loader=StaticJsonWrapper("examples/data/raw_data/NTSO-T.json", max_samples=2_000),
        clean_interval=100,
        cluster_interval=1000,
        log_interval=100,
        decay_rate=0.005,
        sigma=1
    )
    clusterer.start()