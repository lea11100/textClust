from textClust.data.wrapper import StaticJsonWrapper

from textClust.algorithms.presets import get_fastClust, get_fastLogClust

if __name__ == "__main__":   
    clusterer = get_fastClust(
        data_loader=StaticJsonWrapper("examples/data/Trends-T_modified.json"),
        clean_interval=100,
        cluster_interval=1000,
        log_interval=100,
        decay_rate=0.005,
        sigma=1
    )
    clusterer.start()