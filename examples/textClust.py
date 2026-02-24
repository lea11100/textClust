from textClust.core.textClust import textClust
from textClust.data.wrapper import StaticJsonWrapper
from textClust.preprocessing.tfidf import TFIDFPreprocessor

if __name__ == "__main__":
    wrapper = StaticJsonWrapper("examples/data/News-T.json")
    preprocessor = TFIDFPreprocessor()
    clusterer = textClust(wrapper, preprocessor, None, None, None, None, None)
    clusterer.start()