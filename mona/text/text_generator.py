class TextGenerator:
    def __init__(self, name):
        self.name = name

    def generate_text(self):
        raise NotImplementedError("this method have to be implemented")

    def get_lexicon(self):
        return set()
