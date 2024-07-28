import random

from mona.text.text_generator import TextGenerator


class WeightedTextGenerator(TextGenerator):
    def __init__(self):
        super(WeightedTextGenerator, self).__init__("Weighted Generator")

        self.entries = []
        self.weights = []

    def generate_text(self):
        f = random.choices(
            population=self.entries,
            weights=self.weights,
            k=1
        )
        return f[0].generate_text()

    def get_lexicon(self):
        ret = set()
        for entry in self.entries:
            words = entry.get_lexicon()
            for word in words:
                ret.add(word)
        return ret

    def add_entry(self, weight, ins):
        self.entries.append(ins)
        self.weights.append(weight)
