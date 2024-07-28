import random

from mona.text.text_generator import TextGenerator


class RandomNumberTextGenerator(TextGenerator):
    def __init__(self):
        super(RandomNumberTextGenerator, self).__init__("Random Number")
        self.lower = 0
        self.upper = 1000000000

    def generate_text(self):
        n = random.randint(self.lower, self.upper)
        return f"{n}"

    def get_lexicon(self):
        ret = set()
        for i in "0123456789":
            ret.add(i)
        return ret
