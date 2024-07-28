class Lexicon:
    def __init__(self, text_generator):
        self.text_generator = text_generator

        words = list(text_generator.get_lexicon())
        words = sorted(words)

        self.index_to_word = {
            0: "-"
        }
        self.word_to_index = {
            "-": 0
        }
        for index, word in enumerate(words):
            self.index_to_word[index + 1] = word
            self.word_to_index[word] = index + 1

    def lexicon_size(self):
        return len(self.index_to_word)

    def generate_text(self):
        return self.text_generator.generate_text()
