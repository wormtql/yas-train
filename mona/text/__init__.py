from .artifact_name import artifact_name
from .relic_name import relic_name
from .characters import characters_name_genshin, characters_name_starrail
from ..config import config


lexicon = set({})
if config["model_type"] == "Genshin":
    from .stat_genshin import stat_name
    for name in artifact_name:
        for char in name:
            lexicon.add(char)

    for name in stat_name:
        for char in name:
            lexicon.add(char)

    for name in characters_name_genshin:
        for char in name:
            lexicon.add(char)

    numbers = " '0123456789.+%,/已装备圣遗物"

elif config["model_type"] == "StarRail":
    from .stat_starrail import stat_name
    for name in relic_name:
        for char in name:
            lexicon.add(char)

    for name in stat_name:
        for char in name:
            lexicon.add(char)

    for name in characters_name_starrail:
        for char in name:
            lexicon.add(char)

    numbers = " '0123456789.+%,/已装备遗器"

for char in numbers:
    lexicon.add(char)
lexicon = sorted(list(lexicon))

index_to_word = {
    0: "-"
}
word_to_index = {
    "-": 0
}
for index, word in enumerate(lexicon):
    index_to_word[index + 1] = word
    word_to_index[word] = index + 1

print(f"lexicon size: {len(word_to_index)}")
