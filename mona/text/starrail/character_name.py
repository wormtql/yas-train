from mona.text.text_generator import TextGenerator

# Trailblazer is not included
# https://github.com/Mar-7th/StarRailRes/blob/master/index_new/cn/characters.json
characters_name_starrail = [
    "三月七",
    "丹恒",
    "姬子",
    "瓦尔特",
    "卡芙卡",
    "银狼",
    "阿兰",
    "艾丝妲",
    "黑塔",
    "布洛妮娅",
    "希儿",
    "希露瓦",
    "杰帕德",
    "娜塔莎",
    "佩拉",
    "克拉拉",
    "桑博",
    "虎克",
    "玲可",
    "卢卡",
    "托帕&账账",
    "青雀",
    "停云",
    "罗刹",
    "景元",
    "刃",
    "素裳",
    "驭空",
    "符玄",
    "彦卿",
    "桂乃芬",
    "白露",
    "镜流",
    "丹恒•饮月",
    "雪衣",
    "寒鸦",
    "藿藿",
    "椒丘",
    "飞霄",
    "云璃",
    "灵砂",
    "貊泽",
    "三月七",
    "忘归人",
    "加拉赫",
    "银枝",
    "阮•梅",
    "砂金",
    "真理医生",
    "花火",
    "黑天鹅",
    "黄泉",
    "知更鸟",
    "流萤",
    "米沙",
    "星期日",
    "翡翠",
    "波提欧",
    "乱破",
    "大黑塔",
    "阿格莱雅",
]


class StarrailCharacterEquipGenerator(TextGenerator):
    def __init__(self):
        super(StarrailCharacterEquipGenerator, self).__init__("Starrail Character Equip")

    def generate_text(self):
        return "装备中"

    def get_lexicon(self):
        ret = set()
        for character in characters_name_starrail:
            for char in character:
                ret.add(char)
        for c in "装备中":
            ret.add(c)
        return ret
