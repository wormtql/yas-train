from mona.text.text_generator import TextGenerator
import random

# Taveller is not included
characters_name_genshin = [
    "珐露珊",
    "流浪者",
    "纳西妲",
    "莱依拉",
    "赛诺",
    "坎蒂丝",
    "妮露",
    "柯莱",
    "多莉",
    "提纳里",
    "久岐忍",
    "鹿野院平藏",
    "夜兰",
    "瑶瑶",
    "神里绫人",
    "云堇",
    "八重神子",
    "申鹤",
    "荒泷一斗",
    "五郎",
    "托马",
    "埃洛伊",
    "珊瑚宫心海",
    "雷电将军",
    "九条裟罗",
    "宵宫",
    "早柚",
    "神里绫华",
    "枫原万叶",
    "优菈",
    "烟绯",
    "罗莎莉亚",
    "胡桃",
    "魈",
    "甘雨",
    "阿贝多",
    "钟离",
    "辛焱",
    "达达利亚",
    "迪奥娜",
    "可莉",
    "温迪",
    "刻晴",
    "莫娜",
    "七七",
    "迪卢克",
    "琴",
    "砂糖",
    "重云",
    "诺艾尔",
    "班尼特",
    "菲谢尔",
    "凝光",
    "行秋",
    "北斗",
    "香菱",
    "雷泽",
    "芭芭拉",
    "丽莎",
    "凯亚",
    "安柏",
    "白术",
    "卡维",
    "瑶瑶",
    "艾尔海森",
    "迪希雅",
    "米卡",
    "琳妮特",
    "林尼",
    "菲米尼",
    "芙宁娜",
    "那维莱特",
    "夏沃蕾",
    "娜维娅",
    "嘉明",
    "闲云",
    "千织",
    "阿蕾奇诺",
    "夏洛蒂",
    "莱欧斯利",
]


class GenshinCharacterEquipTextGenerator(TextGenerator):
    def __init__(self):
        super(GenshinCharacterEquipTextGenerator, self).__init__("Genshin Equip Name")

    def generate_text(self):
        return random.choice(characters_name_genshin) + "已装备"

    def get_lexicon(self):
        ret = set()
        for name in characters_name_genshin:
            for char in name:
                ret.add(char)

        for char in "已装备":
            ret.add(char)

        return ret
