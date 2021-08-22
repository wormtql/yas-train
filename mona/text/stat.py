import json
import random


stat_name = [
    "暴击率",
    "暴击伤害",
    "攻击力",
    "治疗加成",
    "生命值",
    "防御力",
    "元素充能效率",
    "元素精通",
    "火元素伤害加成",
    "水元素伤害加成",
    "岩元素伤害加成",
    "风元素伤害加成",
    "冰元素伤害加成",
    "雷元素伤害加成",
    "物理伤害加成",
    "治疗加成"
]

with open("./assets/output.json") as f:
    string = f.read()

data = json.loads(string)


def random_value():
    key = random.choice(list(data.keys()))
    temp = data[key]
    key = random.choice((list(temp.keys())))
    temp = temp[key]

    value = 0
    times = random.randint(1, 6)
    for i in range(times):
        value += random.choice(temp)

    if value < 2:
        ret = str(round(value * 100, 1)) + "%"
    else:
        ret = str(round(value))

    return ret


def random_stat():
    stat = random.choice(stat_name)
    value = random_value()

    return stat + "+" + value
