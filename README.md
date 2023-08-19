<div align="center">

# Yas-Train
Yet Another Genshin Impact Scanner  
又一个原神圣遗物导出器

</div>

## 介绍
该仓库为 [Yas](https://github.com/wormtql/yas) 的模型训练程序
### 相关资料
- [MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf)
- [CRNN](https://arxiv.org/pdf/1507.05717.pdf)

## 使用
假设你会设置基本的pytorch环境。  
- 训练
```
python main.py train
```
- 验证
```
python onlineval.py xxx.model
```
在`mona/config.py`处可以修改数据集大小、epoch等参数

### 炼丹心得
请移步 [TIPS.md](TIPS.md)

## 反馈
- Issue
- QQ群：801106595
