# Data cleaner for YAS

工作流程：
1. 使用yas-dump，获取{灰度图片，二值化图片，识别结果}
2. 使用一个Python代码来将dump结果输出到Excel文档，这个过程图片会被自动插入
3. Excel里做数据清洗
4. Python代码读取清洗过的数据，生成标注好的数据集

上述提到的两个python代码都在example.ipynb里