# To-do List

## 注意

1. 时刻记录——防止注意力不集中:时间相当有限，每一分钟都不会重来

   1. 按小时设置目标，并按小时进行目标检查——防止精力分散
   2. 输入一定同时输出——避免无效输入
   3. 如果实在想玩耍，设置一个5分钟沉浸式：5分钟内完成一个最小化的任务，屏蔽一切干扰，5分钟后我有权选择停止
   4. 注意休息——番茄工作法：20-20-15分钟/小时，前两个20分钟后休息20秒，最后15分钟后休息5分钟。
2. 动力来源：

   1. 别担心，做下去，just do it
   2. 比自己优秀的博主那么多，随便找1个都是鸡血满满
3. 5点起床健身上班-10点下班11点休息

## 时间

1. 会议
   1. [ICML](https://icml.cc/Conferences/2026) [OpenReview](https://openreview.net/group?id=ICML.cc/2026/Conference) 回复截止4.7
   2. [NeurIPS](https://neurips.cc/Conferences/2025)、SIGMOD 4月份开放提交
      1. 摘要截止 4.10
      2. 全文截止 4.17
   3. [AAAI](https://aaai.org/conference/aaai/aaai-26/) 7月份
   4. ICLR 9月份
2. 开题：院系开题答辩4.8

## 重要

1. 开题
2. EvolveLRM（详见日志.md）
   1. 跑通整个工程
   2. 主实验结果与可复现性报告
   3. 消融实验结果与可复现性报告
   4. 相关工作进一步调研
   5. 论文第二稿
3. LogicEvolve优化
   1. to do list 推进
   2. to do list 推进
   3. 专利
4. LogicSurvey

### 📑 开题

1. 开题条件

   1. [【腾讯文档】KEG毕业建议](https://docs.qq.com/doc/DQ0FWYlpHaG9KbG9K)
   2. [【腾讯文档】创新成果认定办法-计算机分委员会-最新通用202008.docx](https://docs.qq.com/doc/DQ0pDSXNlRVh2Z2xZ?_bid=1&client=drive_file)
   3. [【腾讯文档】KEG答辩清单](https://docs.qq.com/doc/DQ3lwVk5sUlJQd3lW)
2. 文献综述、选题背景及其意义、研究内容、工作特色及难点、预期成果及可能的创新点、论文工作计划等。
3. 博士开题PPT一般讲20分钟：春季工学博士开题预计将安排在4月8日
4. todo

   1. PPT修改
      1. 现有方法在 X 上失败，因此我们必须 Y
5. 

### ⌛️ EvolveLRM

1. Datamaker改造
   1. 测试LogicEvolve相关数据合成部分功能
2. 工程planner+优化目标实现
3. 工程论文经验抽取
4. 基础实验
5. Trainer修改
6. Evaluator改造
   1. 封装evaluator相关功能，可能需要调整部分命令支持非交互式配置:config非交互式命令支持
   2. LogicEvolve新增更多逻辑推理相关评测
