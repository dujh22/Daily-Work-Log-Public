# Deepseek


## 核心版本

### r1

1. GRPO：[DeepSeekMath： Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300)
   1. DeepSeekMath‑Base 以 DeepSeek‑Coder‑Base‑v1.5 7B（Guo等人，2024年）为基础进行初始化，因为我们发现相较于通用大型语言模型，从代码训练模型起步是更优选择。
   2. 在预训练完成后，我们采用思维链（Wei等人，2022年）、程序思维（Chen等人，2022年；Gao等人，2023年）和工具集成推理（Gou等人，2023年）数据对DeepSeekMath‑Base进行数学指令调优。最终得到的模型DeepSeekMath‑Instruct 7B击败了所有7B规模的同类模型，并与700亿规模的开源指令调优模型性能相当。
