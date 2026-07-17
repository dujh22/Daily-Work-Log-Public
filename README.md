# Daily-Work-Log-Public

> A public personal work log: daily logs, LLM study notes, work records, awards, and sports — written in Chinese, organized in English kebab-case directories.

本仓库为个人公开工作日志存储库，定期同步记录工作内容、任务进展与成果沉淀。既用于自我复盘总结、梳理工作脉络，也方便他人直观了解项目推进情况与工作动态，实现透明化的工作分享与交流。

## 目录结构

```
.
├── logs/          # 工作日志
│   ├── README.md          # 日志说明与年份索引
│   ├── 2025/README.md     # 2025 年度总结
│   └── 2026/              # 2026:年度摘要 README.md + 每日日志 YYYY-MM-DD.md
├── notes/         # 学习与研究笔记
│   └── llm/               # 大模型方向
│       ├── frameworks/    # 框架:verl / rlhf / sglang
│       ├── models/        # 模型:deepseek / gpt / qwen
│       └── research/      # 研究:索引 README + 笔记 + 灵感收集 inbox.md
├── works/         # 工作事务
│   └── social/            # 社会工作(组织生活等;大文件见 materials/README.md)
├── awards/        # 荣誉与奖项(README.md 为总表)
├── sports/        # 运动记录(README.md 为总览,tennis/swimming 专题)
└── templates/     # 模板(daily.md 为每日日志模板)
```

## 阅读指南

- **看今天做了什么**:`logs/2026/` 下按日期找 `YYYY-MM-DD.md`;快速浏览用年度摘要 `logs/2026/README.md`。
- **看学习笔记**:从 [notes/llm/README.md](notes/llm/README.md) 索引进入。
- **看获奖情况**:[awards/README.md](awards/README.md) 是奖学金/荣誉总表。

## 仓库约定

- 目录与文件名使用英文小写 kebab-case;文档标题与正文为中文。每日日志命名 `YYYY-MM-DD.md`。
- 仓库只提交 `.md` 与 `.png`;其他二进制文件(pptx/pdf/xlsx 等)存放云盘归档,文中留占位说明。
- 图片存放于所在 md 文件旁的 `image/<文件名>/` 目录(兼容 VSCode 粘贴图片插件)。
- 详细规则见 [CONTRIBUTING.md](CONTRIBUTING.md);AI 协作约定见 [CLAUDE.md](CLAUDE.md)。

> ⚠️ 2026-07-17 仓库进行过一次全面重构(目录重命名 + git 历史重写),旧链接与旧克隆已失效,新旧路径对照见 [MIGRATION.md](MIGRATION.md)。
