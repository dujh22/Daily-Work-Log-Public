# 贡献指南

感谢关注本仓库。这是个人工作日志仓库,如果你想补充或修正内容:

## 内容归属

- 每日日志 → `logs/<year>/YYYY-MM-DD.md`,从 [templates/daily.md](templates/daily.md) 复制起稿。
- LLM 笔记 → `notes/llm/` 下对应子类(frameworks / models / research),新主题先在 [notes/llm/README.md](notes/llm/README.md) 索引登记。
- 其他内容 → 参考 [README.md](README.md) 目录结构;拿不准就开 Issue 讨论。

## 基本规则

1. **命名**:目录与文件名用英文小写 kebab-case;每日日志 `YYYY-MM-DD.md`;正文中文。
2. **图片**:放在 md 文件旁 `image/<文件名>/` 目录,PNG 格式。
3. **二进制**:不提交 md/png 之外的文件;大文件走云盘归档 + 文中占位说明。
4. **Commit message**:`<type>: <摘要>`,type 见 [CLAUDE.md](CLAUDE.md) Commit 规范。

## 提交流程

Fork 后向 `main` 分支发起 Pull Request,PR 描述里说明改动内容与动机即可。
