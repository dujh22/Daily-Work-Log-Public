# CLAUDE.md

AI 协作约定。本仓库是纯 Markdown 的个人公开工作日志,无构建、无测试、无依赖。

## 目录职责

| 目录 | 内容 | 说明 |
| --- | --- | --- |
| `logs/<year>/` | 每日日志 `YYYY-MM-DD.md` | `README.md` 为年度摘要(人工维护的视图) |
| `notes/llm/` | LLM 笔记 | 三个子类:frameworks / models / research |
| `works/` | 工作事务 | 按领域分子目录(如 social) |
| `awards/` | 荣誉奖项 | `README.md` 为总表 |
| `sports/` | 运动记录 | `README.md` 为总览 |
| `templates/` | 模板 | 新日志从 `templates/daily.md` 复制 |

目录深度不超过 3 层(image 子目录除外)。新建目录前先确认现有目录放不下。

## 命名规则

- 目录与文件名:英文小写 kebab-case(如 `r2-bench.md`),禁止空格、大写、`^`、全角标点。
- 每日日志:`YYYY-MM-DD.md`,放在 `logs/<year>/`。
- 文档 H1 标题与正文用中文。
- 索引文件统一大写 `README.md`(GitHub 自动渲染),不用小写 `readme.md`。

## 图片约定

- 图片放在所在 md 文件旁:`image/<md文件名去后缀>/<毫秒时间戳>.png`(VSCode 粘贴图片插件自动生成此结构)。
- 引用格式:`![<时间戳>](image/<文件名>/<时间戳>.png)`。
- **重命名 md 文件时**,必须同步重命名对应 `image/<旧名>/` 目录并更新文内所有引用。

## 二进制规则

仓库只提交 `.md` 和 `.png`。其他文件(pptx/docx/xlsx/pdf 等)一律移入云盘归档(仓库外),并在引用处留一行占位说明指向归档位置(参考 `works/social/materials/README.md` 的写法)。

## 内容去重约定

- 每日日志 `logs/<year>/YYYY-MM-DD.md` 是事实源。
- `logs/<year>/README.md`(年度摘要)和 `sports/README.md`(运动总览)是人工整理的摘要视图,可以概括但不应是某个事实的唯一记录处。

## Commit 规范

格式:`<type>: <中文或英文摘要>`,type 取 `log`(日常日志)/ `docs`(文档索引)/ `fix`(修链接等)/ `chore`(清理维护)/ `refactor`(结构调整)。例:`log: 2026-07-17 日志`。不要使用纯时间戳作为 commit message。

## 修改检查

改动涉及移动/重命名文件时,提交前检查:(1) 相对链接是否仍可达;(2) image 目录与引用是否同步更新;(3) `git ls-files` 无 md/png 之外的新文件。
