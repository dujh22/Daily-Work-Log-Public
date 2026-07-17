# 迁移说明(2026-07-17)

本仓库于 2026-07-17 进行了一次全面重构:

1. **目录重命名**:顶层结构改为英文 kebab-case(`logs` / `notes` / `works` / `awards` / `sports` / `templates`),消除多层嵌套。
2. **大文件外移**:pptx/docx/xlsx/pdf 等二进制文件移出仓库,存放于云盘归档;并用 `git filter-repo` 从历史中清除了对应 blob。
3. **历史重写**:因第 2 条,git 历史已重写并 force push。**旧克隆不能 pull,请重新 clone**;GitHub 上指向旧路径或旧 commit 的深链已失效。

## 新旧路径对照表

| 旧路径 | 新路径 |
| --- | --- |
| `Years/2025.md` | `logs/2025/README.md` |
| `Years/2026.md` | `logs/2026/README.md` |
| `Years/Days/HIs/YYYY-MM-DD.md`(63 个) | `logs/2026/YYYY-MM-DD.md`(文件名不变) |
| `Years/Days/HIs/2026-05-23.xlsx` | 云盘归档 `logs/2026/`;仓库内留桩 `logs/2026/2026-05-23.md` |
| `Years/Days/模版.md` | `templates/daily.md` |
| `NoteBooks/LLM/框架/RL/VeRL.md` | `notes/llm/frameworks/verl.md` |
| `NoteBooks/LLM/框架/RLHF/RLHF.md` | `notes/llm/frameworks/rlhf.md` |
| `NoteBooks/LLM/框架/部署/SGLang.md` | `notes/llm/frameworks/sglang.md` |
| `NoteBooks/LLM/框架/SFT/`(空目录) | 取消;在 `notes/llm/README.md` 中列为计划主题 |
| `NoteBooks/LLM/模型/{Deepseek,GPT,Qwen}.md` | `notes/llm/models/{deepseek,gpt,qwen}.md` |
| `NoteBooks/LLM/研究/临时记录.md` | `notes/llm/research/inbox.md` |
| `NoteBooks/LLM/研究/评估/智能体/r^2-Bench.md` | `notes/llm/research/r2-bench.md` |
| `NoteBooks/LLM/研究/**/readme.md`(5 个单行桩) | 删除,合并为 `notes/llm/research/README.md` |
| `NoteBooks/.../r^2-Bench/r^2-Bench_{zh,en}.pdf` | 云盘归档 `notes/r2-bench/` |
| `Works/Social_Works/20260130-一月德育组织生活.md` | `works/social/2026-01-30-org-life.md` |
| `Works/Social_Works/相关材料/*.{pptx,docx}` | 云盘归档 `works/social/materials/`;仓库内见 `works/social/materials/README.md` |
| `Rewards/ALL.md` | `awards/README.md` |
| `Rewards/{2025,2026}.md` | `awards/{2025,2026}.md` |
| `Rewards/甲级团支部.md` | `awards/outstanding-league-branch.md` |
| `Rewards/三好学生与优秀学生干部.md` | `awards/merit-student-and-cadre.md` |
| `Rewards/先进基层组织.md` | `awards/advanced-grassroots-organization.md` |
| `Rewards/优秀党支部书记.md` | `awards/outstanding-party-branch-secretary.md` |
| `Rewards/优秀共产党员.md` | `awards/outstanding-party-member.md` |
| `Rewards/优秀共青团员.md` | `awards/outstanding-league-member.md` |
| `Others/Sports/LOG.md` | `sports/README.md` |
| `Others/Sports/网球.md` | `sports/tennis.md` |
| `Others/Sports/游泳.md` | `sports/swimming.md` |
| `Others/Sports/image/LOG/*.png`(未被引用) | 云盘归档 `orphans/` |
| 各 md 旁的 `image/<旧名>/` | 随 md 改名同步更名(约定不变) |
