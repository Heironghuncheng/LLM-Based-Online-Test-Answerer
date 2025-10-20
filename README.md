# OCR + DeepSeek 助手

一个轻量的桌面工具：按住 `Ctrl` 用左键拖拽选区，自动截图 + OCR 文本清理，并通过 DeepSeek 两阶段管线给出结构化答案与解释。专为考试题、说明文本等中英文混排场景优化。

## 核心特性
- 选区截图 + OCR（Tesseract，支持中英混合）。
- 两阶段 AI 管线：预处理（文本清洗/题型识别）→ 正式作答（严格 JSON）。
- 模型推荐与严格门控：默认 `chat`，复杂推理建议 `reasoner`；第二阶段仅在满足硬性条件时启用推理模型。
- 记忆与主题校正：积累背景知识与主题，自动去重；对低频主题动态纠正（<10% 出现率自动剔除）。
- 国际化日志与输出语言：UI 支持中/英；模型输出语言可独立配置。
- 超时自动回退：正式作答超时自动切换快速模型重试。

## 工作流程
1. 屏幕选区 → `mss` 截图 → `pytesseract` 识别。
2. 发送到 DeepSeek：
   - 阶段一（预处理）：清洗 OCR、抽取题干/选项、判断是否题目与题型、给出模型推荐与思考链建议、相关主题与背景知识。
   - 阶段二（正式作答）：根据题型输出严格 JSON（答案、解释、置信度），并附带预处理给出的思考长度提示与背景知识参考。
3. 控制台以 Rich 面板展示各阶段摘要与最终结果。

## 模型选择策略（重要）
- 预处理阶段提示词：
  - 默认推荐 `chat`；当任务明显受益于“结构化多步推理、分解或形式化论证”时推荐 `reasoner`。
  - 对“复杂或高歧义问题”明确警示：不要选择 `chat/fast`，误将复杂问题选为 `chat` 会被视为严重错误。
- 第二阶段硬性门槛（在代码中实现）：
  - 仅当 `recommended_model == 'reasoner'` 且 `suggest_thinking_length >= 128` 时，使用推理模型；否则使用快速模型。
  - 正式作答若超时，会自动回退到快速模型重试。

## 记忆与主题校正
- 背景知识：无长度上限，按最近条目显示；用于第二阶段作为“仅供参考”的知识块。
- 相关主题：用集合去重累计；维护 `topic_counts` 与 `topic_total_count`。
- 动态纠正：当某主题出现频率低于所有样本的 10%，会从 `related_topics` 中移除，降低噪声与偏题风险。

## 安装与环境
- Python：`>= 3.13`（建议使用官方 Python 3.13）。
- 依赖：`pynput`, `mss`, `pillow`, `pytesseract`, `httpx[http2]`, `rich>=14.2.0`。
- Tesseract：需安装并确保中文语言包可用（一般为 `chi_sim`）。

安装依赖（推荐使用 uv，也提供 pip 方案）：

- 使用 uv（推荐，快速且可复用）：
  1) 安装 uv：`pipx install uv`，或直接 `pip install uv`；
  2) 在项目目录执行：
  ```bash
  uv sync
  ```
  3) 运行：
  ```bash
  uv run python main.py
  ```

- 使用 pip（传统方案）：
  1) 创建并激活虚拟环境（Windows）：
  ```bash
  python -m venv .venv
  .\.venv\Scripts\activate
  ```
  2) 升级基础工具并安装依赖：
  ```bash
  python -m pip install -U pip setuptools wheel
  pip install -r requirements.txt
  ```
  如果没有 `requirements.txt`，也可按需安装：
  ```bash
  pip install pynput mss pillow pytesseract "httpx[http2]" "rich>=14.2.0"
  ```
  3) 运行：
  ```bash
  python main.py
  ```

- Tesseract 提示：请提前安装并确保中文包 `chi_sim` 可用；若未在 `PATH`，可在 `config.toml` 配置 `tesseract_cmd` 路径。

## 配置
所有配置从 `config.toml` 读取；未知键会被忽略，缺失项使用默认值。

示例：
```toml
# UI/日志语言与模型输出语言（未设置时 model_output_language 跟随 log_language）
log_language = "zh"
model_output_language = "zh"

# DeepSeek API Key（请替换为你自己的密钥）
deepseek_api_key = "your_api_key_here"

# 阶段模式与重试/超时
analysis_stage_mode = "multi"       # "single" 或 "multi"（默认 multi）
single_stage_model = "deepseek-chat"
formal_answer_timeout_seconds = 60
request_retry_attempts = 3

# 两阶段模型配置
deepseek_fast_model = "deepseek-chat"
deepseek_reasoning_model = "deepseek-reasoner"
deepseek_model = "deepseek-reasoner"

# Tesseract 设置（如需自定义路径）
# tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
# tesseract_lang = "chi_sim+eng"
```

## 使用方法
1. 在项目目录运行：
   ```bash
   python main.py
   ```
2. 操作方式：按住 `Ctrl`，用左键拖拽形成矩形选区（从按下到松开）。
3. 程序流程：
   - 截图 → OCR → 发送到 DeepSeek；
   - 控制台展示预处理摘要（题型、模型推荐、置信度、建议思考链长度、背景知识、相关主题等）；
   - 展示正式作答的严格 JSON（答案/解释/置信度）。

## 目录结构
```
├── config.py          # 严格读取 config.toml，仅允许已知键
├── config.toml        # 配置文件（示例见上）
├── deepseek.py        # DeepSeek 两阶段管线 + 内存与门控
├── i18n.py            # 国际化初始化与语言解析
├── input_handler.py   # 键鼠监听（Ctrl + 左键拖拽选区）
├── locales/           # zh/en 本地化文案
├── main.py            # 程序入口与协同
├── ocr.py             # 选区截图 + Tesseract OCR
├── pyproject.toml     # 项目依赖定义
└── README.md          # 文档（即本文件）
```

## 常见问题与排障
- 没有识别到文本：可能字体太小/背景干扰；尝试放大或换高对比度背景。
- DeepSeek API Key 缺失：在 `config.toml` 中设置 `deepseek_api_key`。
- Tesseract 中文不可用：安装中文语言包（`chi_sim`），并确认 `tesseract_cmd` 正确。
- 预处理解析失败：会直接走推理模型作答（并在第二阶段仍受 128 链长门控影响）。
- 正式作答超时：自动回退到快速模型重试并打印告警面板。
- 多显示器偏移：优先在主显示器测试，或更新显卡驱动；确保系统坐标正确。

## 开发提示
- 语法检查：
  ```bash
  python -m py_compile c:\\Users\\BLESS\\Project\\huawei\\deepseek.py
  ```
- 国际化：`log_language` 控制 UI 文案（`locales/zh.json` 与 `locales/en.json`）；`model_output_language` 控制模型答案语言。
- 模型成本与隐私：文本会发往 DeepSeek API；请勿提交敏感数据，留意调用成本与速率限制。

——
如果你希望调整模型门控阈值（如将 128 改为配置项）、增加日志采样或在面板中展示主题频率分布，我可以继续完善与扩展。