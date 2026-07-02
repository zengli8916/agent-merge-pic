# Agent Merge Pic - 直播间产品图片生成 Agent

AI 驱动的电商直播间产品展示图片生成工具。输入参考直播间照片和产品图，自动生成符合要求的直播间场景图，支持多轮迭代优化。

## 功能特性

- **多轮自动迭代**：自动评估生成质量，针对性修正，直到满足要求
- **智能回滚**：检测到质量下降时自动回滚到最佳状态重新生成
- **多产品支持**：同时输入多张产品图片
- **灵活配置**：可选图片比例、迭代轮次、商品位置
- **Web UI**：实时查看生成进度、每轮评分、中间结果
- **二次编辑**：对最终结果通过文字指令进行微调
- **CLI 工具**：支持命令行批量调用

## 快速开始

### 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 设置 API Key

```bash
export GEMINI_API_KEY="your-api-key"
```

### Web UI 启动

```bash
python run_web.py
# 浏览器打开 http://localhost:8000
```

### CLI 使用

```bash
python cli.py \
  -r ./reference.png \
  -p ./product1.jpg \
  -p ./product2.jpg \
  --person "马来西亚年轻女性" \
  --background "保健品货柜" \
  -n "产品图中的瓶子大小是一样大的" \
  --max-rounds 5
```

## CLI 参数

| 参数 | 说明 |
|------|------|
| `-r, --reference` | 参考直播间图片路径（必填） |
| `-p, --product` | 产品图片路径，可多次使用（必填） |
| `--person` | 人物描述（必填） |
| `--background` | 背景描述（必填） |
| `-n, --note` | 补充要求，可多次使用 |
| `--max-rounds` | 最大迭代轮次（默认5） |
| `--restart-after` | 连续失败多少轮后重新生成（默认5） |
| `-o, --output` | 输出目录（默认output） |
| `-v, --verbose` | 详细日志 |

## Web UI 功能

- 上传参考图和多张产品图
- 选择背景预设或自定义输入
- 选择商品位置（桌前展示/主播手持/参考原图/自定义）
- 选择图片比例（9:16/3:4/1:1/4:3/16:9/自动）
- 实时显示每轮生成结果和评分
- 点击图片放大查看
- 查看每轮生成提示词
- 下载任意轮次图片
- 对最终结果进行文字编辑微调

## 技术架构

- **模型**：Gemini 3.1 Flash Image（通过中转 API 调用）
- **后端**：FastAPI + uvicorn
- **前端**：原生 HTML + JS（无框架依赖）
- **实时通信**：Server-Sent Events (SSE)
- **评估维度**：姿势真实感、产品准确度、产品大小、需求匹配度、物理合理性

## 项目结构

```
agent_merge_pic/
├── agent.py              # 主编排器：迭代循环 + 回滚逻辑
├── generator.py          # Gemini API 调用 + 多轮对话管理
├── evaluator.py          # 图片质量评估
├── fix_strategy.py       # 修正指令生成
├── prompt_builder.py     # 初始 prompt 构建
├── config.py             # 配置 + prompt 模板
├── models.py             # 数据模型
├── cli.py                # CLI 入口
├── consultant.py         # 设计咨询模块
├── run_web.py            # Web 启动入口
├── web/
│   ├── __init__.py       # 进度事件定义
│   ├── app.py            # FastAPI 应用
│   └── static/
│       └── index.html    # 前端页面
└── requirements.txt
```

## License

MIT
