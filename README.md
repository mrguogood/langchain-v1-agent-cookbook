# 🚀 LangChain V1.x Agent 实战教程

> 从零到精通 LangChain V1.x + LangGraph，掌握智能体（Agent）开发的完整链路！

![Python](https://img.shields.io/badge/python-3.13+-blue)
![LangChain](https://img.shields.io/badge/langchain-1.x-green)
![LangGraph](https://img.shields.io/badge/langgraph-powered-orange)
![PostgreSQL](https://img.shields.io/badge/postgresql-required-blue)
![Docker](https://img.shields.io/badge/docker-optional-blue)
![Stars](https://img.shields.io/badge/stars-⭐-yellow)

---

## 📖 项目介绍

本教程带你系统性掌握 **LangChain V1.x** 生态，从基础概念到生产级 Agent 应用，覆盖智能体开发的核心技术栈：

**基础入门 → Prompt工程 → 流式输出 → 短期记忆 → 长期记忆 → 人机协作(HITL) → RAG检索 → MCP协议 → 可观测性 → 向量数据库 → API服务**


每个章节都是独立可运行的实战项目，附带详细的中文注释和说明文档，让你不仅能跑通代码，更能理解**为什么这么设计**。

---

## 🎯 你将学到什么

| 章节 | 核心能力 | 关键技术 |
|:---:|:---|:---|
| [01_Quickstart](#01quickstart) | Agent 基础 | LLM集成、Tools、结构化输出 |
| [02_PromptTemplate](#02prompttemplate) | Prompt 工程 | 模板文件、变量渲染、动态Prompt |
| [03_StreamOutput](#03streamoutput) | 流式输出 | invoke/stream/batch 三种模式 |
| [04_ShortTermMemory](#04shorttermmemory) | 短期记忆 | InMemory/PostgresSaver、消息修剪与摘要 |
| [05_LongTermMemory](#05longtermmemory) | 长期记忆 | PostgresStore、跨会话持久化 |
| [06_HumanInTheLoop](#06humanintheloop) | 人机协作 | HITL中间件、approve/edit/reject |
| [07_RAG](#07rag) | 检索增强 | 2-Step RAG、Agentic RAG、Chroma |
| [08_MCP](#08mcp) | MCP协议 | Server/Client、工具统一接入 |
| [09_ObservabilityAndEvaluation](#09observabilityandevaluation) | 可观测性 | Langfuse、Tracing、Prompt管理 |
| [10_RagWithMilvus](#10ragwithmilvus) | 向量数据库 | Milvus、全文/语义/混合搜索 |
| [11_AgentAPIServer](#11agentapiserver) | API服务 | FastAPI、Gradio、多会话管理 |

---

## 🏗️ 项目结构

```markdown
langchain-v1-agent-cookbook/
├── 01_Quickstart/                    # 🚀 Agent 快速入门
│   └── 从零开始搭建第一个可运行的 Agent
├── 02_PromptTemplate/                # 📝 Prompt 模板工程
│   └── 结构化 Prompt 设计与复用最佳实践
├── 03_StreamOutput/                  # 📡 流式输出实践
│   └── 实时 Token 流式推送与前端对接
├── 04_ShortTermMemory/               # 💾 短期记忆（PostgreSQL）
│   └── 基于 PostgreSQL 的对话上下文管理
├── 05_LongTermMemory/                # 🧠 长期记忆（跨会话）
│   └── 跨会话持久化记忆存储与召回
├── 06_HumanInTheLoop/                # 👥 人机协作（HITL）
│   └── 人工审核、中断与恢复机制
├── 07_RAG/                           # 🔍 检索增强生成（Chroma）
│   └── 基于 Chroma 的文档检索与生成
├── 08_MCP/                           # 🔌 MCP 协议集成
│   └── Model Context Protocol 工具链对接
├── 09_ObservabilityAndEvaluation/    # 📊 可观测性（Langfuse）
│   └── 全链路追踪、评估与指标监控
├── 10_RagWithMilvus/                 # 🗄️ 向量数据库（Milvus）
│   └── 大规模向量检索与 Milvus 集成
├── 11_AgentAPIServer/                # 🌐 Agent API 服务
│   └── 生产级 Agent 服务化部署方案
├── pyproject.toml                    # 项目依赖与配置
└── README.md                         # 项目说明文档
```

---

## ⚡ 快速开始

### 环境要求

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) 包管理器（推荐）
- Docker（部分章节需要）
- PostgreSQL（部分章节需要）

### 安装依赖

```python
使用 uv（推荐）
uv sync
```

### 运行第一个 Agent

```python
进入快速入门章节
cd 01_Quickstart

运行 Agent
python agent.py
```

---

## 📚 章节详解

### 01_Quickstart

**🎯 目标**：快速上手 LangChain V1.x，理解 Agent 的核心组件

**核心特性**：
- 🤖 四种 LLM 后端（OpenAI / OneAPI / qwen / Ollama）
- 🔧 自定义工具（@tool 装饰器）
- 📦 结构化输出（ToolStrategy / ProviderStrategy）
- 🎛️ 中间件机制
- 📍 多模型动态切换

**运行方式**：

```python
cd 01_Quickstart 
uv run python agent.py
```

---

### 02_PromptTemplate

**🎯 目标**：掌握 Prompt 工程，实现提示词与代码解耦

**核心特性**：
- 📝 Markdown 格式的 Prompt 模板
- 🔄 变量动态渲染
- 📂 文件化管理，便于版本控制
- 👨‍💻 支持非技术人员修改 Prompt

**运行方式**：

```python
cd 02_PromptTemplate 
uv run python agent.py
```

---

### 03_StreamOutput

**🎯 目标**：掌握 Agent 的三种调用方式，实现流式输出

**核心特性**：
- 📥 `invoke()`：同步调用，一次性返回完整响应
- 📤 `stream()`：流式输出，支持 messages/updates/custom 三种模式
- 📦 `batch()`：批量处理多个独立请求
- ✨ 自定义流数据推送

**运行方式**：

```python
cd 03_StreamOutput 
uv run python agent_invoke.py # 同步调用 
uv run python agent_stream.py # 流式输出 
uv run agent_batch.py # 批量处理
```

---

### 04_ShortTermMemory

**🎯 目标**：实现 Agent 的短期记忆，支持多轮对话

**核心特性**：
- 💾 InMemorySaver：内存存储（开发调试）
- 🗄️ PostgresSaver：数据库持久化（生产环境）
- ✂️ TrimMessages：消息修剪策略
- 📝 SummarizationMiddleware：消息摘要中间件
- 🔧 自定义中间件开发

**运行方式**：
```python
启动 PostgreSQL（Docker）
cd 04_ShortTermMemory/postgresql 
docker-compose up -d

返回项目根目录，运行示例
cd ../../ 
uv run python 04_ShortTermMemory/agent_InMemorySaver.py 
uv run python 04_ShortTermMemory/agent_PostgresSaver.py
```

---

### 05_LongTermMemory

**🎯 目标**：实现跨会话的长期记忆，记住用户偏好

**核心特性**：
- 🧠 PostgresStore：键值存储
- 📝 长期记忆的写入与读取
- 👤 基于用户ID的记忆隔离
- 🔄 跨会话、跨线程复用

**运行方式**：
```python
cd 05_LongTermMemory
uv run python agent_PostgresStore.py
```

---

### 06_HumanInTheLoop

**🎯 目标**：实现人机协作，让 Agent 在敏感操作前暂停审核

**核心特性**：
- ✅ **Approve**：批准执行
- ✏️ **Edit**：编辑参数后执行
- ❌ **Reject**：拒绝执行
- 🔄 多轮审核支持
- 🛡️ 安全可控的工具调用

**运行方式**：
```python
cd 06_HumanInTheLoop
uv run python agent_invoke_hitl.py 
uv run python agent_stream_hitl.py
```

---

### 07_RAG

**🎯 目标**：构建检索增强生成系统，让 Agent 拥有领域知识

**核心特性**：
- 📄 PDF 文档解析
- ✂️ 文档切分与 Embedding
- 🏪 Chroma 向量数据库
- 🔍 2-Step RAG：固定流程
- 🤖 Agentic RAG：灵活推理
- 💡 Hybrid RAG：质量校验

**运行方式**：
```python
uv run python 07_RAG/1_create_index.py # 创建索引 
uv run python 07_RAG/2_2step_rag.py # 2-Step RAG 
uv run python 07_RAG/3_agentic_rag.py # Agentic RAG 
uv run python 07_RAG/4_agent_rag.py # 完整 Agent RAG
```

---

### 08_MCP

**🎯 目标**：集成 Model Context Protocol，统一工具接入

**核心特性**：
- 🔌 自定义 MCP Server
- 🔗 MultiServerMCPClient 多服务端连接
- 🔄 stdio/SSE/HTTP 传输协议
- 🛠️ 工具、资源、预设提示词统一管理

**运行方式**：
```python
uv run python 08_MCP/rag_mcp_server.py # 启动 MCP Server 
uv run python 08_MCP/mcp_start.py # 测试 MCP 连接 
uv run python 08_MCP/agent_rag.py # Agent 使用 MCP 工具
```

---

### 09_ObservabilityAndEvaluation

**🎯 目标**：构建可观测的 LLM 应用，实现调试与评估闭环

**核心特性**：

- 🔍 Traces：全链路追踪
- 📊 Observations：分层观测
- 💬 Sessions：会话管理
- 📝 Prompt Management：版本控制
- 🧪 Evaluation：LLM-as-a-Judge 评估

**运行方式**：
```python
启动 Langfuse（Docker）
git clone https://github.com/langfuse/langfuse.git cd langfuse && docker-compose up -d
运行示例
uv run python 09_ObservabilityAndEvaluation/agent_rag.py
```

---

### 10_RagWithMilvus

**🎯 目标**：使用 Milvus 构建企业级向量检索系统

**核心特性**：
- 🏗️ Milvus 数据库管理
- 📐 Schema 定义与索引配置
- 🔍 语义搜索、全文搜索、混合搜索
- 🎯 多条件过滤与排序
- 🔌 MCP Server 封装

**运行方式**：
```python
启动 Milvus（Docker）
cd 10_RagWithMilvus/docker_files/milvus docker-compose up -d
运行 Milvus 测试
cd ../../milvus 
uv run python 01_create_database.py 
uv run python 02_create_collection.py 
uv run python 03_insert_data.py 
uv run python 04_basic_earch.py 
uv run python 05_full_text_search.py 
uv run python 06_hybrid_search.py
运行 Agent
cd .. 
uv run python agent_rag.py
```

---

### 11_AgentAPIServer

**🎯 目标**：构建生产级 Agent API 服务，提供 Web 界面

**核心特性**：
- 🌐 FastAPI 后端服务
- 🎨 Gradio Web 界面
- 🔄 HITL 多轮审核
- 👥 多会话并行（thread_id）
- 👤 用户隔离（user_id）
- 💾 状态持久化（短期+长期记忆）

**API 接口**：
| 方法 | 路径         | 说明                   |
| :--: | :----------- | :--------------------- |
| POST | `/ask`       | 启动 Agent 执行        |
| POST | `/intervene` | 提交人工决策，恢复执行 |

**运行方式**：
```python
启动 API 服务
uv run python 11_AgentAPIServer/agent_api.py

启动 Gradio 界面
uv run python 11_AgentAPIServer/gradio_ui.py

测试 API
uv run python 11_AgentAPIServer/api_test.py
```

---

## 🛠️ 技术栈

|   类别   | 技术             | 说明               |
| :------: | :--------------- | :----------------- |
|  🤖 框架  | LangChain V1.x   | Agent 开发核心框架 |
|  🎨 编排  | LangGraph        | 有状态图执行引擎   |
| 🏪 向量库 | Chroma / Milvus  | 向量存储与检索     |
| 🗄️ 数据库 | PostgreSQL       | 持久化存储         |
|  🌐 Web   | FastAPI / Gradio | API 服务与 UI      |
|  📊 观测  | Langfuse         | 可观测性平台       |
|  🔌 协议  | MCP              | 工具统一接入协议   |
|  🐳 容器  | Docker           | 服务部署           |

---

## 🗺️ 学习路径建议

***初学者路径： 01 → 02 → 03 ↓ 进阶路径： 04 → 05 → 06 ↓ 高级路径： 07 → 08 → 09 ↓ 实战路径： 10 → 11**

### 学习建议

1. **按顺序学习**：每个章节都建立在前一章的基础上
2. **动手实践**：修改代码中的参数、Prompt，观察变化
3. **调试分析**：使用 Langfuse 追踪 Agent 的执行过程
4. **扩展优化**：尝试添加自己的工具、修改系统提示词

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 贡献流程

1. Fork 本仓库
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 提交 Pull Request

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

## ⭐ 如果这个项目对你有帮助

请给本仓库一个 Star，让更多人发现这个教程！

[![Star this repo](https://img.shields.io/github/stars/langchain-ai/langchain.svg?style=social)](https://github.com/mrguogood/langchain-v1-agent-cookbook)

---

## 🙏 致谢

感谢 LangChain 社区的持续贡献！

- 附带详细的中文注释和说明

🚀 **祝你学习愉快！** 让我们一起探索 AI Agent 的无限可能！
