# Agent_MCP

## 1、介绍

核心功能包含：    

- 自定义 MCP Server：将 RAG 工具封装为一个 MCP Server 对外提供使用 
- Agent使用 MCP Server 

### MCP介绍

Model Context Protocol（MCP）是一个开放标准，用来规范「大模型如何安全、统一地调用外部工具和数据源」                   

- MCP 由 Anthropic 在 2024 年提出，被多家大模型厂商采用，用来统一「模型 ↔ 外部系统」的交互方式
- 底层基于 JSON-RPC 2.0，定义了一套通用消息格式，让模型可以发现工具、调用函数、读取资源和使用预设 prompt

**MCP 的核心概念**      

(1)Tools（工具） 

- MCP Server 可以暴露一组「可执行函数」，例如查数据库、调第三方 API、发请求到内部系统等
- 在 LangChain 中，这些 MCP tools 会被自动映射成 LangChain 的 Tool 对象，Agent 可以像用普通 Tool 一样调用

(2)Resources（资源）

- 用来暴露数据，如文件内容、数据库记录、HTTP 响应等，客户端可以以统一方式读取文本或二进制内容
- LangChain 会把这些资源转成 Blob 对象，方便后续做检索、解析或加载进上下文 

(3)Prompts（预设提示词）

- MCP 还支持让服务器提供一系列预设 prompt 模板，客户端或模型可以直接复用这些高质量提示词 

**MCP 与 LangChain 的集成**

- LangChain 通过 langchain-mcp-adapters 等库接入 MCP，提供 MCP 客户端（如 MultiServerMCPClient），可以同时连多个 MCP Server
- MCP tools 会被转成 LangChain 的 tools 列表，统一交给 Agent；MCP resources 会变成 Blob，用于读取上下文数据
- 传输层支持本地 stdio、SSE（Server-Sent Events）、HTTP Streamable，既适合本地开发，又能部署到云端服务

**相关链接** 

- MCP官方简介:https://www.anthropic.com/news/model-context-protocol             
- MCP文档手册:https://modelcontextprotocol.io/introduction           
- MCP官方服务器列表:https://github.com/modelcontextprotocol/servers           

## 2、功能测试   

### 2.1 使用Docker方式运行PostgreSQL数据库     

进入官网 https://www.docker.com/ 下载安装Docker Desktop软件并安装，安装完成后打开软件                      

打开命令行终端，`cd 04_ShortTermMemory/postgresql` 文件夹下                     
- 进入到 postgresql 下执行 `docker-compose up -d` 运行 PostgreSQL 服务                            

运行成功后可在Docker Desktop软件中进行管理操作或使用命令行操作或使用指令                                 

### 2.2 运行脚本测试            

```bash
python create_index.py
python mcp_start.py
python agent_rag.py
```