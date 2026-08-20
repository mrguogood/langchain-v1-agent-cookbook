# ReAct 风格 Agent API 接口服务  

## 1、介绍

核心功能包含：    

- HITL（人工介入审查）：支持工具调用需要人工审核（approve / reject / edit）
- 多轮中断支持：可以连续多轮工具调用 + 审核
- 会话隔离：通过 thread_id 实现多会话并行
- 用户隔离：通过 user_id 隔离长期记忆
- 状态持久化：对话历史（短期） + 用户偏好/记忆（长期）都保存在 PostgreSQL
- 上下文传递：Context 对象携带 user_id，工具函数可访问
- 2个核心 POST API 接口 /ask 和 /intervene。/ask 接收用户问题，执行 Agent（带 HITL 支持）。/intervene，接收人工决策，恢复被中断的 Agent 执行 
- Gradio实现的简单Web端测试页面

### FastAPI介绍

FastAPI 是一个现代、高性能的 Python Web 框架，用于构建 API 接口           

FastAPI 的核心是结合了 Starlette（高性能 ASGI 框架）和 Pydantic（数据验证库），这使得它在速度上与 Node.js 或 Go 等框架相当，而开发体验更友好        

**FastAPI 的关键特点：**      

FastAPI 提供了许多开箱即用的功能，让 API 开发变得高效       

- 高性能：基于 ASGI，支持异步编程，基准测试中 FastAPI 是 Python 框架中最快的之一，仅次于 Starlette 本身  
- 类型提示与自动验证：利用 Python 类型注解（如 str, int, List），Pydantic 会自动处理数据验证、序列化和错误处理。这减少了 boilerplate 代码，并提高了代码安全性
- 自动文档生成：内置 OpenAPI 支持，自动生成 Swagger UI 和 ReDoc 接口文档。你可以直接在浏览器中测试 API，无需额外工具
- 依赖注入：内置依赖系统，支持数据库连接、认证等依赖的自动管理
- 异步支持：原生支持 async/await，适合 IO-bound 操作如数据库查询或外部 API 调用
- 安全性：内置 OAuth2、JWT 等认证支持，以及 CORS、CSRF 防护
- 测试友好：由于类型提示和依赖注入，单元测试非常简单              


## 2、功能测试   

### 2.1 使用Docker方式运行PostgreSQL数据库和Milvus向量数据库

进入官网 https://www.docker.com/ 下载安装Docker Desktop软件并安装，安装完成后打开软件                      

打开命令行终端，运行如下指令进行部署                     

- 进入到 postgresql 下执行 `docker-compose up -d` 运行 PostgreSQL 服务                             
- 进入到 milvus 下执行 `docker-compose up -d` 运行 Milvus 服务                  

运行成功后可在Docker Desktop软件中进行管理操作或使用命令行操作或使用指令                          

### 2.2 功能测试          

```bash
# 1、Milvus向量数据库测试
cd milvus
python 01_create_database.py
python 02_create_collection.py
python 03_insert_data.py
python 04_basic_earch.py
python 05_full_text_search.py
python 06_hybrid_search.py

# 2、MCP Server测试
cd rag_mcp
python mix_text_search.py
python mcp_start.py
python rag_mcp_server.py

# 3、Agent测试
python agent_api.py
python api_test.py
python gradio_ui.py

```

按照如下参考问题进行测试:          
(1)不指定搜索类型和条件过滤                   
搜索关于多模态大模型持续学习系列研究相关的文章，并给出文章的标题、链接、发布者               

(2)指定搜索类型和条件过滤     
全文搜索关于多模态大模型持续学习系列研究相关的文章，文章发布时间在2025.09.10之前，返回3篇文章并给出文章的标题、链接、发布者               
语义搜索关于多模态大模型持续学习系列研究相关的文章，文章发布时间在2025.09.10之前，返回3篇文章并给出文章的标题、链接、发布者                
混合搜索关于多模态大模型持续学习系列研究相关的文章，文章发布时间在2025.09.10之前，返回3篇文章并给出文章的标题、链接、发布者           

(3)不指定搜索类型但指定条件过滤(存在多个过滤条件)             
搜索关于多模态大模型持续学习系列研究相关的文章，文章发布时间在2025.09.10之前，发布者是新智元，返回3篇文章并给出文章的标题、链接、发布者        

(4)不指定搜索类型但指定条件过滤(存在无关过滤条件干扰)            
搜索关于多模态大模型持续学习系列研究相关的文章，文章发布时间在2025.09.10之前，发布者是新智元，字数在200字内，价格不超过500元，返回3篇文章并给出文章的标题、链接、发布者        
