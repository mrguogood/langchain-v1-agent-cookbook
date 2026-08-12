# RAG(检索增强生成)

## 1、介绍  

核心功能包含：    

- RAG流程：加载文档->切分文档->Embedding创建索引->存入向量数据库->检索文档->交给Agent生成回复
- 2‑Step RAG、Agentic RAG 检索

### RAG介绍

大型语言模型（LLM）非常强大，但也有两大局限性:            

- 有限上下文，它们无法一次性摄入整个语料库    
- 静态知识，它们的训练数据在某个时间点后就不再更新  

**检索**通过在查询时获取相关的外部知识来解决这些问题。这是**检索增强生成 (Retrieval-Augmented Generation, RAG)** 的基础：用特定于上下文的信息来增强 LLM 的答案。

RAG架构：2-Step RAG、Agentic RAG、Hybrid RAG                         
官方文档链接:https://docs.langchain.com/oss/python/langchain/retrieval
中文文档链接：[检索 | LangChain 中文文档](https://langchain-doc.cn/v1/python/langchain/retrieval.html)         

#### 2‑Step RAG

2‑Step RAG 是最经典、最简单的一种 RAG 架构：永远是“先检索，再生成”，检索到的文档固定作为上下文喂给 LLM，一般每个请求只需要一次模型调用，延迟和成本都比较好控制     

**特点**           

- 结构简单、行为可预测，适合把“查资料”视为前置条件的应用，如 FAQ、文档机器人等
- 控制力高：最大 LLM 调用次数是预先固定的，一般是一次
- 延迟较快且可预估，但仍会受检索 API、网络和数据库性能影响

**典型处理流程拆解**            

以 LangChain 文档里的描述为基础，一个 2‑Step RAG 请求大致会经历这些步骤：          

(1) 用户提出问题或任务                    

(2) 用问题去查询知识库（向量库、SQL、搜索引擎等），检索一批最相关的文档片段            

(3) 将原始问题 + 若干检索到的文档组合成 prompt，作为上下文发给 LLM           

(4) LLM 在这些上下文基础上生成答案，实现“有依据”的回复              

#### Agentic RAG

让一个由 LLM 驱动的 agent 在推理过程中(思考‑行动‑观察‑再思考)自己决定“什么时候检索、怎么检索、用什么工具检索”，而不是一开始就固定先检索再回答，因此更灵活，但控制力和时延可预测性会下降     

**特点**           

- 控制：低，因为调用多少次工具、多少次 LLM 步骤由 agent 动态决定
- 灵活性：高，适合复杂任务和多工具场景
- 时延：可变，调用越多步、越多工具，整体延迟越大、越难预估

**典型处理流程拆解**            

在这种架构中，RAG 的关键不再是“管道的顺序”，而是 将检索封装成工具，让 agent 在推理过程中按需调用      

(1) 用户提出问题或任务       

(2) Agent（LLM）读取系统提示、工具列表和历史对话，决定下一步行动                 
 - 直接回答（如果已有足够信息）
 - 调用某个检索或数据工具获取外部知识

(3) 工具执行（例如访问 API、数据库、向量库或网页），返回结果文本          

(4) Agent 将工具结果视为新的“观察（observation）”，继续推理         
 - 可能再调用更多工具
 - 或者整理信息，生成最终答案

(5) 把整理后的回答返回给用户                   

#### Hybrid RAG

Hybrid RAG 结合了 2‑Step RAG 的固定流程和 Agentic RAG 的灵活推理，在检索前后增加预处理与验证步骤，可以多轮迭代提升答案质量             

**特点**          

- 控制：中等，比 Agentic 可控，比纯 2‑Step 更灵活 
- 灵活性：中等，可以调整查询和答案，但不完全交给 agent 自由发挥
- 时延：可变，因为可能会多轮重试和自纠错
- 典型场景：需要质量校验的领域问答系统         

**典型流程结构**              

(1) Query enhancement（查询增强）      

在检索前对用户问题进行处理，例如： 
- 改写不清晰的问题 
- 生成多个变体
- 用额外上下文扩展查询，以提高召回质量

(2) Retrieval validation（检索验证）    

- 判断当前检索结果是否足够相关、数量是否足够
- 如果不够，可以调整查询、阈值或检索策略，再检索一次

(3) Answer validation（答案验证）         
- 检查生成的答案是否准确、完整、是否和来源内容一致
- 如果不满足要求，可以让系统重新生成或局部修正答案                              

## 2、功能测试   

### 2.1 使用Docker方式运行PostgreSQL数据库     

进入官网 https://www.docker.com/ 下载安装Docker Desktop软件并安装，安装完成后打开软件                      

打开命令行终端，`cd 04_ShortTermMemory/postgresql` 文件夹下                     
- 进入到 postgresql 下执行 `docker-compose up -d` 运行 PostgreSQL 服务                            

运行成功后可在Docker Desktop软件中进行管理操作或使用命令行操作或使用指令                                 

### 2.2 运行脚本测试            

```python
python 1_create_index.py
python 2_2step_rag.py
python 3_agentic_rag.py
python 4_agent_rag.py
```