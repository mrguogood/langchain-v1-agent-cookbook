# ReAct 风格 Agent API 接口服务（带 Skill 按需加载）

## 1、介绍   

- Agent 按需加载 skill 技能，每个 skill 必包括 `SKILL.md` 、 可选可执行的脚本(*.py)       

### Skill设计思想

Skill 的设计思想可以概括为三句话：**按需加载、说明即能力、工具化接入**   

1. **按需加载（Progressive Disclosure）**
- **问题：** 如果把所有“摘要 / 翻译 / 写 SQL …”的详细说明都塞进系统提示，会占很多 token，成本高、上下文也容易乱
- **做法：** 系统提示里只告诉 Agent“你有 load_skill 工具，可用技能有 summarize、translate”。当用户真的问“总结一下这段话”或“翻译成英文”时，Agent 再调用 load_skill("summarize") 或 load_skill("translate")，把对应 SKILL.md 的全文拉进对话
- **效果：** 平时不占 token，只有用到某技能时才加载该技能的说明

2. **说明即能力（Prompt-Driven Specialization）**
- **思想：** 一个“技能”本质上是一段**专门说明**：在什么场景用、输入输出是什么、要遵循什么步骤和原则。Agent 不需要为“摘要”单独写死代码，只要按这段说明执行即可
- **实现：** 每个技能必须包括一个 SKILL.md：
    - Frontmatter：name、description（给 Agent 做“要不要用这个技能”的匹配）
    - 正文：步骤、原则、注意事项（给 Agent 做“怎么用”的执行指南）
- **好处：** 
    - 改能力 = 改文档，不必改 Python
    - 新技能 = 新目录 + 新 SKILL.md + 可选的可执行脚本，load_skill 自动发现，扩展简单

3. **工具化接入（Skill as a Tool）**
- **思想：** 不改变现有 Agent 架构（create_agent + tools + HITL），把“技能”当成一种**特殊工具**：
    - 工具名：load_skill
    - 输入：skill_name（如 summarize、translate）
    - 输出：该技能的完整说明文本（即 SKILL.md 内容）
- **效果：** 
    - Agent 的决策流程不变：先想“用户要摘要 → 我该用哪个工具？”→ 选 load_skill("summarize")，拿到说明后再按说明生成摘要                       

## 2、功能测试

### 2.1 使用Docker方式运行PostgreSQL数据库和Milvus向量数据库

进入官网 [https://www.docker.com/](https://www.docker.com/) 下载安装Docker Desktop软件并安装，安装完成后打开软件                      

打开命令行终端，运行如下指令进行部署                     

- 进入到 postgresql 下执行 `docker-compose up -d` 运行 PostgreSQL 服务                             
- 进入到 milvus 下执行 `docker-compose up -d` 运行 Milvus 服务

运行成功后可在Docker Desktop软件中进行管理操作或使用命令行操作或使用指令                           

### 2.2 功能测试

```python
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
python rag_mcp_server_test.py

# 3、Agent测试
python agent_api.py
python api_test.py
python gradio_ui.py

```
