# 从自定义配置模块导入 Config 类，用于读取模型类型等配置
from utils.config import Config
# 从自定义 LLM 工具模块导入 get_llm 方法，用于获取对话模型和向量模型实例
from utils.llms import get_llm
# 从 langchain_chroma 包中导入 Chroma，用于构建和使用基于 Chroma 的向量数据库/向量存储
from langchain_chroma import Chroma
# 从 langchain 导入创建 Agent 的函数
from langchain.agents import create_agent
# 从 langchain 的 Agent 中间件模块导入动态提示装饰器和模型请求类
from langchain.agents.middleware import dynamic_prompt, ModelRequest

# 根据配置中指定的 LLM 类型，获取对话模型 llm_chat 和嵌入模型 llm_embedding 实例
llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

# 使用嵌入模型实例化内存向量数据库实例，用于存储文档向量
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=llm_embedding,
    persist_directory="./chroma_langchain_db",
)

# 使用 @dynamic_prompt 装饰器定义动态提示词函数，该函数会在每次模型调用前执行
@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    # 从请求状态中获取最后一条消息的文本内容作为查询
    last_query = request.state["messages"][-1].text
    # 在向量数据库中搜索与查询相关的文档
    retrieved_docs = vector_store.similarity_search(last_query)

    # 将检索到的所有文档内容用双换行符连接成一个字符串
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # 构建系统消息，将检索到的上下文信息注入到提示词中
    system_message = """你是一个智能、友好且乐于助人的AI助手。你的目标是：

    1. **准确理解**：仔细理解用户的意图和需求
    2. **提供价值**：给出有用、准确、可操作的信息和建议
    3. **保持专业**：在不确定时明确说明，不胡编乱造
    4. **友善沟通**：使用温和、尊重的语气，适当使用表情符号增强亲和力

    ## 回答原则
    - 如果不确定答案，直接说"我不确定"或"我需要更多信息"
    - 对于事实性问题，提供来源或引用
    - 对于复杂问题，分步骤或分要点回答
    - 适当追问以澄清模糊需求

    ## 输出格式
    - 列表项用 - 或 1. 2. 3. 组织，
    - 不使用markdown格式输出
    """


    propt  = (
        system_message + "在你的回复中使用以下上下文："
        f"\n\n{docs_content}"
    )

    # 返回包含上下文信息的系统消息
    return propt

# 创建 Agent 实例，传入对话模型、空工具列表和动态提示词中间件
agent = create_agent(llm_chat, middleware=[prompt_with_context])

# 定义用户查询内容，包含两个问题
query = (
    "张三九基本信息？"
)

response = agent.invoke(
    # 将用户输入包装为消息列表，角色为 user，内容为 query
    {"messages": [{"role": "user", "content": query}]}
)
# 从 response 字典中取出 "messages" 列表的最后一条消息的内容，作为代理（Agent）的最终回复结果
result = response["messages"][-1].content
# 在控制台打印代理最终回复内容，方便调试和查看
print(f"Agent最终回复是: {result} \n")