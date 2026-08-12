# 从自定义配置模块导入 Config 类，用于读取模型类型等配置
from utils.config import Config
# 从自定义 LLM 工具模块导入 get_llm 方法，用于获取对话模型和向量模型实例
from utils.llms import get_llm
# 从 langchain_chroma 包中导入 Chroma，用于构建和使用基于 Chroma 的向量数据库/向量存储
from langchain_chroma import Chroma
# 从 langchain 导入 tool 装饰器，用于定义 Agent 可以使用的工具函数
from langchain.tools import tool
# 从 langchain 导入创建 Agent 的函数
from langchain.agents import create_agent


# 根据配置中指定的 LLM 类型，获取对话模型 llm_chat 和嵌入模型 llm_embedding 实例
llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

# 使用嵌入模型实例化内存向量数据库实例，用于存储文档向量
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=llm_embedding,
    persist_directory="./chroma_langchain_db",
)

# 使用 @tool 装饰器注册工具，工具名为 "retrieve_context"，描述为根据查询内容在向量数据库中进行相似度搜索
@tool("retrieve_context", description="根据查询内容在向量数据库中进行相似度搜索。",
      response_format="content_and_artifact")
def retrieve_context(query: str):
    # 根据查询内容在向量数据库中进行相似度搜索，返回最相关的 2 个文档
    retrieved_docs = vector_store.similarity_search(query, k=2)
    # 将检索到的文档序列化为字符串格式，包含来源和内容信息
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    print(f"检索到的文本块：{serialized} \n")
    # 返回序列化的字符串和原始文档列表
    return serialized, retrieved_docs

# 将检索工具添加到工具列表中
tools = [retrieve_context]

# 定义 Agent 的系统提示词，说明 Agent 可以使用的工具及其用途
prompt = (
    "你可以使用一个工具来从文档中检索上下文。 "
    "使用该工具来帮助回答用户的问题。"
)
# 创建 Agent 实例，传入对话模型、工具列表和系统提示词
agent = create_agent(llm_chat, tools, system_prompt=prompt)

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