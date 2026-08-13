# 导入MCP服务器的底层实现类
from mcp.server.lowlevel import Server
# 导入MCP协议的类型定义，包括工具和文本内容
from mcp.types import Tool, TextContent
# 导入配置模块，包含系统配置信息
from utils.config import Config
# 从 langchain_chroma 包中导入 Chroma，用于构建和使用基于 Chroma 的向量数据库/向量存储
from langchain_chroma import Chroma
# 从自定义 LLM 工具模块导入 get_llm 方法，用于获取对话模型和向量模型实例
from utils.llms import get_llm
# 导入日志管理器模块
from utils.logger import LoggerManager

# 获取全局日志记录器，用于输出运行过程中的日志信息
logger = LoggerManager.get_logger()

# 根据配置中指定的 LLM 类型，获取对话模型 llm_chat 和嵌入模型 llm_embedding 实例
llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

# 创建一个名为"rag_mcp_server"的MCP服务器实例
mcp = Server("rag_mcp_server")

# 声明 list_tools 函数为一个列出工具的接口
# 使用装饰器将该函数注册为列出工具的处理函数
@mcp.list_tools()
# 定义异步函数，返回工具列表
# 返回值: Tool对象的列表
async def list_toogls() -> list[Tool]:
    # 记录正在列出工具的日志信息
    logger.info("Listing tools...")
    # 函数返回一个列表，其中包含一个 Tool 对象
    # 每个 Tool 对象代表一个工具，其属性定义了工具的功能和输入要求
    # 返回包含一个Tool对象的列表
    return [
        # 创建Tool对象，定义文档搜索工具
        Tool(
            # 设置工具名称为"retrieve_context"
            name="retrieve_context",
            # 设置工具的功能描述
            description="根据查询内容在向量数据库中进行相似度搜索。",
            # 定义输入参数的JSON Schema
            inputSchema={
                # 指定输入类型为对象
                "type": "object",
                # 定义对象的各个属性
                "properties": {
                    # 定义query属性，存储搜索查询文本
                    "query": {
                        # 指定属性类型为字符串
                        "type": "string",
                        # 设置属性的描述信息
                        "description": "执行搜索的内容"
                    }
                },
                # 列出输入对象的必需属性
                # 指定必需的属性列表
                "required": ["query"]
            }
        )
    ]


# 声明 call_tool 函数为一个工具调用的接口
# 使用装饰器将该函数注册为调用工具的处理函数
@mcp.call_tool()
# 定义异步函数，处理工具调用请求
# name: 要调用的工具名称
# arguments: 工具参数字典
# 返回值: TextContent对象的列表
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # 检查工具名称 name 是否是 retrieve_context
    # 如果 query 为空或未提供，抛出 ValueError 异常，提示用户必须提供查询语句
    # 验证工具名称是否为"retrieve_context"
    if name != "retrieve_context":
        # 如果工具名称不匹配，抛出值错误异常
        raise ValueError(f"Unknown tool: {name}")

    # 从参数字典中获取query参数
    query = arguments.get("query")
    # 验证query参数是否存在
    if not query:
        # 如果不存在，抛出值错误异常
        raise ValueError("Query is required")

    # 使用try-except捕获可能的异常
    try:
        # 使用嵌入模型实例化内存向量数据库实例，用于存储文档向量
        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=llm_embedding,
            persist_directory="./chroma_langchain_db",
        )

        # 根据查询内容在向量数据库中进行相似度搜索，返回最相关的 2 个文档
        retrieved_docs = vector_store.similarity_search(query, k=2)
        # 将检索到的文档序列化为字符串格式，包含来源和内容信息
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        logger.info(f"检索到的文本块：{serialized}")

        # 返回一个包含查询结果的 TextContent 对象
        # 创建TextContent对象，将搜索结果封装为文本内容并返回
        return [TextContent(type="text", text=serialized)]

    # 捕获所有异常
    except Exception as e:
        # 记录主程序执行异常的错误日志
        logger.error(f"主程序执行异常: {e}")
        # 打印程序异常终止的信息
        print("程序异常终止。")
        logger.info("程序异常终止。")
        # 返回程序异常的错误信息
        return [TextContent(type="text", text="\n主程序执行异常，程序异常终止")]


# 主程序入口
if __name__ == "__main__":

    # 初始化并运行服务器，使用streamable_http传输协议
    mcp.run(transport="streamable_http")
