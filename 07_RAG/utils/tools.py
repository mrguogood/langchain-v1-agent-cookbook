# 从 LangChain 导入 tool 装饰器和 ToolRuntime，用于定义可被 Agent 调用的工具及其运行上下文类型
from langchain.tools import tool, ToolRuntime
# 从 langgraph.config 模块中导入 get_stream_writer 函数
# get_stream_writer 用于在图节点或工具内部获取一个“流式写入器”
# 通过这个写入器可以在 stream_mode="custom" 时向外持续推送自定义数据(如进度日志、调试信息等)
from langgraph.config import get_stream_writer
# Human-in-the-loop 中间件，用于在工具调用前做人审查
from langchain.agents.middleware import HumanInTheLoopMiddleware
# 从当前包中导入 Context 模型，用于在工具运行时携带用户等上下文信息
from .models import Context
# 从当前包中导入 LoggerManager，用于获取日志记录器实例
from .logger import LoggerManager
# 从 langchain_chroma 包中导入 Chroma，用于构建和使用基于 Chroma 的向量数据库/向量存储
from langchain_chroma import Chroma
# 从自定义 LLM 工具模块导入 get_llm 方法，用于获取对话模型和向量模型实例
from .llms import get_llm
# 从自定义配置模块导入 Config 类，用于读取模型类型等配置
from .config import Config

# 获取全局日志实例，用于在工具加载和调用过程中记录日志
logger = LoggerManager.get_logger()

# 根据配置中指定的 LLM 类型，获取对话模型 llm_chat 和嵌入模型 llm_embedding 实例
llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

# 定义一个函数，用于构建并返回当前 Agent 可用的工具列表
def get_tools():

    ###################### 1、定义工具 ######################

    # 使用 @tool 装饰器注册一个工具，工具名为 "list_refund_reasons"，描述为“为指定的城市获取天气。”
    # 该工具接收城市名称，并返回该城市的天气描述字符串
    @tool("get_weather_for_location", description="为指定的城市获取天气。")
    def get_weather_for_location(city: str) -> str:
        # 从当前 LangGraph 执行上下文中获取一个流式写入器,用于发送自定义流数据
        writer = get_stream_writer()
        # 通过流式写入器发送自定义日志: 表示正在查找该城市的数据
        writer(f"正在查找城市数据: {city}")
        # 再次通过流式写入器发送自定义日志: 表示已成功获取该城市的数据
        writer(f"已获取城市数据: {city}")
        # 根据传入的城市名返回一个固定的晴天描述（此处为示例逻辑，未实际调用天气 API）
        return f"{city}的天气是晴天!"

    # 使用 @tool 装饰器注册第二个工具，工具名为 "get_user_location"，描述为“根据用户 ID 检索用户信息。”
    # 该工具通过 ToolRuntime 获取上下文中的用户信息，从而推断用户所在城市
    @tool("get_user_location", description="根据用户 ID 检索用户信息。")
    def get_user_location(runtime: ToolRuntime[Context]) -> str:
        # 从运行时上下文中读取 user_id，用于根据用户 ID 判断所属城市
        user_id = runtime.context.user_id
        # 简单的示例映射：user_id 为 "1" 时返回“北京”，否则返回“上海”
        return "北京" if user_id == "user_001" else "上海"

    # 使用 @tool 装饰器注册第三个工具，工具名为 "retrieve_context"，描述为根据查询内容在向量数据库中进行相似度搜索
    # 使用嵌入模型实例化内存向量数据库实例，用于存储文档向量
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=llm_embedding,
        persist_directory="./chroma_langchain_db",
    )
    # 使用 @tool 装饰器注册第二个工具，工具名为 "retrieve_context"，描述为“根据查询内容在向量数据库中进行相似度搜索”
    @tool("retrieve_context", description="根据查询内容在向量数据库中进行相似度搜索。",response_format="content_and_artifact")
    def retrieve_context(query: str):
        # 根据查询内容在向量数据库中进行相似度搜索，返回最相关的 2 个文档
        retrieved_docs = vector_store.similarity_search(query, k=2)
        # 将检索到的文档序列化为字符串格式，包含来源和内容信息
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        logger.info(f"检索到的文本块：{serialized}")
        # 返回序列化的字符串和原始文档列表
        return serialized, retrieved_docs

    # 将定义好的两个工具函数封装到列表中，作为 Agent 可调用的工具集合
    tools = [
        get_weather_for_location,
        get_user_location,
        retrieve_context
    ]
    # 记录当前获取到的工具列表，方便在调试或运行中查看已注册的工具
    logger.info(f"获取并提供的工具列表: {tools} ")


    ###################### 2、Human-in-the-loop 策略配置 ######################

    # 构建一个工具中断配置字典，键为工具名称，值为该工具的审核配置信息
    interrupt_on = {
        'get_weather_for_location': {
            'allowed_decisions': ['approve', 'edit', 'reject'],
            'description': '调用 get_weather_for_location 工具需要人工审批。请输入 approve(同意)、reject(拒绝) 或 edit(编辑参数)'
        },
         'get_user_location': {
             'allowed_decisions': ['approve', 'edit', 'reject'],
             'description': '调用 get_user_location 工具需要人工审批。请输入 approve(同意)、reject(拒绝) 或 edit(编辑参数)'
         },
        'retrieve_context': False
    }
    logger.info(f"需要人工审批的工具有：{interrupt_on}")

    # 创建一个人工介入循环（Human-in-the-loop）中间件实例
    # 该中间件用于在工具调用时拦截并等待人工审核
    hitl_middleware = HumanInTheLoopMiddleware(
        # 传入中断配置字典，指定哪些工具需要人工审核以及审核规则
        # interrupt_on 包含每个工具的审核配置（allowed_decisions 和 description）
        interrupt_on=interrupt_on,
        # 设置描述信息的前缀文本
        # 当触发人工审核时，会在提示信息前添加此前缀
        description_prefix="工具调用需要人工审批"
    )

    # 返回完整的工具列表和中间件
    return tools, hitl_middleware