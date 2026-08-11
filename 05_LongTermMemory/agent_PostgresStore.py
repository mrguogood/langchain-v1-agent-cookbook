# 导入 Python 标准库的 uuid 模块，用来生成全局唯一ID
import uuid
# 从 LangChain 导入 create_agent 方法，用于创建智能体（Agent）
from langchain.agents import create_agent
# 这是一个“摘要中间件”，用于在对话过长时，
# 自动用聊天模型对早期消息做摘要并替换原始消息，
# 以此控制上下文长度、同时尽量保留历史关键信息
from langchain.agents.middleware import SummarizationMiddleware
# 从 langchain_core.prompts 模块中导入
# PromptTemplate 用于构建单条文本提示模板,通过占位符+format 的方式动态生成提示词
# ChatPromptTemplate 用于构建多轮对话风格的提示模板,支持 system/human 等多种角色消息组合
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# 从 LangGraph 导入内存检查点存储器，用于短期记忆与会话状态持久化
from langgraph.checkpoint.postgres import PostgresSaver
# 从 langgraph 的 Postgres 存储模块中导入 PostgresStore 类，用于把长期记忆/状态持久化到 PostgreSQL 数据库中
from langgraph.store.postgres import PostgresStore
# 从 LangChain 导入 ToolStrategy，用于指定代理使用“工具调用”的结构化输出格式
from langchain.agents.structured_output import ToolStrategy
# 从自定义配置模块导入 Config 类，用于读取模型类型等配置
from utils.config import Config
# 从自定义 LLM 工具模块导入 get_llm 方法，用于获取对话模型和向量模型实例
from utils.llms import get_llm
# 从自定义工具模块导入 get_tools 方法，用于获取可供 Agent 调用的工具列表
from utils.tools import get_tools
# 从自定义模型定义模块导入上下文 Context 和结构化响应模型 ResponseFormat
from utils.models import Context, ResponseFormat
# 从自定义日志模块导入 LoggerManager，用于获取日志记录器实例
from utils.logger import LoggerManager

# 获取全局日志记录器，用于输出运行过程中的日志信息
logger = LoggerManager.get_logger()

# 根据配置中指定的 LLM 类型，获取对话模型 llm_chat 和嵌入模型 llm_embedding 实例
llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

# 获取当前智能体可用的工具列表
tools = get_tools()

# 使用 PromptTemplate.from_file 从外部文件加载系统提示词模板
# template_file 指定模板文件路径(从配置中读取),encoding 指定文件编码为 UTF-8
# .template 属性返回模板的原始字符串内容(尚未进行变量格式化)
system_prompt = PromptTemplate.from_file(
    template_file = Config.SYSTEM_PROMPT_TMPL,
    encoding = "utf-8"
).template

# 同样使用 PromptTemplate.from_file 从外部文件加载用户提示词(人类消息)模板,
# 用于包装用户输入问题等,形成标准化的人类提示文本
human_prompt = PromptTemplate.from_file(
    template_file = Config.HUMAN_PROMPT_TMPL,
    encoding = "utf-8"
).template

# # 打印加载到的系统提示词模板内容,方便调试和确认是否读取正确
# print(f'system_prompt_tmpl:{system_prompt} \n')
# # 打印加载到的用户提示词模板内容,确认 human prompt 文件内容是否正确
# print(f'human_prompt:{human_prompt} \n')

# 使用 ChatPromptTemplate.from_messages 构建一个聊天提示模板
# 其中包含一条 system 消息和一条 human 消息:
# - system 使用上面加载的系统提示词模板,用于定义 Agent 的角色与规则
# - human 使用上面加载的用户提示词模板,用于定义用户提问的表达方式
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt)
])

# 创建一个基于数据库的检查点存储器，用于保存对话状态，实现短期记忆与多轮会话关联
with (
    PostgresStore.from_conn_string(Config.DB_URI) as store,
    PostgresSaver.from_conn_string(Config.DB_URI) as checkpointer,
):
    # 初始化检查点保存器的数据库表结构
    checkpointer.setup()

    # 初始化长期记忆的数据库表结构
    store.setup()

    # 使用 LangChain 的 create_agent 创建一个 Agent 实例
    # - model: 指定使用的对话 LLM 模型
    # - system_prompt: 指定系统级提示词，约束 Agent 行为
    # - tools: 传入可供 Agent 调用的工具列表
    # - middleware：传入SummarizationMiddleware中间件进行消息修剪
    # - context_schema: 指定上下文对象的 Pydantic（或类似）schema，用于扩展状态信息（如 user_id）
    # - response_format: 使用 ToolStrategy + ResponseFormat 定义结构化输出格式，支持从 Agent 状态中读取 structured_response 字段
    # - checkpointer: 传入 PostgresSaver，使 Agent 具备按线程维度存储和恢复对话状态的能力
    # - store: 传入 PostgresStore，使 Agent 具备按用户维度存储和查找长期记忆(如用户偏好设置)的能力
    agent = create_agent(
        model=llm_chat,
        system_prompt=system_prompt,
        tools=tools,
        middleware=[SummarizationMiddleware(model=llm_chat, trigger=("tokens", 4000), keep=("messages", 3))],
        context_schema=Context,
        response_format=ToolStrategy(ResponseFormat),
        checkpointer=checkpointer,
        store=store
    )

    # 读取长期记忆内容
    def read_long_term_info(user_id: str):
        # 定义用于查询的命名空间，通常用 (类别, 用户ID) 这种形式做分区
        namespace = ("memories", user_id)

        # 在指定命名空间下搜索所有记忆数据，这里 query="" 意味着不带语义过滤，取全部或由实现决定
        memories = store.search(namespace, query="")

        # 如果没有查到任何结果（返回 None），这里先占位，不做额外处理
        if memories is None:
            pass

        # 如果有 memories，则从每个条目中提取 value["data"] 字段，并用空格拼接成一个长字符串；
        # 条件判断确保 d.value 是字典且包含 "data" 键；
        # 如果 memories 为 None 或空列表，则整体结果为 ""（空字符串）
        long_term_info = " ".join(
            [d.value["data"] for d in memories if isinstance(d.value, dict) and "data" in d.value]
        ) if memories else ""

        # 打日志，记录成功获取到的用户长期记忆，以及拼接后的文本长度，方便排查与监控
        logger.info(f"成功获取用户ID: {user_id} 的长期记忆，内容长度: {len(long_term_info)} 字符，内容: {long_term_info} 字符")

        # 返回拼接好的长期记忆文本
        return long_term_info

    # 写入指定用户的长期记忆
    def write_long_term_info(user_id: str, memory_info: str):
        # 定义命名空间，用于把某个用户的记忆归到 ("memories", user_id) 这个层级路径下
        namespace = ("memories", user_id)

        # 生成一个全局唯一的记忆ID，作为该条记忆在命名空间内的 key
        memory_id = str(uuid.uuid4())

        # 调用存储接口，将记忆写入存储：
        # - namespace: 命名空间，用于分层组织数据
        # - key: 该命名空间下唯一的主键（这里用随机UUID）
        # - value: 实际存储内容，这里用 dict 包一层，字段名为 "data"
        result = store.put(
            namespace=namespace,
            key=memory_id,
            value={"data": memory_info}
        )

        # 记录日志，说明为该用户成功写入了一条长期记忆，并打印记忆ID，便于排查
        logger.info(f"成功为用户ID: {user_id} 存储记忆，记忆ID: {memory_id}")

        # 返回给上层一个简单的成功提示文案
        return "记忆存储成功"

    # 写入长期记忆
    write_long_term_info("user_001", "gcs")
    write_long_term_info("user_002", "ggg")

    # （1）第一次问答
    # 定义调用配置，其中 configurable.thread_id 用于标识一段对话的唯一“线程 ID”
    # configurable.user_id 用于标识唯一“用户 ID”
    # 不同 thread_id 之间状态隔离，相同 thread_id 则共享对话上下文与短期记忆
    # 不同 user_id 之间数据隔离，相同 user_id 则共享长期记忆
    config = {
        "configurable": {
            "thread_id": "01",
            "user_id": "user_001",
        }
    }
    # 定义原始用户问题
    raw_question = "杭州的天气怎么样？"
    # 获取长期记忆内容，用户偏好设置(用户名称)
    name = read_long_term_info("user_001")

    # 使用 chat_prompt.format_messages 方法,将模板中的 {question}、{name} 等占位符
    # 替换为实际变量,生成一组完整的对话消息列表(messages)
    messages = chat_prompt.format_messages(question=raw_question, name=name)
    # 取出消息列表中的最后一条消息,通常对应 user(用户)消息,作为本轮要发送给 Agent 的用户提示
    human_msg = messages[-1]
    # 打印最终生成的人类提示内容,便于调试查看模板渲染后的实际文案
    print(f'用户的问题是: {human_msg.content} \n')
    # 将用户提示内容写入日志,方便后续排查问题或重现对话
    logger.info(f"用户的问题是: {human_msg.content}")

    # 调用 Agent 进行对话
    # - messages: 传入用户消息列表，这里用户问“外面的天气怎么样？”
    # - config: 传入包含 thread_id 的配置，用于绑定会话上下文
    # - context: 传入自定义的 Context 对象（如包含 user_id 等业务相关信息）
    response = agent.invoke(
        {"messages": [{"role": "user", "content": human_msg.content}]},
        config=config,
        context=Context(user_id="user_001")
    )

    # 打印 Agent 返回的结构化响应部分（structured_response 一般是按 ResponseFormat 定义的结构化数据）
    print(f"Agent最终回复是: {response['structured_response']} \n\n")
    # 通过日志记录器输出本次回复的 structured_response 内容，便于排查与分析
    logger.info(f"Agent最终回复是: {response['structured_response']}")

    # （2）第二次问答
    # 定义调用配置，其中 configurable.thread_id 用于标识一段对话的唯一“线程 ID”，
    # configurable.user_id 用于标识唯一“用户 ID”
    # 不同 thread_id 之间状态隔离，相同 thread_id 则共享对话上下文与短期记忆
    # 不同 user_id 之间数据隔离，相同 user_id 则共享长期记忆
    config = {
        "configurable": {
            "thread_id": "02",
            "user_id": "user_002",
        }
    }
    # 定义原始用户问题
    raw_question = "南京的天气怎么样？"
    # 获取长期记忆内容，用户偏好设置(用户名称)
    name = read_long_term_info("user_002")

    # 使用 chat_prompt.format_messages 方法,将模板中的 {question}、{name} 等占位符
    # 替换为实际变量,生成一组完整的对话消息列表(messages)
    messages = chat_prompt.format_messages(question=raw_question, name=name)
    # 取出消息列表中的最后一条消息,通常对应 user(用户)消息,作为本轮要发送给 Agent 的用户提示
    human_msg = messages[-1]
    # 打印最终生成的人类提示内容,便于调试查看模板渲染后的实际文案
    print(f'用户的问题是: {human_msg.content} \n')
    # 将用户提示内容写入日志,方便后续排查问题或重现对话
    logger.info(f"用户的问题是: {human_msg.content}")

    # 调用 Agent 进行对话
    # - messages: 传入用户消息列表，这里用户问“外面的天气怎么样？”
    # - config: 传入包含 thread_id 的配置，用于绑定会话上下文
    # - context: 传入自定义的 Context 对象（如包含 user_id 等业务相关信息）
    response = agent.invoke(
        {"messages": [{"role": "user", "content": human_msg.content}]},
        config=config,
        context=Context(user_id="user_002")
    )

    # 打印 Agent 返回的结构化响应部分（structured_response 一般是按 ResponseFormat 定义的结构化数据）
    print(f"Agent最终回复是: {response['structured_response']} \n\n")
    # 通过日志记录器输出本次回复的 structured_response 内容，便于排查与分析
    logger.info(f"Agent最终回复是: {response['structured_response']}")