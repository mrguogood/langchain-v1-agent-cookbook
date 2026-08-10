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
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.memory import InMemorySaver
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

# 设置 LangSmith 相关环境变量，开启 LangChain V2 版链路追踪，用于观测与调试 Agent 执行过程
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# 设置 LangSmith 的 API Key（实际项目中应通过环境变量或安全配置管理，避免写死在代码中）
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

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

# 当 LangGraph 使用 InMemorySaver checkpointer 保存状态时，它会用 msgpack 序列化整个状态（包括 structured_response 字段）
# 为了确保在恢复状态时能够正确解析 ResponseFormat 和 Context，我们需要在 serde 中指定白名单
# 定义自定义序列化器，把 ResponseFormat（以及 Context）加入白名单
serde = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("utils.models", "ResponseFormat"),
        # 如果 Context 也是自定义 dataclass/Pydantic，建议一起加
        ("utils.models", "Context"),
    ]
)
# 创建一个基于内存的检查点存储器，用于保存对话状态，实现短期记忆与多轮会话关联
checkpointer = InMemorySaver(serde=serde)

# 使用 LangChain 的 create_agent 创建一个 Agent 实例
# - model: 指定使用的对话 LLM 模型
# - system_prompt: 指定系统级提示词，约束 Agent 行为
# - tools: 传入可供 Agent 调用的工具列表
# - context_schema: 指定上下文对象的 Pydantic（或类似）schema，用于扩展状态信息（如 user_id）。
# 告诉 LangGraph：Agent 的状态中除了 messages 之外，还有一个 context 字段，类型是 Context。不直接发给模型（但 Tools 可以访问），不会被摘要，始终保留原始值
# - response_format: 使用 ToolStrategy + ResponseFormat 定义结构化输出格式，支持从 Agent 状态中读取 structured_response 字段
# - checkpointer: 传入 InMemorySaver，使 Agent 具备按线程维度存储和恢复对话状态的能力
agent = create_agent(
    model=llm_chat,
    system_prompt=system_prompt,
    tools=tools,
    # 使用一个摘要中间件来管理对话历史,避免上下文过长
    # 用于生成摘要的小模型,一般比主模型更便宜,专门负责“压缩历史”
    # 触发摘要的条件: 当累计 token 数超过 4000 时,启动一次“对历史消息做摘要”，摘要对象包括全部消息类型：用户输入，模型输出，工具返回结果
    # 摘要之后仍然“保留”的原始内容: 这里表示保留最近 3 条 messages 不做摘要，keep=("messages", 3) 只是说明“在做摘要时保留最近 3 条原文消息”
    middleware=[SummarizationMiddleware(model=llm_chat, trigger=("tokens", 4000), keep=("messages", 3))],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

# （1）第一次问答
# 定义调用配置，其中 configurable.thread_id 用于标识一段对话的唯一“线程 ID”
# 不同 thread_id 之间状态隔离，相同 thread_id 则共享对话上下文与短期记忆
config = {"configurable": {"thread_id": "1"}}
raw_question = "杭州的天气怎么样？"
# 定义用户名称
name = "gcs"
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
    context=Context(user_id="1")
)

# 打印 Agent 返回的结构化响应部分（structured_response 一般是按 ResponseFormat 定义的结构化数据）
print(f"Agent最终回复是: {response['structured_response']} \n\n")
# 通过日志记录器输出本次回复的 structured_response 内容，便于排查与分析
logger.info(f"Agent最终回复是: {response['structured_response']}")


# （2）第二次问答 相同的会话ID
# 定义调用配置，其中 configurable.thread_id 用于标识一段对话的唯一“线程 ID”
# 不同 thread_id 之间状态隔离，相同 thread_id 则共享对话上下文与短期记忆
config = {"configurable": {"thread_id": "1"}}
raw_question = "我刚问的是哪个城市的天气？"
# 定义用户名称
name = "gcs"
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
#context=Context(user_id="1") 是在告诉 Agent "当前对话是由用户 1 发起的"，这个信息会被注入到 Agent 状态中并可被 Tools 访问，且不会像 messages 一样被摘要压缩。
# Agent 状态中，context.user_id 会被保存下来，并作为参数传递给 Tools，Tools 可以通过 context.user_id 访问该信息。
#应用场景: 多租户隔离,每个用户都有自己的状态,不会被其他用户干扰。工具获取上下文信息时,通过 context.user_id 可以区分不同用户的对话历史.审计日志，
response = agent.invoke(
    {"messages": [{"role": "user", "content": human_msg.content}]},
    config=config,
    context=Context(user_id="1")
)

# 打印 Agent 返回的结构化响应部分（structured_response 一般是按 ResponseFormat 定义的结构化数据）
print(f"Agent最终回复是: {response['structured_response']} \n\n")
# 通过日志记录器输出本次回复的 structured_response 内容，便于排查与分析
logger.info(f"Agent最终回复是: {response['structured_response']}")


# （3）第三次问答 不同的会话ID
# 定义调用配置，其中 configurable.thread_id 用于标识一段对话的唯一“线程 ID”
# 不同 thread_id 之间状态隔离，相同 thread_id 则共享对话上下文与短期记忆
config = {"configurable": {"thread_id": "2"}}
# 定义原始用户问题
raw_question = "我刚问的是哪个城市的天气？"
# 定义用户名称
name = "gcs"
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
    context=Context(user_id="1")
)

# 打印 Agent 返回的结构化响应部分（structured_response 一般是按 ResponseFormat 定义的结构化数据）
print(f"Agent最终回复是: {response['structured_response']} \n\n")
# 通过日志记录器输出本次回复的 structured_response 内容，便于排查与分析
logger.info(f"Agent最终回复是: {response['structured_response']}")