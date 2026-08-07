# 从 LangChain 导入 create_agent 方法，用于创建智能体（Agent）
from langchain.agents import create_agent
# 从 langchain_core.prompts 模块中导入
# PromptTemplate 用于构建单条文本提示模板,通过占位符+format 的方式动态生成提示词
# ChatPromptTemplate 用于构建多轮对话风格的提示模板,支持 system/human 等多种角色消息组合
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# 从 LangGraph 导入内存检查点存储器，用于短期记忆与会话状态持久化
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

# 打印加载到的系统提示词模板内容,方便调试和确认是否读取正确
# print(f'system_prompt_tmpl:{system_prompt} \n')
# 打印加载到的用户提示词模板内容,确认 human prompt 文件内容是否正确
# print(f'human_prompt:{human_prompt} \n')

# 使用 ChatPromptTemplate.from_messages 构建一个聊天提示模板
# 其中包含一条 system 消息和一条 human 消息:
# - system 使用上面加载的系统提示词模板,用于定义 Agent 的角色与规则
# - human 使用上面加载的用户提示词模板,用于定义用户提问的表达方式
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt)
])

# 创建一个基于内存的检查点存储器，用于保存对话状态，实现短期记忆与多轮会话关联
checkpointer = InMemorySaver()

# 使用 LangChain 的 create_agent 创建一个 Agent 实例
# - model: 指定使用的对话 LLM 模型
# - system_prompt: 指定系统级提示词，约束 Agent 行为
# - tools: 传入可供 Agent 调用的工具列表
# - context_schema: 指定上下文对象的 Pydantic（或类似）schema，用于扩展状态信息（如 user_id）
# - response_format: 使用 ToolStrategy + ResponseFormat 定义结构化输出格式，支持从 Agent 状态中读取 structured_response 字段
# - checkpointer: 传入 InMemorySaver，使 Agent 具备按线程维度存储和恢复对话状态的能力
agent = create_agent(
    model=llm_chat,
    system_prompt=system_prompt,
    tools=tools,
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

# 定义调用配置，其中 configurable.thread_id 用于标识一段对话的唯一“线程 ID”
# 不同 thread_id 之间状态隔离，相同 thread_id 则共享对话上下文与短期记忆
config = {"configurable": {"thread_id": "1"}}

# 定义原始用户问题
raw_question_1 = "外面的天气怎么样？"
# 定义用户名称
name = "gcs"
# 使用 chat_prompt.format_messages 方法,将模板中的 {question}、{name} 等占位符
# 替换为实际变量,生成一组完整的对话消息列表(messages)
messages_1 = chat_prompt.format_messages(question=raw_question_1, name=name)
# 取出消息列表中的最后一条消息,通常对应 user(用户)消息,作为本轮要发送给 Agent 的用户提示
human_msg_1 = messages_1[-1]
# 打印最终生成的人类提示内容,便于调试查看模板渲染后的实际文案
# print(f'用户的问题是: {human_msg_1.content} \n')
# 将用户提示内容写入日志,方便后续排查问题或重现对话
logger.info(f"用户的问题是: {human_msg_1.content}")

# 定义原始用户问题
raw_question_2 = "杭州的天气怎么样？"
# 定义用户名称
name = "gcs"
# 使用 chat_prompt.format_messages 方法,将模板中的 {question}、{name} 等占位符
# 替换为实际变量,生成一组完整的对话消息列表(messages)
messages_2 = chat_prompt.format_messages(question=raw_question_2, name=name)
# 取出消息列表中的最后一条消息,通常对应 user(用户)消息,作为本轮要发送给 Agent 的用户提示
human_msg_2 = messages_2[-1]
# 打印最终生成的人类提示内容,便于调试查看模板渲染后的实际文案
# print(f'用户的问题是: {human_msg_2.content} \n')
# 将用户提示内容写入日志,方便后续排查问题或重现对话
logger.info(f"用户的问题是: {human_msg_2.content}")

# 调用 Agent 进行对话
# - messages: 传入用户消息列表，这里用户问“外面的天气怎么样？”
# - config: 传入包含 thread_id 的配置，用于绑定会话上下文
# - context: 传入自定义的 Context 对象（如包含 user_id 等业务相关信息）

# （1）返回整个批次的最终输出 默认在所有请求完成后一次性返回整个列表
# 一次性并行提交两条用户消息给 Agent 处理
responses = agent.batch(
    [
        {"messages": [{"role": "user", "content": human_msg_1.content}]},
        {"messages": [{"role": "user", "content": human_msg_2.content}]}
    ],
    config=config,
    context=Context(user_id="1")
)
# batch() 返回的是一个「响应列表」，每个元素对应一条输入的最终结果
for response in responses:
    # 打印 Agent 返回的结构化响应部分（structured_response 一般是按 ResponseFormat 定义的结构化数据）
    print(f"Agent最终回复是: {response['structured_response']} \n")
    # 通过日志记录器输出本次回复的 structured_response 内容，便于排查与分析
    logger.info(f"Agent最终回复是: {response['structured_response']}")

# （2）在每个输入生成完成时接收输出 谁先算完先拿谁的结果
for response in agent.batch_as_completed(
    [
        {"messages": [{"role": "user", "content": human_msg_1.content}]},
        {"messages": [{"role": "user", "content": human_msg_2.content}]}
    ],
    config=config,
    context=Context(user_id="1")
):
    # batch_as_completed 每次返回的通常是一个元组 (idx, result)
    # idx：当前这条结果对应的是第几个输入（下标索引）
    # result：当前这条输入对应的真正 Agent 执行结果
    idx, result = response
    # 打印 Agent 返回的结构化响应部分（structured_response 一般是按 ResponseFormat 定义的结构化数据）
    print(f"Agent最终回复是: {result['structured_response']} \n")
    # 通过日志记录器输出本次回复的 structured_response 内容，便于排查与分析
    logger.info(f"Agent最终回复是: {result['structured_response']}")