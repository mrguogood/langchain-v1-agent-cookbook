# 导入操作系统模块，用于设置和读取环境变量
import os
# 从 LangChain 导入 create_agent 方法，用于创建智能体（Agent）
from langchain.agents import create_agent
# 从 langchain_core.messages 模块中导入三种消息类型:
# AIMessageChunk: 表示 LLM 流式输出的增量消息分片(用于 messages 流式模式)
# AIMessage: 表示完整的一条 AI 消息(包含最终文本、工具调用等完整信息)
# ToolMessage: 表示工具执行完成后返回给代理的消息(包含工具返回内容等)
from langchain_core.messages import AIMessageChunk, AIMessage, ToolMessage
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
raw_question = "外面的天气怎么样？"
# 定义用户名称
name = "gcs"
# 使用 chat_prompt.format_messages 方法,将模板中的 {question}、{name} 等占位符
# 替换为实际变量,生成一组完整的对话消息列表(messages)
messages = chat_prompt.format_messages(question=raw_question, name=name)
# 取出消息列表中的最后一条消息,通常对应 user(用户)消息,作为本轮要发送给 Agent 的用户提示
human_msg = messages[-1]
# 打印最终生成的人类提示内容,便于调试查看模板渲染后的实际文案
# print(f'用户的问题是: {human_msg.content} \n\n')
# 将用户提示内容写入日志,方便后续排查问题或重现对话
logger.info(f"用户的问题是: {human_msg.content}")

# 调用 Agent 进行对话
# - messages: 传入用户消息列表，这里用户问“外面的天气怎么样？”
# - config: 传入包含 thread_id 的配置，用于绑定会话上下文
# - context: 传入自定义的 Context 对象（如包含 user_id 等业务相关信息）
# - stream_mode: 传入模式

# （1）使用updates模式 Agent步骤级进度流 eg:厨房每做完一道菜,喊一声"菜好了！"
# 遍历 agent 的流式结果,每次循环拿到当前一步的状态增量(chunk)
for chunk in agent.stream(
    # 传入对话消息,这里 human_msg.content 是用户输入内容
    {"messages": [{"role": "user", "content": human_msg.content}]},
    # 传入运行配置,例如线程 id、检查点等
    config=config,
    # 传入上下文对象,可用于在图中读取 user_id 等信息
    context=Context(user_id="1"),
    # 使用 "updates" 模式,按代理步骤(stream agent progress)推送状态更新
    stream_mode="updates",
):
    # chunk 是一个 dict, key 是步骤名(step), value 是该步骤的状态数据
    for step, data in chunk.items():
        # 打印当前步骤名,便于在控制台观察执行流程
        print(f"当前步骤: {step}")
        # 通过 logger 记录当前步骤名,方便日志排查
        logger.info(f"当前步骤: {step}")
        # 打印当前步骤最后一条消息的 content_blocks,通常是本步骤的主要输出内容
        print(f"当前步骤返回内容: {data['messages'][-1].content_blocks} \n\n")
        # 通过 logger 记录当前步骤的主要输出内容
        logger.info(f"当前步骤返回内容: {data['messages'][-1].content_blocks}")

# （2）使用messages模式 LLM token 级流式输出 c厨师每切一刀菜,都向你汇报切了什么
# 遍历 agent 在 "messages" 模式下流式返回的结果
for token, metadata in agent.stream(
    # 传入当前轮对话的消息体,这里 human_msg.content 是用户输入内容
    {"messages": [{"role": "user", "content": human_msg.content}]},
    # 运行配置,例如线程、检查点等
    config=config,
    # 自定义上下文,在图中可通过 Context 读取 user_id 等
    context=Context(user_id="1"),
    # 使用 "messages" 模式,按 LLM token/message chunk 粒度流式输出
    stream_mode="messages"
):
    # 打印当前产生 token 的 LangGraph 节点名(例如 "model" 或 "tools")
    print(f"当前节点: {metadata['langgraph_node']}")
    # 将当前节点名写入日志,方便排查和观测
    logger.info(f"当前节点: {metadata['langgraph_node']}")
    # 打印该 token 的内容块列表,包括文本 token 或 tool_call_chunk 等
    print(f"当前节点内容: {token.content_blocks} \n\n")
    # 将当前 token 的内容块写入日志,用于回放或调试
    logger.info(f"当前节点内容: {token.content_blocks}")

# （3）使用custom模式 任意自定义数据流 eg:厨师通过对讲机随时喊话,内容完全自定义
# 在 "custom" 模式下遍历 agent 的流式输出,每个 chunk 都是由工具等节点通过 get_stream_writer 主动写出的自定义数据
for chunk in agent.stream(
    # 传入当前轮的对话消息,这里 human_msg.content 是用户输入内容
    {"messages": [{"role": "user", "content": human_msg.content}]},
    # 运行配置,如线程 id、检查点等
    config=config,
    # 传入上下文,在图中可通过 Context 读取 user_id 等信息
    context=Context(user_id="1"),
    # 使用 "custom" 模式,只接收通过 stream writer 发出的自定义流数据
    stream_mode="custom",
):
    # 打印当前收到的自定义数据块,便于在控制台实时查看进度或业务日志
    print(f"当前数据块: {chunk}")
    # 将自定义数据块写入日志,用于后续排查或审计
    logger.info(f"当前数据块: {chunk}")

# （4）使用updates和custom组合模式
# 在同一个流中同时启用 "updates" 和 "custom" 两种流式模式
for stream_mode, chunk in agent.stream(
    # 传入当前轮对话消息,其中 human_msg.content 是用户输入的文本
    {"messages": [{"role": "user", "content": human_msg.content}]},
    # 运行配置,例如线程 id、检查点等
    config=config,
    # 传入上下文,在图/中间件中可以拿到 user_id 等信息
    context=Context(user_id="1"),
    # 指定要开启的流式模式列表: 代理步骤更新 + 自定义数据
    stream_mode=["updates", "custom"]
):
    # 打印当前收到的数据属于哪种流式模式(updates/custom)
    print(f"当前流式模式: {stream_mode}")
    # 将当前流式模式写入日志,方便后续排查
    logger.info(f"当前流式模式: {stream_mode}")

    # 当流式模式为 custom 时,chunk 是工具等节点通过 get_stream_writer 写出的自定义数据
    if stream_mode == "custom":
        # 打印当前 custom 模式返回的数据内容,通常是业务进度或调试信息
        print(f"当前流式模式返回的内容: {chunk} \n\n")
        # 将 custom 数据写入日志
        logger.info(f"当前流式模式返回的内容: {chunk}")
    # 当流式模式为 updates 时,chunk 是一次代理步骤(state)更新的字典
    elif stream_mode == "updates":
        # 遍历本次步骤更新中所有节点(步骤)的数据,step 是节点名,如 "model" 或 "tools"
        for step, data in chunk.items():
            # 打印当前步骤名,用于观察 agent 执行到哪一步
            print(f"当前步骤: {step}")
            # 将当前步骤名写入日志
            logger.info(f"当前步骤: {step}")
            # 打印当前步骤中最后一条消息的内容块,通常是该步骤的主要输出
            print(f"当前步骤返回内容: {data['messages'][-1].content_blocks} \n\n")
            # 将当前步骤输出内容写入日志
            logger.info(f"当前步骤返回内容: {data['messages'][-1].content_blocks}")

# （5）使用messages和custom组合模式
# 在同一个流中同时启用 "messages" 和 "custom" 两种流式模式
for stream_mode, payload in agent.stream(
    # 传入当前轮对话消息,其中 human_msg.content 是用户输入内容
    {"messages": [{"role": "user", "content": human_msg.content}]},
    # 运行配置,例如线程 id、检查点等
    config=config,
    # 上下文信息,在图中可以通过 Context 获取 user_id 等
    context=Context(user_id="1"),
    # 指定要开启的流式模式列表: LLM 消息分片 + 自定义数据
    stream_mode=["messages", "custom"]
):
    # 打印当前收到的数据属于哪种流式模式(messages/custom)
    print(f"当前流式模式: {stream_mode}")
    # 将当前流式模式写入日志,方便观测和排查
    logger.info(f"当前流式模式: {stream_mode}")

    # 当模式为 custom 时,payload 是通过 get_stream_writer 写出的自定义数据
    if stream_mode == "custom":
        # 打印当前 custom 模式返回的内容,通常用来承载业务进度或调试信息
        print(f"当前流式模式返回的内容: {payload} \n\n")
        # 将 custom 数据写入日志
        logger.info(f"当前流式模式返回的内容: {payload}")
    # 当模式为 messages 时,payload 是 (token, metadata) 二元组
    elif stream_mode == "messages":
        # 解包出当前 token 以及其对应的元数据(包含节点名等信息)
        token, metadata = payload
        # 打印当前产生 token 的 LangGraph 节点名,如 "model" 或 "tools"
        print(f"当前节点: {metadata['langgraph_node']}")
        # 将当前节点名写入日志
        logger.info(f"当前节点: {metadata['langgraph_node']}")
        # 打印当前 token 的内容块列表,包括文本 token 或 tool_call_chunk 等
        print(f"当前节点内容: {token.content_blocks} \n\n")
        # 将当前 token 内容块写入日志,用于回放或调试
        logger.info(f"当前节点内容: {token.content_blocks}")

# （6）使用messages和updates组合模式
# 只有在涉及工具调用,且想拿到“完整解析好”的工具调用或完整消息时使用messages和custom组合模式
# 在 stream_mode="messages" 时,LLM 输出的是一连串增量的message chunk
# 对于工具调用会先以多个tool_call_chunk的形式逐步吐出JSON片段
# 比如先输出 {", 再输出 "city", 再输出 :"Boston" 等
# 在同一个流里同时启用 "messages" 和 "updates" 两种流式模式
for stream_mode, payload in agent.stream(
    # 传入当前轮对话消息,其中 human_msg.content 是用户输入内容
    {"messages": [{"role": "user", "content": human_msg.content}]},
    # 运行配置,例如线程 id、检查点等信息
    config=config,
    # 上下文信息,在图中可以通过 Context 获取 user_id 等
    context=Context(user_id="1"),
    # 指定要开启的流式模式列表: LLM 消息分片 + 代理步骤更新
    stream_mode=["messages", "updates"]
):
    # 打印当前返回的数据属于哪种流式模式(messages/updates)
    print(f"当前流式模式: {stream_mode}")
    # 将当前流式模式写入日志,便于观测和排查
    logger.info(f"当前流式模式: {stream_mode}")

    # 当模式为 messages 时,payload 是 (token, metadata) 二元组
    if stream_mode == "messages":
        # 解包出当前的消息分片 token 以及其元数据(包含节点名等信息)
        token, metadata = payload
        # 如果是 LLM 的增量输出(AIMessageChunk),分别处理文本和工具调用参数
        if isinstance(token, AIMessageChunk):
            # 若当前 chunk 中包含文本内容,则流式打印出来(结尾不换行,用于打字机效果)
            if token.text:
                print(f"流式文本数据：{token.text} ", end="|")
                logger.info(f"流式文本数据：{token.text} ")
            # 若当前 chunk 中包含工具调用参数增量(tool_call_chunks),则一次性打印
            if token.tool_call_chunks:
                print(f"流式工具参数数据：{token.tool_call_chunks}")
                logger.info(f"流式工具参数数据：{token.tool_call_chunks}")

    # 当模式为 updates 时,payload 是一次代理步骤(state)更新的字典
    elif stream_mode == "updates":
        # 遍历本次更新中所有来源(source),通常是节点名,如 "model"、"tools"
        for source, update in payload.items():
            # 只关心来自模型节点或工具节点的更新
            if source in ("model", "tools"):
                # 取出该节点最新的一条消息,通常是本步骤的核心输出
                message = update["messages"][-1]
                # 如果是 AIMessage 且带有 tool_calls,说明这里有完整的工具调用参数
                if isinstance(message, AIMessage) and message.tool_calls:
                    print(f"非流式工具调用完整参数: {message.tool_calls}")
                    logger.info(f"非流式工具调用完整参数: {message.tool_calls}")
                # 如果是 ToolMessage,则说明这是工具执行后的完整返回内容
                if isinstance(message, ToolMessage):
                    print(f"非流式工具调用完整返回内容: {message.content_blocks}")
                    logger.info(f"非流式工具调用完整返回内容: {message.content_blocks}")