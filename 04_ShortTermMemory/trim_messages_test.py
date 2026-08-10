# 从 langchain.agents 中导入创建 Agent 的函数 create_agent, 以及默认的 AgentState 状态类型
from langchain.agents import create_agent, AgentState
# 从 langchain.agents.middleware 中导入 before_model 装饰器,用于声明“模型调用前”的中间件
from langchain.agents.middleware import before_model
# 从 langchain_core.messages 中导入 RemoveMessage, 用于在状态中删除消息
from langchain_core.messages import RemoveMessage
# 从 langgraph.graph.message 中导入特殊常量 REMOVE_ALL_MESSAGES, 表示删除全部消息的占位 ID
from langgraph.graph.message import REMOVE_ALL_MESSAGES
# 从 langgraph.runtime 中导入 Runtime, 表示运行时上下文(包含状态、配置等)
from langgraph.runtime import Runtime
# 从 typing 中导入 Any 类型, 用于类型注解中表示“任意类型”
from typing import Any
# 从 LangGraph 导入内存检查点存储器，用于短期记忆与会话状态持久化
from langgraph.checkpoint.memory import InMemorySaver
# 从自定义配置模块导入 Config 类，用于读取模型类型等配置
from utils.config import Config
# 从自定义 LLM 工具模块导入 get_llm 方法，用于获取对话模型和向量模型实例
from utils.llms import get_llm
# 从自定义日志模块导入 LoggerManager，用于获取日志记录器实例
from utils.logger import LoggerManager

# 获取全局日志记录器，用于输出运行过程中的日志信息
logger = LoggerManager.get_logger()

# 根据配置中指定的 LLM 类型，获取对话模型 llm_chat 和嵌入模型 llm_embedding 实例
llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

# 创建一个基于内存的检查点存储器，用于保存对话状态，实现短期记忆与多轮会话关联
checkpointer = InMemorySaver()

# 使用 @before_model 装饰器,声明这是一个在调用模型前执行的中间件函数
@before_model
# 定义 trim_messages 函数,用于在每次调 LLM 前对消息进行修剪
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    # 从当前的 AgentState 中获取消息列表(对话历史)
    messages = state["messages"]
    # 打印修剪前的消息列表,方便调试查看原始上下文
    logger.info(f'修剪前的消息: {messages}')

    # 如果消息数量不超过 3 条,说明上下文还很短,无需修剪
    # 返回 None 表示不对 state 做任何修改
    if len(messages) <= 3:
        return None

    # 根据消息总数的奇偶性决定保留最后 3 条或 4 条消息
    # 目的是尽量保留完整的 user/assistant 轮次结构
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    # 打印修剪后的消息列表,用于对比和调试
    logger.info(f'修剪后的消息: {recent_messages}')

    # 返回一个用于更新 state 的字典,只修改 "messages" 这个字段
    return {
        "messages": [
            # 先插入一个 RemoveMessage 指令,表示清空当前所有历史消息
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            # 再把刚才选出的最近几条消息追加进去,形成新的精简上下文
            *recent_messages
        ]
    }

# 使用 LangChain 的 create_agent 创建一个 Agent 实例
agent = create_agent(
    model=llm_chat,
    middleware=[trim_messages],
    checkpointer=checkpointer
)

# 定义调用配置，其中 configurable.thread_id 用于标识一段对话的唯一“线程 ID”
# 不同 thread_id 之间状态隔离，相同 thread_id 则共享对话上下文与短期记忆
config = {"configurable": {"thread_id": "1"}}

# 调用 Agent 进行第一次对话
response = agent.invoke(
    {"messages": [{"role": "user", "content": "我的名字叫 gcs。"}]},
    config=config
)
response["messages"][-1].pretty_print()
print("\n")


# 调用 Agent 进行第二次对话
response = agent.invoke(
    {"messages": [{"role": "user", "content": "写一首关于冬天的四言绝句"}]},
    config=config
)
response["messages"][-1].pretty_print()
print("\n")


# 调用 Agent 进行第三次对话
response = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
    config=config
)
response["messages"][-1].pretty_print()
print("\n")