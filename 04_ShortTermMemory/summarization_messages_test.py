# 从 langchain.agents 中导入创建 Agent 的函数 create_agent, 以及默认的 AgentState 状态类型
from langchain.agents import create_agent, AgentState
# 从 langchain.agents.middleware 中导入 before_model 装饰器,用于声明“模型调用前”的中间件
from langchain.agents.middleware import before_model, SummarizationMiddleware
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

# 使用 LangChain 的 create_agent 创建一个 Agent 实例
agent = create_agent(
    model=llm_chat,
    # 使用一个摘要中间件来管理对话历史,避免上下文过长
    # 用于生成摘要的小模型,一般比主模型更便宜,专门负责“压缩历史”
    # 触发摘要的条件: 当累计 token 数超过 4000 时,启动一次“对历史消息做摘要”
    # 摘要之后仍然“保留”的原始内容: 这里表示保留最近 3 条 messages 不做摘要，keep=("messages", 3) 只是说明“在做摘要时保留最近 3 条原文消息”
    middleware=[SummarizationMiddleware(model=llm_chat,trigger=("tokens", 4000),keep=("messages", 3))],
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
    {"messages": [{"role": "user", "content": "写一首关于秋天的四言绝句"}]},
    config=config
)
response["messages"][-1].pretty_print()
print("\n")

# 调用 Agent 进行第四次对话
response = agent.invoke(
    {"messages": [{"role": "user", "content": "写一首关于夏天的四言绝句"}]},
    config=config
)
response["messages"][-1].pretty_print()
print("\n")

# 调用 Agent 进行第五次对话
response = agent.invoke(
    {"messages": [{"role": "user", "content": "写一首关于春天的四言绝句"}]},
    config=config
)
response["messages"][-1].pretty_print()
print("\n")


# 调用 Agent 进行第六次对话
response = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
    config=config
)
response["messages"][-1].pretty_print()
print("\n")