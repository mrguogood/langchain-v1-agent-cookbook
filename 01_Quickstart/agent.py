# 导入操作系统模块，用于设置和读取环境变量
import os
# 从 LangChain 导入 create_agent 方法，用于创建智能体（Agent）
from langchain.agents import create_agent
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
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# 设置 LangSmith 的 API Key（实际项目中应通过环境变量或安全配置管理，避免写死在代码中）
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

# 获取全局日志记录器，用于输出运行过程中的日志信息
logger = LoggerManager.get_logger()

# 根据配置中指定的 LLM 类型，获取对话模型 llm_chat 和嵌入模型 llm_embedding 实例
llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

# 获取当前智能体可用的工具列表
tools = get_tools()

# 定义系统提示词，指定 Agent 的角色和行为约束
SYSTEM_PROMPT = """# 角色
你是一名专业天气预报员，回复风格幽默，擅长在天气播报中融入谐音梗和冷笑话。

# 可用工具
1. get_user_location — 根据当前用户 ID 查询其所在城市（无需传参）
2. get_weather_for_location(city) — 查询指定城市的天气，city 为城市名称

# 工作流程
按以下顺序处理每条用户消息，不要跳过步骤：

## 1. 识别意图
- 天气查询：用户询问某地或"外面/这里/我这边"的天气
- 位置查询：用户询问"我在哪/我现在在哪里/我的位置"
- 闲聊：与天气或位置无关 → 直接结构化回复，不调用工具

## 2. 确定地点（仅天气查询）
| 情况 | 处理方式 |
|------|----------|
| 用户明确说出城市名（如"北京天气"） | 直接调用 get_weather_for_location |
| 指代当前位置（"外面""这里""我这边""当地"等） | 先调用 get_user_location，再调用 get_weather_for_location |
| 未提及任何地点且无法从对话历史推断 | 在 punny_response 中礼貌追问地点，不调用天气工具 |
| 多轮对话中上一轮已确定地点 | 沿用已知地点，无需重复定位 |

## 3. 调用工具
- 先定位、后查天气：get_user_location → get_weather_for_location
- 工具返回的是事实数据，不要编造或修改
- 位置查询只需调用 get_user_location

## 4. 结构化输出（必须）
收集到足够信息后，**必须**通过 ResponseFormat 输出最终回复，禁止输出纯文本。

字段要求：
- punny_response（必填）：主回复，包含天气/位置信息与冷笑话或谐音梗，语气轻松自然
- weather_conditions（选填）：仅在有天气数据时填写，简洁描述天气状况（如"晴，25°C"）

# 示例
用户："外面的天气怎么样？"
→ get_user_location → get_weather_for_location → ResponseFormat

用户："上海天气如何？"
→ get_weather_for_location("上海") → ResponseFormat

用户："我现在在哪里？"
→ get_user_location → ResponseFormat（punny_response 告知城市，weather_conditions 留空）
"""

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
# - context_schema: 指定上下文对象的 Pydantic（或类似）schema，用于扩展状态信息（如 user_id）
# - response_format: 使用 ToolStrategy + ResponseFormat 定义结构化输出格式，支持从 Agent 状态中读取 structured_response 字段
# - checkpointer: 传入 InMemorySaver，使 Agent 具备按线程维度存储和恢复对话状态的能力
agent = create_agent(
    model=llm_chat,
    system_prompt=SYSTEM_PROMPT,
    tools=tools,
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat, handle_errors=True),
    checkpointer=checkpointer
)

# 定义调用配置，其中 configurable.thread_id 用于标识一段对话的唯一“线程 ID”
# 不同 thread_id 之间状态隔离，相同 thread_id 则共享对话上下文与短期记忆
config = {"configurable": {"thread_id": "1"}}

# 调用 Agent 进行第一次对话
# - messages: 传入用户消息列表，这里用户问“外面的天气怎么样？”
# - config: 传入包含 thread_id 的配置，用于绑定会话上下文
# - context: 传入自定义的 Context 对象（如包含 user_id 等业务相关信息）
response = agent.invoke(
    {"messages": [{"role": "user", "content": "外面的天气怎么样？"}]},
    config=config,
    context=Context(user_id="1")
)


# 打印 Agent 返回的结构化响应部分（structured_response 一般是按 ResponseFormat 定义的结构化数据）
print(f"Agent最终回复是: {response['structured_response']} \n")
# 通过日志记录器输出本次回复的 structured_response 内容，便于排查与分析
logger.info(f"Agent最终回复是: {response['structured_response']}")

# 再次调用 Agent，继续同一 thread_id 下的对话，从而复用短期记忆和已有上下文
response = agent.invoke(
    {"messages": [{"role": "user", "content": "我现在在哪里？"}]},
    config=config,
    context=Context(user_id="1")
)

# 打印第二次调用的结构化响应内容
print(f"Agent最终回复是: {response['structured_response']} \n")
# 通过日志记录器记录第二轮对话的 structured_response 结果
logger.info(f"Agent最终回复是: : {response['structured_response']}")