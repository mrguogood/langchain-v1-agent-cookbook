# 从 LangChain 导入 tool 装饰器和 ToolRuntime，用于定义可被 Agent 调用的工具及其运行上下文类型
from langchain.tools import tool, ToolRuntime
# 从当前包中导入 Context 模型，用于在工具运行时携带用户等上下文信息
from .models import Context
# 从当前包中导入 LoggerManager，用于获取日志记录器实例
from .logger import LoggerManager


# 获取全局日志实例，用于在工具加载和调用过程中记录日志
logger = LoggerManager.get_logger()

# 定义一个函数，用于构建并返回当前 Agent 可用的工具列表
def get_tools():

    # 使用 @tool 装饰器注册一个工具，工具名为 "list_refund_reasons"，描述为“为指定的城市获取天气。”
    # 该工具接收城市名称，并返回该城市的天气描述字符串
    @tool("get_weather_for_location", description="为指定的城市获取天气。")
    def get_weather_for_location(city: str) -> str:
        # 根据传入的城市名返回一个固定的晴天描述（此处为示例逻辑，未实际调用天气 API）
        return f"{city}的天气是晴天!"

    # 使用 @tool 装饰器注册第二个工具，工具名为 "get_user_location"，描述为“根据用户 ID 检索用户信息。”
    # 该工具通过 ToolRuntime 获取上下文中的用户信息，从而推断用户所在城市
    @tool("get_user_location", description="根据用户 ID 检索用户信息。")
    def get_user_location(runtime: ToolRuntime[Context]) -> str:
        # 从运行时上下文中读取 user_id，用于根据用户 ID 判断所属城市
        user_id = runtime.context.user_id
        # 简单的示例映射：user_id 为 "1" 时返回“北京”，否则返回“上海”
        return "北京" if user_id == "1" else "上海"

    # 将定义好的两个工具函数封装到列表中，作为 Agent 可调用的工具集合
    tools = [
        get_weather_for_location,
        get_user_location
    ]

    # 记录当前获取到的工具列表，方便在调试或运行中查看已注册的工具
    logger.info(f"获取并提供的工具列表: {tools} ")

    # 返回完整的工具列表，供上层创建 Agent 时注入
    return tools