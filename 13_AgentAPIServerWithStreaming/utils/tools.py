# 从 LangChain 导入 tool 装饰器和 ToolRuntime，用于定义可被 Agent 调用的工具及其运行上下文类型
from langchain.tools import tool, ToolRuntime
# 用于 JSON 序列化工具输出
import json
# 用于从文件动态加载 python 模块（执行本地脚本逻辑）
import importlib.util
import sys
from pathlib import Path
from functools import lru_cache
import re
# 从 langgraph.config 模块中导入 get_stream_writer 函数
# get_stream_writer 用于在图节点或工具内部获取一个“流式写入器”
# 通过这个写入器可以在 stream_mode="custom" 时向外持续推送自定义数据(如进度日志、调试信息等)
from langgraph.config import get_stream_writer
# Human-in-the-loop 中间件，用于在工具调用前做人审查
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_mcp_adapters.client import MultiServerMCPClient
# 从自定义配置模块导入 Config 类，用于读取模型类型等配置
from .config import Config
# 从当前包中导入 Context 模型，用于在工具运行时携带用户等上下文信息
from .models import Context
# 从自定义 LLM 工具模块导入 get_llm 方法，用于获取对话模型和向量模型实例
from .llms import get_llm
# 从当前包中导入 LoggerManager，用于获取日志记录器实例
from .logger import LoggerManager
# 从技能模块导入 load_skill 工具，用于按需加载 Skill 说明（摘要、翻译等）
from .skills import get_load_skill_tool



# 获取全局日志实例，用于在工具加载和调用过程中记录日志
logger = LoggerManager.get_logger()

# 根据配置中指定的 LLM 类型，获取对话模型 llm_chat 和嵌入模型 llm_embedding 实例
llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

# skills 根目录（与 utils 同级的 skills 目录）
_SKILLS_ROOT = Path(__file__).resolve().parent.parent / "skills"

_RE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# 使用 lru_cache 装饰器缓存模块加载结果，最多缓存16个实例
@lru_cache(maxsize=16)
def _load_skill_python_module(skill_name: str, filename: str):
    """
    从 skills/<skill_name>/<filename> 动态加载 python 模块。
    用于把“技能目录中的脚本”变成可执行逻辑（供工具调用）。
    """
    # 校验 skill_name 是否为空或包含非法字符（防止目录穿越）
    if not skill_name or ".." in skill_name or "/" in skill_name or "\\" in skill_name:
        raise ValueError("技能目录名 skill_name 不能为空或包含非法字符")
    # 校验 filename 是否合法，只允许以 .py 结尾，且不能包含路径穿越符号
    if not filename.endswith(".py") or ".." in filename or "/" in filename or "\\" in filename:
        raise ValueError("技能脚本文件名 filename 不能为空或包含非法字符")

    # 组合脚本的完整路径
    path = _SKILLS_ROOT / skill_name / filename
    # 检查目标脚本文件是否真实存在
    if not path.is_file():
        raise FileNotFoundError(f"未找到技能脚本: {path}")

    # 构造模块名称，避免命名冲突
    module_name = f"skill_{skill_name}_{path.stem}"
    # 创建模块加载的 spec 对象
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    # 检查 spec 是否创建成功
    if spec is None or spec.loader is None:
        raise ImportError(f"无法创建模块加载 spec: {path}")
    # 根据 spec 创建一个新的模块对象
    module = importlib.util.module_from_spec(spec)
    # 必须首先注册到 sys.modules，否则脚本内 @dataclass 等装饰器会因 __module__ 查不到导致报错
    sys.modules[module_name] = module
    # 执行模块代码，将其加载到内存中
    spec.loader.exec_module(module)
    # 返回已加载的模块对象
    return module


def _jsonify_tool_result(obj):
    """
    尝试把工具结果转换为可 JSON 序列化的对象：
    - 若为 None，返回空 dict 避免后续 __dict__ 等访问报错
    - 若存在 as_dict()，优先使用
    - 否则原样返回（json.dumps 会在失败时抛错）
    """
    if obj is None:
        return {}
    as_dict = getattr(obj, "as_dict", None)
    if callable(as_dict):
        return as_dict()
    return obj


# 定义一个函数，用于构建并返回当前 Agent 可用的工具列表
async def get_tools():

    ###################### 1、定义工具 ######################

    # 使用 @tool 装饰器注册一个工具，工具名为 "get_weather_for_location"，描述为“为指定的城市获取天气。”
    # 该工具接收城市名称，并返回该城市的天气描述字符串
    @tool("get_weather_for_location", description="为指定的城市获取天气。")
    async def get_weather_for_location(city: str) -> str:
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
    async def get_user_location(runtime: ToolRuntime[Context]) -> str:
        # 从运行时上下文中读取 user_id，用于根据用户 ID 判断所属城市
        user_id = runtime.context.user_id
        # 简单的示例映射：user_id 为 "user_001" 时返回“北京”，否则返回“上海”
        return "北京" if user_id == "user_001" else "上海"

    # 通用：执行 skills/<skill_name>/<script_name> 中的任意函数
    # 使用 @tool 装饰器定义工具，名称和功能描述如下
    @tool(
        "run_skill_python",
        description=(
            "执行 skills/<skill_name>/<script_name> 内的指定函数，并返回 JSON 字符串。"
            "用于让 Agent 具备“执行技能目录脚本”的能力。"
        ),
    )
    # 定义异步工具方法，支持传入技能名、脚本名、函数名、函数参数和额外参数
    async def run_skill_python(
        skill_name: str,# 技能目录名
        script_name: str,# 脚本文件名（含 .py）
        function_name: str, # 要执行的函数名
        function_kwargs: dict | None = None, # 函数参数（优先使用）
        # 兼容部分校验/运行层注入的额外字段（例如 v__kwargs）
        **extra,
    ) -> str:
        # 获取流式写入器，用于向 Agent 控制台输出执行信息
        writer = get_stream_writer()
        # 向控制台输出正在执行的技能脚本信息
        writer(f"正在执行技能脚本：{skill_name}/{script_name}::{function_name} …")
        # 记录日志，表示技能脚本开始执行
        logger.info(f"正在执行技能脚本：{skill_name}/{script_name}::{function_name} …")
        try:
            # 校验函数名是否合法（必须符合 Python 标识符规范）
            if not _RE_IDENTIFIER.match(function_name or ""):
                logger.error(f"function_name 非法（必须是 python 标识符）")
                return json.dumps({"error": "function_name 非法（必须是 python 标识符）"}, ensure_ascii=False)
            # 加载指定技能目录下对应脚本模块
            mod = _load_skill_python_module(skill_name, script_name)
            # 根据函数名获取模块中的函数对象
            fn = getattr(mod, function_name, None)
            # 若未找到可调用的函数，则报错并返回
            if fn is None or not callable(fn):
                logger.error(f"脚本中未找到可调用函数：{function_name}")
                return json.dumps(
                    {"error": f"脚本中未找到可调用函数：{function_name}"},
                    ensure_ascii=False,
                )
            # 兼容性处理：尝试从额外参数获取 v__kwargs
            v_kwargs = extra.get("v__kwargs")
            # 优先使用 function_kwargs，其次使用 v__kwargs，最后为空字典
            raw = function_kwargs if function_kwargs is not None else (v_kwargs if v_kwargs is not None else {})
            # 兼容 LLM/框架传入参数为 JSON 字符串的情况，若为字符串则尝试解析
            # 避免 raw 为 None 时出现 __dict__ 等属性访问错误
            if raw is None:
                raw = {}
            elif isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    raw = {}
            # 确认参数为字典格式，否则给空字典
            call_args = raw if isinstance(raw, dict) else {}
            # 调用指定函数并传入参数
            result = fn(**call_args)
            # 结果序列化处理，转为可 JSON 序列化对象
            payload = _jsonify_tool_result(result)
            # 记录执行结果日志
            logger.info(f"技能脚本执行结果：{payload}")
            # 返回 JSON 字符串形式的执行结果
            return json.dumps(payload, ensure_ascii=False)
        except Exception as e:
            # 引入 traceback 便于日志报错定位
            import traceback
            # 记录异常日志，带上详细堆栈信息
            logger.error(f"run_skill_python 执行失败: {e}\n{traceback.format_exc()}")
            # 以 JSON 字符串返回异常信息
            return json.dumps({"error": f"run_skill_python 执行失败: {e}"}, ensure_ascii=False)

    # 调用 MCP Server，工具名为 "search_documents"，描述为根据查询内容在向量数据库中进行相似度搜索
    client = MultiServerMCPClient({
        "rag_mcp_server": {
            "url": "http://127.0.0.1:8010/mcp",
            "transport": "streamable_http",
        }
    })
    # 从 MCP Server 中获取可提供使用的全部工具
    tools = await client.get_tools()

    # 注册 Skill 加载工具：Agent 在需要摘要、翻译等能力时先调用 load_skill 获取技能说明，再按说明执行
    tools.append(get_load_skill_tool())
    # 将定义好的天气与用户位置工具加入列表，作为 Agent 可调用的工具集合
    tools.append(get_weather_for_location)
    tools.append(get_user_location)
    # 将“可执行技能工具”加入工具列表
    tools.append(run_skill_python)

    # 记录当前获取到的工具列表，方便在调试或运行中查看已注册的工具
    logger.info(f"获取并提供的工具列表: {tools} ")


    ###################### 2、Human-in-the-loop 策略配置 ######################

    # 构建一个工具中断配置字典，键为工具名称，值为该工具的审核配置信息
    # 未在 interrupt_on 中的工具默认为 auto-approved；显式设置 load_skill: False 表示不中断、直接执行
    interrupt_on = {
        'load_skill': False,
        'run_skill_python': False,
        'get_weather_for_location': {
            'allowed_decisions': ['approve', 'edit', 'reject'],
            'description': '调用 get_weather_for_location 工具需要人工审批。请输入 approve(同意)、reject(拒绝) 或 edit(编辑参数)'
        },
         'get_user_location': {
             'allowed_decisions': ['approve', 'edit', 'reject'],
             'description': '调用 get_user_location 工具需要人工审批。请输入 approve(同意)、reject(拒绝) 或 edit(编辑参数)'
         },
        'search_documents': {
             'allowed_decisions': ['approve', 'edit', 'reject'],
             'description': '调用 search_documents 工具需要人工审批。请输入 approve(同意)、reject(拒绝) 或 edit(编辑参数)'
         },
    }
    logger.info(f"需要人工审批的工具有：{interrupt_on}")

    # 创建一个人工介入循环（Human-in-the-loop）中间件实例
    # 该中间件用于在工具调用时拦截并等待人工审核
    hitl_middleware = HumanInTheLoopMiddleware(
        # 传入中断配置字典，指定哪些工具需要人工审核以及审核规则
        interrupt_on=interrupt_on,
        # 设置描述信息的前缀文本，当触发人工审核时会在提示信息前添加此前缀
        description_prefix="工具调用需要人工审批"
    )

    # 返回完整的工具列表和中间件
    return tools, hitl_middleware
