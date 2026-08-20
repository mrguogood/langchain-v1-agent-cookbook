# 导入 uvicorn，用于运行 FastAPI 服务
import uvicorn
# 导入 uuid 模块，用于生成全局唯一标识符（例如记忆 ID）
import uuid
# 导入 typing 模块中的类型提示工具，用于类型注解
from typing import Dict, Any, Optional
# FastAPI 核心框架导入
from fastapi import FastAPI, Request
# 实现 lifespan 的上下文管理器，用于管理应用生命周期
from contextlib import asynccontextmanager
# LangChain Agent 创建相关导入
from langchain.agents import create_agent
# 导入摘要中间件，用于在上下文过长时自动摘要历史消息
from langchain.agents.middleware import SummarizationMiddleware
# 导入提示词模板相关类，用于构建系统/用户提示
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# 导入异步 PostgreSQL 连接池
from psycopg_pool import AsyncConnectionPool
# 导入异步 PostgreSQL 检查点保存器（用于短期记忆/对话状态持久化）
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
# 导入异步 PostgreSQL 键值存储（用于长期记忆）
from langgraph.store.postgres import AsyncPostgresStore
# 导入工具调用结构化输出策略
from langchain.agents.structured_output import ToolStrategy
# 导入 LangGraph 中的 Command 类型，用于中断后恢复执行
from langgraph.types import Command
# 导入项目自定义配置、工具、模型、日志等模块
from utils.config import Config
from utils.llms import get_llm
from utils.tools import get_tools
from utils.models import Context, ResponseFormat
from utils.models import AskRequest, InterveneRequest, AgentResponse
from utils.logger import LoggerManager

# 使用 lifespan 管理应用生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 应用生命周期管理器：
      - 启动阶段：创建连接池、初始化 checkpointer 和 store
      - 运行阶段：yield 让 FastAPI 开始接受请求
      - 关闭阶段：清理资源（关闭连接池）
    """
    # 声明使用全局变量（在模块级别定义的 pool、checkpointer、store）
    global pool, checkpointer, store

    # 记录应用启动日志
    logger.info("应用正在启动... 初始化数据库资源")

    # 创建异步 PostgreSQL 连接池
    pool = AsyncConnectionPool(
        # 数据库连接字符串，从配置读取
        conninfo=Config.DB_URI,
        # 连接池最小连接数
        min_size=Config.MIN_SIZE,
        # 连接池最大连接数
        max_size=Config.MAX_SIZE,
        # 连接参数：启用自动提交，禁用预处理语句
        kwargs={"autocommit": True, "prepare_threshold": 0},
        # 启动时不立即打开（稍后手动打开）
        open=False
    )
    # 显式打开连接池
    await pool.open()

    # 创建短期记忆检查点保存器
    checkpointer = AsyncPostgresSaver(pool)
    # 初始化检查点所需的数据库表结构
    await checkpointer.setup()
    # 记录检查点初始化成功日志
    logger.info("短期记忆 Checkpointer 初始化成功")

    # 创建长期记忆键值存储器
    store = AsyncPostgresStore(pool)
    # 初始化存储器所需的数据库表结构
    await store.setup()
    # 记录长期记忆存储器初始化成功日志
    logger.info("长期记忆 store 初始化成功")

    logger.info(f"API接口服务启动成功")

    # ──────────────── 进入正常运行阶段，让 FastAPI 开始接收请求 ────────────────
    yield

    # ──────────────── 应用即将关闭，清理资源 ────────────────
    logger.info("应用正在关闭... 清理资源")
    # 如果连接池存在，则关闭它
    if pool is not None:
        await pool.close()
        # 记录连接池关闭日志
        logger.info("数据库连接池已关闭")


# 创建 FastAPI 应用实例，并绑定 lifespan 生命周期管理器
app = FastAPI(
    # API 标题
    title="ReAct Agent API",
    # API 描述
    description="带有 HITL 功能的智能体 API 接口服务",
    # 版本号
    version="1.0.0",
    # 绑定生命周期管理器
    lifespan=lifespan
)


# 获取项目统一的日志记录器实例
logger = LoggerManager.get_logger()

# 声明全局变量，用于在 lifespan 和路由函数之间共享数据库资源
pool: Optional[AsyncConnectionPool] = None
checkpointer: Optional[AsyncPostgresSaver] = None
store: Optional[AsyncPostgresStore] = None


# 内部辅助函数：读取指定用户的长期记忆内容
async def read_long_term_info(user_id: str) -> str:
    # 定义记忆的命名空间，通常为 ("memories", user_id)
    namespace = ("memories", user_id)

    # 在该命名空间下搜索所有记忆条目（不带语义过滤）
    memories = await store.asearch(namespace, query="")

    # 如果有记忆，则将每个记忆的 data 字段用空格拼接
    long_term_info = " ".join(
        [d.value["data"] for d in memories if isinstance(d.value, dict) and "data" in d.value]
    ) if memories else ""

    # 记录获取到的长期记忆长度日志
    logger.info(f"成功获取用户ID: {user_id} 的长期记忆，内容长度: {len(long_term_info)} 字符")

    # 返回拼接后的长期记忆文本
    return long_term_info


# 内部辅助函数：为指定用户写入一条长期记忆
async def write_long_term_info(user_id: str, memory_info: str) -> str:
    # 定义记忆存储的命名空间
    namespace = ("memories", user_id)

    # 为本次记忆生成一个随机 UUID 作为 key
    memory_id = str(uuid.uuid4())

    # 将记忆内容写入存储（value 包一层 dict，字段名为 data）
    await store.aput(
        namespace=namespace,
        key=memory_id,
        value={"data": memory_info}
    )

    # 记录写入成功的日志
    logger.info(f"成功为用户ID: {user_id} 存储记忆，记忆ID: {memory_id}")

    # 返回简单成功提示
    return "记忆存储成功"


# 内部函数：为当前请求创建一个独立的 Agent 实例
async def create_agent_instance() -> Any:
    # 根据配置获取聊天模型和嵌入模型
    llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

    # 获取可用工具列表以及 HITL 中间件实例
    tools, hitl_middleware = await get_tools()

    # 从文件读取系统提示词模板
    system_prompt = PromptTemplate.from_file(
        template_file=Config.SYSTEM_PROMPT_TMPL,
        encoding="utf-8"
    ).template

    # 使用 LangChain 的 create_agent 创建一个 Agent 实例
    agent = create_agent(
        # 使用的聊天大模型
        model=llm_chat,
        # 系统提示词，约束 Agent 行为
        system_prompt=system_prompt,
        # 可调用工具列表
        tools=tools,
        # 中间件列表：摘要 + 人工介入审核
        middleware=[
            # 上下文自动摘要中间件（token 超 4000 时触发，保留最后 3 条消息）
            SummarizationMiddleware(model=llm_chat, trigger=("tokens", 4000), keep=("messages", 3)),
            # 人工介入审核中间件
            hitl_middleware
        ],
        # 上下文结构定义（包含 user_id 等业务字段）
        context_schema=Context,
        # 结构化输出格式（支持从状态中读取 structured_response）
        response_format=ToolStrategy(ResponseFormat),
        # 短期记忆检查点
        checkpointer=checkpointer,
        # 长期记忆存储
        store=store
    )
    # 返回创建好的 Agent 实例
    return agent


# 核心运行函数：执行 Agent 并处理 HITL 中断逻辑
async def run_agent_with_hitl(agent: Any, user_content: str, config: dict, context: Context) -> Dict[str, Any]:

    # 第一次调用 Agent，传入用户消息
    result = await agent.ainvoke(
        # 消息列表，初始只有一条用户消息
        {"messages": [{"role": "user", "content": user_content}]},
        # 运行配置（包含 thread_id、user_id 等）
        config=config,
        # 上下文对象（包含 user_id 等信息）
        context=context,
    )

    # 判断是否出现人工介入中断
    if "__interrupt__" in result:
        # 取出中断请求（通常只有一个）
        hitl_requests = result["__interrupt__"]
        hitl_req = hitl_requests[0]

        # 返回中断状态及需要审核的工具调用细节
        return {
            "status": "interrupted",
            "interrupt_details": {
                # 待审核的工具调用请求列表
                "action_requests": hitl_req.value["action_requests"],
                # 每个工具对应的审核配置（允许的决策类型等）
                "review_configs": hitl_req.value["review_configs"]
            },
            # 保留中间结果（为后续扩展保留）
            "intermediate_result": result
        }
    else:
        # 无中断，视为正常完成，取出最后一条消息作为回答
        final_result = result["messages"][-1].content
        return {
            "status": "completed",
            "result": final_result
        }


# API 端点：接收用户问题并启动 Agent 执行
@app.post("/ask", response_model=AgentResponse)
async def ask(request: AskRequest):
    # 请求数据日志
    logger.info(f"/ask接口接收用户问题并启动 Agent 执行，用户ID： {request.user_id} 会话ID： {request.thread_id} 用户问题： {request.question}")

    # 为本次请求创建独立的 Agent 实例
    agent = await create_agent_instance()

    # 写入一条固定的长期记忆
    await write_long_term_info(request.user_id, "GCS")

    # 读取该用户的长期记忆内容（例如用户名、偏好等）
    name = await read_long_term_info(request.user_id)

    # 读取用户提示模板文件
    human_prompt = PromptTemplate.from_file(
        template_file=Config.HUMAN_PROMPT_TMPL,
        encoding="utf-8"
    ).template

    # 构建完整的聊天提示模板（system + human）
    chat_prompt = ChatPromptTemplate.from_messages([
        # 系统提示（角色、规则等）
        ("system", PromptTemplate.from_file(template_file=Config.SYSTEM_PROMPT_TMPL, encoding="utf-8").template),
        # 用户提示模板
        ("human", human_prompt)
    ])

    # 使用模板渲染实际的消息内容（替换占位符）
    messages = chat_prompt.format_messages(question=request.question, name=name)
    # 取出最后一条（即用户消息）
    human_msg = messages[-1]

    # 构造运行时配置（thread_id 和 user_id）
    config = {
        "configurable": {
            "thread_id": request.thread_id,
            "user_id": request.user_id,
        }
    }

    # 创建上下文对象
    context = Context(user_id=request.user_id)

    # 执行 Agent 并处理可能的 HITL 中断
    run_result = await run_agent_with_hitl(
        agent=agent,
        user_content=human_msg.content,
        config=config,
        context=context
    )

    # 根据执行结果返回不同响应
    if run_result["status"] == "completed":
        # 记录最终回答日志
        logger.info(f"Agent最终回复是: {run_result['result']}")
        # 直接返回完成结果
        return AgentResponse(**run_result)
    else:
        # 需要人工介入，直接返回中断信息
        return AgentResponse(
            status="interrupted",
            interrupt_details=run_result["interrupt_details"]
        )


# API 端点：人工提交决策，继续执行被中断的 Agent
@app.post("/intervene", response_model=AgentResponse)
async def intervene(request: InterveneRequest):
    # 请求数据日志
    logger.info(f"/intervene接口接收人工提交决策并继续执行被中断的 Agent，用户ID： {request.user_id} 会话ID： {request.thread_id} 人工决策反馈数据： {request.decisions}")

    # 为本次恢复创建一个新的 Agent 实例
    agent = await create_agent_instance()

    # 恢复时携带 thread_id 和 user_id（checkpointer 会自动加载历史状态）
    config = {
        "configurable": {
            "thread_id": request.thread_id,
            "user_id": request.user_id
        }
    }

    # 使用 Command.resume 携带人工决策继续执行
    result = await agent.ainvoke(
        # 恢复执行并传入人工决策
        Command(resume={"decisions": request.decisions}),
        # 配置（包含 thread_id 和 user_id）
        config=config,
        # 上下文对象
        context=Context(user_id=request.user_id)
    )


    # 检查是否还有新的中断（支持多轮审核）
    while "__interrupt__" in result:
        # 取出中断请求
        hitl_requests = result["__interrupt__"]
        hitl_req = hitl_requests[0]
        # 返回新的中断信息，让前端继续审核
        return AgentResponse(
            status="interrupted",
            interrupt_details={
                "action_requests": hitl_req.value["action_requests"],
                "review_configs": hitl_req.value["review_configs"]
            }
        )

    # 成功完成，取出最终回答
    final_result = result["messages"][-1].content
    # 记录最终回答日志
    logger.info(f"Agent最终回复是: {final_result}")

    # 返回完成结果
    return AgentResponse(status="completed", result=final_result)


# 主程序入口：使用 uvicorn 启动 FastAPI 服务
if __name__ == "__main__":
    import sys

    # Windows + Python 3.13 默认使用 ProactorEventLoop，与 psycopg 异步驱动不兼容
    # 方案：通过 uvicorn 的 loop 参数（支持 "module:attribute" 导入格式），
    # 指定 asyncio.SelectorEventLoop 作为事件循环工厂
    if sys.platform == "win32":
        uvicorn.run(
            app,
            host=Config.API_SERVER_HOST,
            port=Config.API_SERVER_PORT,
            loop="asyncio:SelectorEventLoop"
        )
    else:
        uvicorn.run(app, host=Config.API_SERVER_HOST, port=Config.API_SERVER_PORT)