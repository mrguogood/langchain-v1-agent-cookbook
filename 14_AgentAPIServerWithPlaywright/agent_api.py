# 导入 uvicorn，用于运行 FastAPI 服务
import uvicorn
# 导入 uuid 模块，用于生成全局唯一标识符（如记忆 ID）
import uuid
# 导入 json，用于将 SSE 事件序列化为 JSON 字符串
import json
# 导入 typing 中的类型提示，用于类型注解（含 AsyncGenerator）
from typing import List, Dict, Any, Optional, AsyncGenerator
# 导入 FastAPI 核心与 Request
from fastapi import FastAPI, Request
# 导入流式响应类，用于返回 SSE 流
from fastapi.responses import StreamingResponse
# 导入 Pydantic 的 BaseModel，用于定义请求/响应模型
from pydantic import BaseModel
# 导入异步上下文管理器，用于实现 lifespan
from contextlib import asynccontextmanager
# 导入 LangChain 的 create_agent，用于创建 Agent
from langchain.agents import create_agent
# 导入摘要中间件，在上下文过长时自动摘要历史消息
from langchain.agents.middleware import SummarizationMiddleware
# 导入提示词模板类，用于构建系统/用户提示
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# 导入异步 PostgreSQL 连接池
from psycopg_pool import AsyncConnectionPool
# 导入异步 PostgreSQL 检查点保存器（短期记忆/对话状态持久化）
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
# 导入异步 PostgreSQL 键值存储（长期记忆）
from langgraph.store.postgres import AsyncPostgresStore
# 导入工具调用结构化输出策略
from langchain.agents.structured_output import ToolStrategy
# 导入 LangGraph 的 Command，用于中断后携带决策恢复执行
from langgraph.types import Command
# 导入项目配置类
from utils.config import Config
# 导入获取 LLM 的函数
from utils.llms import get_llm
# 导入获取工具列表的函数与 Playwright 浏览器句柄管理
from utils.tools import get_tools, set_async_playwright_browser, clear_async_playwright_browser
# 导入上下文与响应格式模型
from utils.models import Context, ResponseFormat
# 导入请求/响应模型（AskRequest、InterveneRequest、AgentResponse）
from utils.models import AskRequest, InterveneRequest, AgentResponse
# 导入日志管理器
from utils.logger import LoggerManager
# 导入异步 Playwright 工具
import threading
from concurrent.futures import Future
from playwright.async_api import async_playwright

# 使用 asynccontextmanager 装饰器定义异步生命周期上下文
@asynccontextmanager
# 定义 lifespan 异步函数，参数为 FastAPI 应用实例
async def lifespan(app: FastAPI):
    """
    FastAPI 应用生命周期管理器：
      - 启动阶段：创建连接池、初始化 checkpointer 和 store
      - 运行阶段：yield 让 FastAPI 开始接受请求
      - 关闭阶段：清理资源（关闭连接池）
    """
    # 声明使用全局变量 pool、checkpointer、store、playwright_instance
    global pool, checkpointer, store, playwright_instance

    # 记录应用启动日志
    logger.info("应用正在启动... 初始化数据库资源")

    # 创建异步 PostgreSQL 连接池（conninfo、最小/最大连接数、kwargs、是否立即打开）
    pool = AsyncConnectionPool(
        conninfo=Config.DB_URI,
        min_size=Config.MIN_SIZE,
        max_size=Config.MAX_SIZE,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False
    )
    # 显式打开连接池
    await pool.open()

    # 使用连接池创建短期记忆检查点保存器
    checkpointer = AsyncPostgresSaver(pool)
    # 初始化检查点所需的数据库表
    await checkpointer.setup()
    # 记录检查点初始化成功
    logger.info("短期记忆 Checkpointer 初始化成功")

    # 使用连接池创建长期记忆键值存储器
    store = AsyncPostgresStore(pool)
    # 初始化存储器所需的数据库表
    await store.setup()
    # 记录存储器初始化成功
    logger.info("长期记忆 store 初始化成功")

    # 启动 Playwright 异步浏览器，供 LangChain PlayWrightBrowserToolkit 工具共享会话
    # 解决兼容性死锁：在 Windows + Python 3.13 上：
    #  - SelectorEventLoop: psycopg 需要它，但 SelectorEventLoop 在 Windows 不支持 create_subprocess_exec
    #  - ProactorEventLoop: Playwright 需要它来启动子进程，但 psycopg 不支持它
    # 方案：将 Playwright 初始化放到单独线程中，线程会创建独立的 ProactorEventLoop 来处理
    # 存储 Playwright 专属事件循环，用于在关闭时清理
    _playwright_loop: Optional[asyncio.AbstractEventLoop] = None

    def init_playwright() -> tuple[Any, Any] | tuple[None, None]:
        """在独立线程中初始化 Playwright，解决事件循环兼容性问题"""
        import asyncio
        import traceback
        nonlocal _playwright_loop
        try:
            # 在新线程中创建默认 ProactorEventLoop（Windows 默认）
            # ProactorEventLoop 支持 create_subprocess_exec，可正常启动 Playwright
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _playwright_loop = loop

            pw = loop.run_until_complete(async_playwright().start())
            browser = loop.run_until_complete(pw.chromium.launch(headless=Config.PLAYWRIGHT_HEADLESS))

            # 保持事件循环在后台运行，使 Playwright 能处理其内部消息
            threading.Thread(target=loop.run_forever, daemon=True).start()

            return pw, browser
        except Exception as e:
            logger.error(f"Playwright init 子线程异常: {e}\n{traceback.format_exc()}")
            if _playwright_loop is not None and not _playwright_loop.is_closed():
                _playwright_loop.close()
            _playwright_loop = None
            return None, None

    pw = None
    browser = None
    try:
        future: Future[tuple[Any, Any] | tuple[None, None]] = Future()
        thread = threading.Thread(target=lambda: future.set_result(init_playwright()), daemon=True)
        thread.start()
        result = future.result()
        pw, browser = result if result else (None, None)
        
        if pw is not None and browser is not None:
            # 将异步浏览器实例注册到工具层，供浏览器工具调用
            set_async_playwright_browser(browser)
            # 保存 Playwright 启动实例，供后续关闭使用
            playwright_instance = pw
            # 记录浏览器启动成功日志，包含 headless 配置信息
            logger.info(
                "Playwright 浏览器已启动（headless=%s）",
                Config.PLAYWRIGHT_HEADLESS,
            )
        else:
            raise RuntimeError("Playwright 初始化在子线程中返回 None")
    except Exception as e:
        # 若发生异常，记录浏览器初始化失败日志，并提示安装 playwright 及依赖
        logger.error(
            "Playwright 初始化失败，浏览器类工具将不可用（请安装 playwright 并执行 playwright install）：%s",
            e,
        )
        # 初始化失败，将 playwright_instance 置为 None
        playwright_instance = None
        # 若 pw 已创建则尝试关闭 Playwright 服务，避免资源泄露
        if pw is not None:
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                loop.run_until_complete(pw.stop())
            except Exception:
                pass
   

    # 记录 API 服务启动成功
    logger.info(f"API接口服务启动成功")

    # 进入运行阶段，将控制权交回 FastAPI，开始接收请求
    yield

    # 应用关闭阶段：记录正在关闭日志
    logger.info("应用正在关闭... 清理资源")
    # 先关闭 Playwright，再关闭数据库连接池
    # 如果 Playwright 实例不为 None，说明需要进行资源清理
    if playwright_instance is not None:
        # 调用 clear_async_playwright_browser 方法清除全局浏览器引用，并获取当前浏览器实例
        br = clear_async_playwright_browser()
        # 使用 Playwright 专属事件循环进行清理，避免在临时事件循环中遗留 pending 任务
        if _playwright_loop is not None and not _playwright_loop.is_closed():
            async def _cleanup_playwright():
                """在 Playwright 专属事件循环中执行关闭，确保所有后台任务被正确取消"""
                try:
                    if br is not None:
                        await br.close()
                    await playwright_instance.stop()
                except Exception:
                    pass
                finally:
                    # 取消所有遗留任务，避免 "Task was destroyed but it is pending!" 警告
                    for task in asyncio.all_tasks(_playwright_loop):
                        task.cancel()
                    if asyncio.all_tasks(_playwright_loop):
                        await asyncio.gather(*asyncio.all_tasks(_playwright_loop), return_exceptions=True)
                    # 停止后台事件循环
                    _playwright_loop.stop()
            # 在 Playwright 事件循环中调度清理任务，并等待完成
            future = asyncio.run_coroutine_threadsafe(_cleanup_playwright(), _playwright_loop)
            try:
                future.result(timeout=15)
            except Exception:
                pass
            # 关闭事件循环
            try:
                _playwright_loop.close()
            except Exception:
                pass
        # 将 playwright_instance 变量置为 None，确保引用被释放
        playwright_instance = None

    # 若连接池存在则关闭
    if pool is not None:
        await pool.close()
        logger.info("数据库连接池已关闭")


# 创建 FastAPI 应用实例，并绑定 lifespan
app = FastAPI(
    title="ReAct Agent API",
    description="带有 HITL、流式输出与 Playwright 浏览器工具的智能体 API 接口服务",
    version="1.0.0",
    lifespan=lifespan
)


# 获取项目统一日志记录器
logger = LoggerManager.get_logger()

# 声明全局变量：异步连接池（在 lifespan 与路由间共享）
pool: Optional[AsyncConnectionPool] = None
# 声明全局变量：检查点保存器
checkpointer: Optional[AsyncPostgresSaver] = None
# 声明全局变量：长期记忆存储器
store: Optional[AsyncPostgresStore] = None
# Playwright 驱动实例（用于应用关闭时 stop）
playwright_instance: Optional[Any] = None


# 定义异步函数：根据 user_id 读取该用户的长期记忆内容
async def read_long_term_info(user_id: str) -> str:
    # 定义记忆的命名空间，元组形式 ("memories", user_id)
    namespace = ("memories", user_id)

    # 在该命名空间下异步搜索，query 为空表示不过滤
    memories = await store.asearch(namespace, query="")

    # 若有记忆则用空格拼接每条记忆的 data 字段，否则为空字符串
    long_term_info = " ".join(
        [d.value["data"] for d in memories if isinstance(d.value, dict) and "data" in d.value]
    ) if memories else ""

    # 记录获取到的长期记忆长度
    logger.info(f"成功获取用户ID: {user_id} 的长期记忆，内容长度: {len(long_term_info)} 字符")

    # 返回拼接后的长期记忆文本
    return long_term_info


# 定义异步函数：为指定用户写入一条长期记忆
async def write_long_term_info(user_id: str, memory_info: str) -> str:
    # 定义记忆的命名空间
    namespace = ("memories", user_id)

    # 生成随机 UUID 作为该条记忆的 key
    memory_id = str(uuid.uuid4())

    # 异步写入存储：namespace、key、value（value 为含 data 字段的字典）
    await store.aput(
        namespace=namespace,
        key=memory_id,
        value={"data": memory_info}
    )

    # 记录写入成功日志
    logger.info(f"成功为用户ID: {user_id} 存储记忆，记忆ID: {memory_id}")

    # 返回成功提示字符串
    return "记忆存储成功"


# 定义异步函数：为当前请求创建并返回一个独立的 Agent 实例
async def create_agent_instance() -> Any:
    # 根据配置获取聊天模型与嵌入模型
    llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

    # 异步获取工具列表及 HITL 中间件
    tools, hitl_middleware = await get_tools()

    # 从文件读取系统提示词模板内容（.template 为原始字符串）
    system_prompt = PromptTemplate.from_file(
        template_file=Config.SYSTEM_PROMPT_TMPL,
        encoding="utf-8"
    ).template

    # 调用 create_agent 创建 Agent 实例
    agent = create_agent(
        model=llm_chat,
        system_prompt=system_prompt,
        tools=tools,
        middleware=[
            SummarizationMiddleware(model=llm_chat, trigger=("tokens", 4000), keep=("messages", 3)),
            hitl_middleware
        ],
        context_schema=Context,
        checkpointer=checkpointer,
        store=store
    )
    # 返回创建好的 Agent
    return agent


# 定义异步函数：非流式执行 Agent 并处理 HITL 中断，返回状态字典
async def run_agent_with_hitl(
    agent: Any, 
    user_content: str, 
    config: dict, 
    context: Context
) -> Dict[str, Any]:
    # 写入一条长期记忆到存储，实际项目可按需修改此处内容与 user_id
    await write_long_term_info("user_001", "gcs")

    # 调用 agent 的 ainvoke 方法，参数为初始消息、配置和上下文，得到运行结果
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": user_content}]},
        config=config,
        context=context,
    )

    # 判断返回结果中是否有 __interrupt__ 字段，表示出现人工审核中断
    if "__interrupt__" in result:
        # 提取所有中断请求（通常只有一个）
        hitl_requests = result["__interrupt__"]
        # 取中断请求列表中的第一个元素
        hitl_req = hitl_requests[0]

        # 返回中断状态及需要审核的 action_requests 和 review_configs，以及中间结果
        return {
            "status": "interrupted",
            "interrupt_details": {
                "action_requests": hitl_req.value["action_requests"],
                "review_configs": hitl_req.value["review_configs"]
            },
            "intermediate_result": result
        }
    else:
        # 如果没有中断，获取结果消息列表的最后一条内容作为最终输出
        final_result = result["messages"][-1].content
        # 返回已完成状态和最终输出内容
        return {
            "status": "completed",
            "result": final_result
        }


# 定义异步函数：流式执行 Agent，遇到 HITL 中断则 yield interrupted 并结束
async def run_agent_with_hitl_streaming(
    agent: Any,
    user_content: str,
    config: dict,
    context: Context,
) -> AsyncGenerator[Dict[str, Any], None]:
    # 使用 agent.astream 进行异步流式输出，含 updates 和 messages 两种流类型
    # 异步循环获取流式输出
    async for stream_mode, chunk in agent.astream(
        {"messages": [{"role": "user", "content": user_content}]},
        config=config,
        context=context,
        stream_mode=["updates", "messages"],
    ):
        # 对于 message 类型流（token/output）
        if stream_mode == "messages":
            # 解析 token 与元数据
            token, metadata = chunk
            # print(f"token: {token}")
            # print(f"metadata: {metadata}")
            # 尝试获取 token 的内容
            raw = getattr(token, "content", None)
            # print(f"raw: {raw}")
            # 如果内容为空，置空字符串
            if raw is None:
                raw = ""
            # 将最终内容转为字符串，并兜底空串
            content = raw or ""
            # 若为空串，则跳过本次循环
            if not content:
                continue
            # 取 node 名称以判断是否为工具节点输出
            node = (metadata or {}).get("langgraph_node") or ""
            # 如果是工具节点，输出工具返回，否则输出普通 token
            if node == "tools":
                yield {"type": "tool_output", "content": content}
            else:
                yield {"type": "token", "content": content}
        # 对于状态更新流（updates 类型）
        elif stream_mode == "updates":
            # 检查是否有人工中断（__interrupt__），如有则 yield 并提前返回
            if "__interrupt__" in chunk:
                hitl_requests = chunk["__interrupt__"]
                hitl_req = hitl_requests[0]
                yield {
                    "type": "interrupted",
                    "interrupt_details": {
                        "action_requests": hitl_req.value["action_requests"],
                        "review_configs": hitl_req.value["review_configs"],
                    },
                }
                return
    # 全部流程结束后，主动获取当前 agent state
    state = await agent.aget_state(config)
    # 如果 messages 存在，提取最后回复给前端
    if state and state.values and state.values.get("messages"):
        final_result = state.values["messages"][-1].content
        print(f"final_result: {final_result}")
        yield {"type": "completed", "result": final_result}
    # 否则返回空字符串结果，防止为 None
    else:
        yield {"type": "completed", "result": ""}


# 定义异步函数，用于恢复被中断的 Agent，并流式执行，遇到新的 HITL 时 yield interrupted 后结束
async def run_agent_resume_streaming(
    agent: Any,
    config: dict,
    context: Context,
    decisions: list,
) -> AsyncGenerator[Dict[str, Any], None]:
    # 调用 agent 的 astream 方法，参数为恢复执行的决策及配置等，监听 "updates" 和 "messages" 两种 stream_mode
    async for stream_mode, chunk in agent.astream(
        Command(resume={"decisions": decisions}),
        config=config,
        context=context,
        stream_mode=["updates", "messages"],
    ):
        # 如果流模式为“messages”，则处理普通文本流
        if stream_mode == "messages":
            # 解包 token 和 metadata
            token, metadata = chunk
            # print(f"token: {token}")
            # print(f"metadata: {metadata}")
            # 获取 token 的内容，取 content，如果都没有则为 None
            raw = getattr(token, "content", None)
            # print(f"raw: {raw}")
            # 如果内容为 None，则赋值为空字符串
            if raw is None:
                raw = ""
            # 最终内容转换为字符串，不存在则兜底为空字符串
            content = raw or ""
            # 若内容为空字符串，则跳过本次循环
            if not content:
                continue
            # 取 node 名判断是否为工具节点输出
            node = (metadata or {}).get("langgraph_node") or ""
            # 如果是工具节点，输出工具返回；否则输出普通 token
            if node == "tools":
                yield {"type": "tool_output", "content": content}
            else:
                yield {"type": "token", "content": content}
        # 如果流模式为“updates”，则检测是否有 HITL 中断请求
        elif stream_mode == "updates":
            # 检查 chunk 是否存在 "__interrupt__" 键，代表有人工中断请求
            if "__interrupt__" in chunk:
                # 获得 hitl 请求队列
                hitl_requests = chunk["__interrupt__"]
                # 取第一个请求对象
                hitl_req = hitl_requests[0]
                # yield 中断信号，包含详细信息
                yield {
                    "type": "interrupted",
                    "interrupt_details": {
                        "action_requests": hitl_req.value["action_requests"],
                        "review_configs": hitl_req.value["review_configs"],
                    },
                }
                # 提前 return，结束本轮流式执行
                return
    # 全部流程结束后，获取当前 agent 的 state
    state = await agent.aget_state(config)
    # 如果 state 存在且包含 messages，则返回最后一次消息的内容作为最终结果
    if state and state.values and state.values.get("messages"):
        final_result = state.values["messages"][-1].content
        yield {"type": "completed", "result": final_result}
    # 否则返回空字符串避免为 None
    else:
        yield {"type": "completed", "result": ""}


# 注册 POST /ask 端点：接收用户问题并同步执行 Agent，返回 AgentResponse
@app.post("/ask", response_model=AgentResponse)
async def ask(request: AskRequest):
    # 记录请求日志（用户 ID、会话 ID、问题）
    logger.info(f"/ask接口接收用户问题并启动 Agent 执行，用户ID： {request.user_id} 会话ID： {request.thread_id} 用户问题： {request.question}")

    # 为本次请求创建独立 Agent 实例
    agent = await create_agent_instance()

    # 读取该用户的长期记忆（如用户名等）
    name = await read_long_term_info(request.user_id)

    # 从文件读取用户提示模板内容
    human_prompt = PromptTemplate.from_file(
        template_file=Config.HUMAN_PROMPT_TMPL,
        encoding="utf-8"
    ).template

    # 构建聊天提示模板：system + human 两条
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", PromptTemplate.from_file(template_file=Config.SYSTEM_PROMPT_TMPL, encoding="utf-8").template),
        ("human", human_prompt)
    ])

    # 使用问题与 name 渲染消息列表
    messages = chat_prompt.format_messages(question=request.question, name=name)
    # 取最后一条作为用户消息
    human_msg = messages[-1]

    # 构造运行时 config（thread_id、user_id）
    config = {
        "configurable": {
            "thread_id": request.thread_id,
            "user_id": request.user_id,
        }
    }

    # 创建上下文对象
    context = Context(user_id=request.user_id)

    # 调用同步运行函数，得到 status + result 或 interrupt_details
    run_result = await run_agent_with_hitl(
        agent=agent,
        user_content=human_msg.content,
        config=config,
        context=context
    )

    # 若状态为已完成，记录日志并返回 AgentResponse
    if run_result["status"] == "completed":
        logger.info(f"Agent最终回复是: {run_result['result']}")
        return AgentResponse(**run_result)
    else:
        return AgentResponse(
            status="interrupted",
            interrupt_details=run_result["interrupt_details"]
        )


# 注册 POST /ask/stream 端点：流式执行 Agent，以 SSE 返回 token / completed / interrupted
@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    # 记录流式请求日志
    logger.info(f"/ask/stream 用户ID: {request.user_id} 会话ID: {request.thread_id} 问题: {request.question}")
    # 创建 Agent 实例
    agent = await create_agent_instance()
    # 读取长期记忆
    name = await read_long_term_info(request.user_id)
    # 读取用户提示模板
    human_prompt = PromptTemplate.from_file(
        template_file=Config.HUMAN_PROMPT_TMPL,
        encoding="utf-8"
    ).template
    # 构建 system+human 聊天模板
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", PromptTemplate.from_file(template_file=Config.SYSTEM_PROMPT_TMPL, encoding="utf-8").template),
        ("human", human_prompt)
    ])
    # 渲染消息并取最后一条用户消息
    messages = chat_prompt.format_messages(question=request.question, name=name)
    human_msg = messages[-1]
    # 构造 config
    config = {
        "configurable": {
            "thread_id": request.thread_id,
            "user_id": request.user_id,
        }
    }
    # 构造 context
    context = Context(user_id=request.user_id)

    # 定义异步生成器：将 run_agent_with_hitl_streaming 的每个事件转为 SSE 行
    async def event_generator():
        async for event in run_agent_with_hitl_streaming(agent, human_msg.content, config, context):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    # 返回流式响应：media_type 为 text/event-stream，并设置禁用缓存的头
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# 注册 POST /intervene 端点：提交人工决策，同步恢复执行并返回 AgentResponse
@app.post("/intervene", response_model=AgentResponse)
async def intervene(request: InterveneRequest):
    # 记录介入请求日志
    logger.info(f"/intervene接口接收人工提交决策并继续执行被中断的 Agent，用户ID： {request.user_id} 会话ID： {request.thread_id} 人工决策反馈数据： {request.decisions}")

    # 为本次恢复创建新的 Agent 实例
    agent = await create_agent_instance()

    # 恢复用的 config（thread_id、user_id，checkpointer 会据此加载状态）
    config = {
        "configurable": {
            "thread_id": request.thread_id,
            "user_id": request.user_id
        }
    }

    # 使用 ainvoke + Command.resume 携带决策继续执行
    result = await agent.ainvoke(
        Command(resume={"decisions": request.decisions}),
        config=config,
        context=Context(user_id=request.user_id)
    )

    # 若结果中仍有 __interrupt__，说明再次中断，返回新中断信息
    while "__interrupt__" in result:
        hitl_requests = result["__interrupt__"]
        hitl_req = hitl_requests[0]
        return AgentResponse(
            status="interrupted",
            interrupt_details={
                "action_requests": hitl_req.value["action_requests"],
                "review_configs": hitl_req.value["review_configs"]
            }
        )

    # 无新中断：取最后一条消息内容为最终回答
    final_result = result["messages"][-1].content
    logger.info(f"Agent最终回复是: {final_result}")
    return AgentResponse(status="completed", result=final_result)


# 注册 POST /intervene/stream 端点：提交决策后以 SSE 流式返回恢复后的输出
@app.post("/intervene/stream")
async def intervene_stream(request: InterveneRequest):
    # 记录流式恢复请求日志
    logger.info(f"/intervene/stream 用户ID: {request.user_id} 会话ID: {request.thread_id} 决策数: {len(request.decisions)}")
    # 创建 Agent 实例
    agent = await create_agent_instance()
    # 构造 config
    config = {
        "configurable": {
            "thread_id": request.thread_id,
            "user_id": request.user_id,
        }
    }
    # 构造 context
    context = Context(user_id=request.user_id)

    # 定义异步生成器：将 run_agent_resume_streaming 的每个事件转为 SSE 行
    async def event_generator():
        async for event in run_agent_resume_streaming(agent, config, context, request.decisions):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    # 返回 SSE 流式响应
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# 主程序入口：使用 uvicorn 启动 FastAPI 服务
if __name__ == "__main__":
    import asyncio
    import selectors
    import sys

    # Windows + Python 3.13 + psycopg 3.x 兼容性问题：
    #  psycopg 异步驱动要求使用 SelectorEventLoop(selectors.SelectSelector())
    #  Playwright 的启动已通过独立线程解决（线程内使用默认 ProactorEventLoop）
    # 主事件循环只需满足 psycopg 的要求即可
    if sys.platform == "win32":
        def selector_event_loop_factory():
            return asyncio.SelectorEventLoop(selector=selectors.SelectSelector())

        uvicorn.run(
            app,
            host=Config.API_SERVER_HOST,
            port=Config.API_SERVER_PORT,
            loop=selector_event_loop_factory
        )
    else:
        uvicorn.run(app, host=Config.API_SERVER_HOST, port=Config.API_SERVER_PORT)