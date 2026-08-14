# 导入操作系统模块，用于设置和读取环境变量
import os
import asyncio
# 导入 json 模块用于解析 JSON 字符串
import json
# 导入 Python 标准库的 uuid 模块，用来生成全局唯一ID
import uuid
# 从 LangChain 导入 create_agent 方法，用于创建智能体（Agent）
from langchain.agents import create_agent
# 这是一个“摘要中间件”，用于在对话过长时，
# 自动用聊天模型对早期消息做摘要并替换原始消息，
# 以此控制上下文长度、同时尽量保留历史关键信息
from langchain.agents.middleware import SummarizationMiddleware
# 从 langchain_core.prompts 模块中导入
# PromptTemplate 用于构建单条文本提示模板,通过占位符+format 的方式动态生成提示词
from langchain_core.prompts import PromptTemplate
# 导入异步PostgreSQL连接池
from psycopg_pool import AsyncConnectionPool
# 导入异步PostgreSQL检查点保存器，用于保存智能体状态
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
# 导入异步PostgreSQL存储器，用于长期记忆存储
from langgraph.store.postgres import AsyncPostgresStore
# 从 LangChain 导入 ToolStrategy，用于指定代理使用“工具调用”的结构化输出格式
from langchain.agents.structured_output import ToolStrategy
# 从Langfuse的LangChain集成模块导入CallbackHandler类
from langfuse.langchain import CallbackHandler
# 从 langfuse 包中导入 Langfuse 客户端类，以及 propagate_attributes 上下文管理器
# propagate_attributes 用于在一个代码块中批量“传播”统一的属性，比如 user_id、session_id、tags、metadata 等，让该代码块里新产生的所有观测都继承这些属性 [web:26][web:28]
from langfuse import Langfuse, propagate_attributes
# 从 LangChain 导入 ChatPromptTemplate，用于构建聊天型提示模板
from langchain_core.prompts import ChatPromptTemplate
# Command 用于在中断后携带决策恢复执行
from langgraph.types import Command
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

async def main():

    # 设置 Langfuse 相关环境变量，用于观测与调试 Agent 执行过程
    os.environ["LANGFUSE_SECRET_KEY"] = Config.LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_PUBLIC_KEY"] = Config.LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_BASE_URL"] = Config.LANGFUSE_BASE_URL
    os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = Config.LANGFUSE_TRACING_ENVIRONMENT

    # 初始化 Langfuse 客户端实例，用于上报 trace、span、generation 以及与 Langfuse 交互
    langfuse = Langfuse()

    # 为 LangChain 初始化 Langfuse CallbackHandler（用于追踪/跟踪）回调处理器
    # LangChain 有自己的一套回调系统，而 Langfuse 正是通过监听这些回调，来记录你的 chains 和 LLM 在做什么
    langfuse_handler = CallbackHandler()

    # 获取全局日志记录器，用于输出运行过程中的日志信息
    logger = LoggerManager.get_logger()

    # 根据配置中指定的 LLM 类型，获取对话模型 llm_chat 和嵌入模型 llm_embedding 实例
    llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

    # 获取当前智能体可用的工具列表和HITL中间件实例
    tools, hitl_middleware = await get_tools()

    # 从 Langfuse 获取名为 system_prompt 的文本类型 prompt，并指定使用 production 标签版本
    langfuse_system_prompt = langfuse.get_prompt(
        name="gcs_agi/agent_rag/system_prompt",
        type="text",
        label="production",
        cache_ttl_seconds=300
    )
    # 打印系统 prompt 的原始内容，方便调试和确认是否读取正确
    # print(f'系统Prompt:{langfuse_system_prompt.prompt} \n')
    logger.info(f'系统Prompt:{langfuse_system_prompt.prompt}')

    # 从 Langfuse 获取名为 human_prompt 的文本类型 prompt，同样使用 production 标签版本
    langfuse_human_prompt = langfuse.get_prompt(
        name="gcs_agi/agent_rag/human_prompt",
        type="text",
        label="production",
        cache_ttl_seconds=300
    )
    # 获取prompt中的config中结构化输出的schema
    cfg = langfuse_human_prompt.config
    response_format = cfg.get("response_format")
    # print(f'结构化输出的schema:{response_format} \n')
    logger.info(f'结构化输出的schema:{response_format}')

    # 将 Langfuse 的 human prompt 转换为 LangChain 的 PromptTemplate，以便在链或 Agent 中复用
    langchain_human_prompt = PromptTemplate.from_template(
        langfuse_human_prompt.get_langchain_prompt()
    )
    # 打印生成的 LangChain PromptTemplate 对象，检查模板结构是否符合预期
    # print(f'用户Prompt:{langchain_human_prompt} \n')
    logger.info(f'用户Prompt:{langchain_human_prompt}')

    # 封装一个带 HITL 审核的调用函数，问答统一用它
    # 定义一个带有人在环路（HITL）机制的运行函数
    # agent：代理对象；user_content：用户输入内容；
    # config：运行配置字典；context：上下文对象
    async def run_with_hitl_invoke(agent, user_content: str, config: dict, context: Context):
        # 第一次调用代理，发送用户消息，发起对话/执行
        result = await agent.ainvoke(
            # 将用户输入包装为消息列表，角色为 user，内容为 user_content
            {"messages": [{"role": "user", "content": user_content}]},
            # 传入配置，用于控制代理行为
            config=config,
            # 传入上下文对象，维持对话或执行环境
            context=context,
        )

        # 使用循环持续处理所有中断情况
        # 只要返回结果中包含 "__interrupt__" 字段，就说明有人工审核步骤需要处理
        # 支持同时有多个工具需要审批，也支持工具链式调用，如先 get_location 再 get_weather
        while "__interrupt__" in result:
            # 从结果中取出所有中断请求列表
            hitl_requests = result["__interrupt__"]
            # 简化处理：只取第一个中断请求
            hitl_req = hitl_requests[0]
            # 从中断请求中取出所有需要人工审核的工具调用请求
            action_requests = hitl_req.value["action_requests"]
            # 从中断请求中取出每个工具调用对应的审核配置
            review_configs = hitl_req.value["review_configs"]

            # 初始化一个决策列表，用来收集对每个工具调用的人工决策
            decisions = []
            # 遍历所有工具调用请求，i 为索引，ar 为单个请求
            for i, ar in enumerate(action_requests):
                # 从请求中取得工具名称
                name = ar.get("name")
                # 兼容两种参数字段写法：优先从 "args" 取，如果没有再从 "arguments" 取
                args = ar.get("args", ar.get("arguments"))

                # 从对应的审核配置中获取当前工具允许的决策类型
                allowed = review_configs[i]["allowed_decisions"]

                # 在控制台提示检测到需要人工审核的工具调用
                print("\n=== 检测到工具调用需要审核 ===")
                # 打印工具名称
                print(f"工具名：{name}")
                # 打印工具调用参数
                print(f"参数：{args}")
                # 打印允许的决策类型列表
                print(f"允许的决策类型：{allowed}")

                # 提示人工输入决策类型，并去掉前后空格
                decision_type = input("请输入决策(approve/edit/reject)：").strip()
                # 如果输入不在允许的决策列表中，则循环要求重新输入
                while decision_type not in allowed:
                    decision_type = input(f"非法决策，请重新输入({','.join(allowed)})：").strip()

                # 如果决策为 edit，表示需要修改工具调用的参数
                if decision_type == "edit":
                    # 让用户输入修改后的参数，要求为 JSON 字符串格式
                    new_args_str = input("请输入修改后的 args(JSON 字符串)：").strip()
                    # 将 JSON 字符串解析为 Python 字典对象
                    new_args = json.loads(new_args_str)

                    # 构造编辑后的工具调用动作，只允许修改 args，保持 name 不变
                    edited_action = {
                        "name": name,
                        "args": new_args,
                    }
                    # 将编辑类型的决策及编辑后的动作加入决策列表
                    decisions.append({
                        "type": "edit",
                        "edited_action": edited_action,
                    })
                else:
                    # 对于 approve 或 reject 等非编辑类决策，仅记录决策类型
                    decisions.append({"type": decision_type})

            # 带着上一步收集的 decisions 信息恢复代理执行
            # Command(resume=...) 告诉代理根据这些人工决策继续往下跑
            result = await agent.ainvoke(
                Command(resume={"decisions": decisions}),
                config=config,
                context=context,
            )
        # 当跳出 while 循环时，说明已经没有新的中断需要审核
        # 此时 result 即为整个流程的最终结果，直接返回
        return result

    # 创建异步PostgreSQL连接池，动态连接池根据负载调整连接池大小
    async with AsyncConnectionPool(
            # 数据库连接字符串
            conninfo=Config.DB_URI,
            # 连接池最小连接数
            min_size=Config.MIN_SIZE,
            # 连接池最大连接数
            max_size=Config.MAX_SIZE,
            # 连接参数：启用自动提交和禁用预处理语句
            kwargs={"autocommit": True, "prepare_threshold": 0},
            # 延迟打开连接池
            open=False
    ) as pool:
        # 创建短期记忆检查点保存器实例
        checkpointer = AsyncPostgresSaver(pool)
        # 初始化检查点保存器的数据库表结构
        await checkpointer.setup()
        # 记录检查点保存器初始化成功日志
        logger.info("短期记忆Checkpointer初始化成功")

        # 创建长期记忆存储器实例
        store = AsyncPostgresStore(pool)
        # 初始化存储器的数据库表结构
        await store.setup()
        # 记录存储器初始化成功日志
        logger.info("长期记忆store初始化成功")

        # 使用 LangChain 的 create_agent 创建一个 Agent 实例
        # - model: 指定使用的对话 LLM 模型
        # - system_prompt: 指定系统级提示词，约束 Agent 行为
        # - tools: 传入可供 Agent 调用的工具列表
        # - middleware：传入SummarizationMiddleware中间件进行消息修剪,hitl_middleware中间件用于人工介入审核
        # - context_schema: 指定上下文对象的 Pydantic（或类似）schema，用于扩展状态信息（如 user_id）
        # - response_format: 使用 ToolStrategy + ResponseFormat 定义结构化输出格式，支持从 Agent 状态中读取 structured_response 字段
        # - checkpointer: 传入 AsyncPostgresSaver，使 Agent 具备按线程维度存储和恢复对话状态的能力
        # - store: 传入 AsyncPostgresStore，使 Agent 具备按用户维度存储和查找长期记忆(如用户偏好设置)的能力
        agent = create_agent(
            model=llm_chat,
            system_prompt=langfuse_system_prompt.prompt,
            tools=tools,
            middleware=[
                SummarizationMiddleware(model=llm_chat, trigger=("tokens", 4000), keep=("messages", 3)),
                hitl_middleware
            ],
            context_schema=Context,
            response_format=ToolStrategy(response_format),
            # response_format=ToolStrategy(ResponseFormat),
            checkpointer=checkpointer,
            store=store
        )

        # 用户ID
        user_id = "user_0001"
        # 线程ID(会话ID)
        thread_id = "session_0001"

        # 读取长期记忆内容
        async def read_long_term_info(user_id: str):
            # 定义用于查询的命名空间，通常用 (类别, 用户ID) 这种形式做分区
            namespace = ("memories", user_id)

            # 在指定命名空间下搜索所有记忆数据，这里 query="" 意味着不带语义过滤，取全部或由实现决定
            memories = await store.asearch(namespace, query="")

            # 如果没有查到任何结果（返回 None），这里先占位，不做额外处理
            if memories is None:
                pass

            # 如果有 memories，则从每个条目中提取 value["data"] 字段，并用空格拼接成一个长字符串；
            # 条件判断确保 d.value 是字典且包含 "data" 键；
            # 如果 memories 为 None 或空列表，则整体结果为 ""（空字符串）
            long_term_info = " ".join(
                [d.value["data"] for d in memories if isinstance(d.value, dict) and "data" in d.value]
            ) if memories else ""

            # 打日志，记录成功获取到的用户长期记忆，以及拼接后的文本长度，方便排查与监控
            logger.info(f"成功获取用户ID: {user_id} 的长期记忆，内容长度: {len(long_term_info)} 字符，内容: {long_term_info} 字符")

            # 返回拼接好的长期记忆文本
            return long_term_info

        # 写入指定用户的长期记忆
        async def write_long_term_info(user_id: str, memory_info: str):
            # 定义命名空间，用于把某个用户的记忆归到 ("memories", user_id) 这个层级路径下
            namespace = ("memories", user_id)

            # 生成一个全局唯一的记忆ID，作为该条记忆在命名空间内的 key
            memory_id = str(uuid.uuid4())

            # 调用存储接口，将记忆写入存储：
            # - namespace: 命名空间，用于分层组织数据
            # - key: 该命名空间下唯一的主键（这里用随机UUID）
            # - value: 实际存储内容，这里用 dict 包一层，字段名为 "data"
            result = await store.aput(
                namespace=namespace,
                key=memory_id,
                value={"data": memory_info}
            )

            # 记录日志，说明为该用户成功写入了一条长期记忆，并打印记忆ID，便于排查
            logger.info(f"成功为用户ID: {user_id} 存储记忆，记忆ID: {memory_id}")

            # 返回给上层一个简单的成功提示文案
            return "记忆存储成功"

        # 写入长期记忆
        await write_long_term_info(user_id, "gcs")

        # 定义调用配置，其中 configurable.thread_id 用于标识一段对话的唯一“线程 ID”
        # configurable.user_id 用于标识唯一“用户 ID”
        # 不同 thread_id 之间状态隔离，相同 thread_id 则共享对话上下文与短期记忆
        # 不同 user_id 之间数据隔离，相同 user_id 则共享长期记忆
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
            },
            # 将 Langfuse 的回调处理器接入 LangChain 的事件系统
            "callbacks": [langfuse_handler],
            "metadata": {"langfuse_prompt": langfuse_human_prompt}
        }
        # 定义原始用户问题
        raw_question = "北京天气怎么样？"
        # raw_question = "杭州的天气怎么样？"
        # 获取长期记忆内容，用户偏好设置(用户名称)
        name = await read_long_term_info(user_id)
        # 获取用户prompt内容 动态传参
        query = langchain_human_prompt.format(question=raw_question, name=name)
        # 打印最终生成的人类提示内容,便于调试查看模板渲染后的实际文案
        print(f'用户的问题是: {query} \n')
        # 将用户提示内容写入日志,方便后续排查问题或重现对话
        logger.info(f"用户的问题是: {query}")

        # 创建根观测（命名追踪入口）
        # 使用 Langfuse 的上下文管理器启动一个观测（observation），类型为 span，名称为 “gcs_agent”
        with langfuse.start_as_current_observation(
                as_type="span",
                name="gcs_agent"
        ) as root_span:
            # 在根 span 对象中设置 trace 级别的属性，便于后续追踪、分析与评估
            root_span.update_trace(
                # 用户标识，用于将数据与具体用户关联
                user_id=user_id,
                # 会话标识，用于区分不同对话线程
                session_id=thread_id,
                # 为此次追踪添加标签，方便在 Langfuse 平台中筛选
                tags=["https://gcsai.top", "gcs测试"],
                # 元数据包含组织信息、作者及提示词版本等，用于跟踪模型配置
                metadata={
                    "organization": "gcs测试",
                    "author": "gcs",
                    "system_prompt_name": "gcs_agi/agent_rag/system_prompt",
                    "system_prompt_version": langfuse_system_prompt.version,
                    "human_prompt_name": "gcs_agi/agent_rag/human_prompt",
                    "human_prompt_version": langfuse_human_prompt.version
                },
                # 追踪版本号，便于维护与更新
                version="1.0",
                # 输入信息，包括用户查询内容与用户名，用于分析上下文
                input={"question": query, "name": "gcs"}
            )

            # 设置属性传播（影响 Agent 内部所有子调用）
            # 使用 propagate_attributes 确保所有子观测继承 Trace 属性（用户ID、会话ID、标签等）
            with propagate_attributes(
                    user_id=user_id,
                    session_id=thread_id,
                    tags=["https://gcsai.top", "gcs测试"],
                    metadata={"organization": "gcs测试", "author": "gcs"},
                    version="1.0"
            ):
                # 调用带人工审核（HITL）的封装函数执行 Agent 对话流程
                # 参数说明：
                # - agent: 预配置的 AI 代理实例
                # - user_content: 用户输入的查询内容
                # - config: 包含 thread_id 的配置，用于维护会话上下文
                # - context: 自定义 Context 对象，携带用户业务信息
                response = await run_with_hitl_invoke(
                    agent=agent,
                    user_content=query,
                    config=config,
                    context=Context(user_id=user_id)
                )

            # 从 response 字典中取出 "messages" 列表的最后一条消息的内容，作为代理（Agent）的最终回复结果
            result = response["messages"][-1].content
            # 在控制台打印代理最终回复内容，方便调试和查看
            print(f"Agent最终回复是: {result} \n")
            # 通过日志记录器写入一条信息级别的日志，内容为代理最终回复，方便后续排查和追踪
            logger.info(f"Agent最终回复是: {result}")

            # 更新 trace 的输出字段，确保评估器能获取完整的输入输出对
            # Agent 生成的最终答案（用于后续评估）
            # 本次对话的消息总数
            root_span.update_trace(
                output={
                    "answer": result
                }
            )

            # 将完整对话历史记录到根 span 的 metadata 中，便于后续分析完整上下文
            root_span.update(
                metadata={
                    "full_conversation": [
                        {"role": msg.type, "content": msg.content}
                        for msg in response["messages"]
                    ]
                }
            )


# 主程序入口
if __name__ == "__main__":
    asyncio.run(main())