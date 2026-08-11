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
# ChatPromptTemplate 用于构建多轮对话风格的提示模板,支持 system/human 等多种角色消息组合
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# 从 LangGraph 导入内存检查点存储器，用于短期记忆与会话状态持久化
from langgraph.checkpoint.postgres import PostgresSaver
# 从 langgraph 的 Postgres 存储模块中导入 PostgresStore 类，用于把长期记忆/状态持久化到 PostgreSQL 数据库中
from langgraph.store.postgres import PostgresStore
# 从 LangChain 导入 ToolStrategy，用于指定代理使用“工具调用”的结构化输出格式
from langchain.agents.structured_output import ToolStrategy
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

# 获取全局日志记录器，用于输出运行过程中的日志信息
logger = LoggerManager.get_logger()

# 根据配置中指定的 LLM 类型，获取对话模型 llm_chat 和嵌入模型 llm_embedding 实例
llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

# 获取当前智能体可用的工具列表和HITL中间件实例
tools, hitl_middleware = get_tools()

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

# # 打印加载到的系统提示词模板内容,方便调试和确认是否读取正确
# print(f'system_prompt_tmpl:{system_prompt} \n')
# # 打印加载到的用户提示词模板内容,确认 human prompt 文件内容是否正确
# print(f'human_prompt:{human_prompt} \n')

# 使用 ChatPromptTemplate.from_messages 构建一个聊天提示模板
# 其中包含一条 system 消息和一条 human 消息:
# - system 使用上面加载的系统提示词模板,用于定义 Agent 的角色与规则
# - human 使用上面加载的用户提示词模板,用于定义用户提问的表达方式
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt)
])

# 封装一个带 HITL 审核的调用函数，问答统一用它
# 定义一个带有人在环路（HITL）机制的运行函数
# agent：代理对象；user_content：用户输入内容；
# config：运行配置字典；context：上下文对象
def run_with_hitl_stream(agent, user_content: str, config: dict, context: Context):
    # 使用 stream 实时打印 LLM token，同时支持多轮 HITL 中断
    # 定义一个内部函数，用于执行一次流式调用
    # 既可以用于“首次从用户输入开始”，也可以用于“带 Command 恢复后继续执行”
    def _stream_once(input_):
        # 调用代理的流式接口，按指定模式逐步返回数据
        for mode, chunk in agent.stream(
            input_,
            config=config,
            stream_mode=["updates", "messages"],
            context=context,
        ):
            # 如果当前返回类型是消息片段（LLM token）
            if mode == "messages":
                # chunk 结构为 (token, metadata)，这里进行解包
                token, metadata = chunk
                # 如果 token 中有内容（有可能是空，如纯控制信号）
                if token.content:
                    # 将 token 内容实时输出到终端（或你的前端），不换行
                    print(token.content, end="", flush=True)
            # 如果当前返回类型是状态更新（包含中断等信息）
            elif mode == "updates":
                # 状态更新中如果包含 "__interrupt__"，说明发生中断
                if "__interrupt__" in chunk:
                    # 打印中断提示
                    print("\n\n=== 检测到中断 ===")
                    # 打印完整的中断内容，方便调试和查看
                    print(f"{chunk['__interrupt__']}")
                    # 把包含中断信息的整个状态返回给外层逻辑处理
                    return chunk
        # 如果循环正常结束，说明没有中断，chunk 即为最终状态
        # 将最终状态返回
        return chunk

    # 1) 第一次流式执行：从用户的问题开始进行流式对话
    first_chunk = _stream_once(
        # 将用户输入封装为 messages 列表，角色为 user
        {"messages": [{"role": "user", "content": user_content}]}
    )

    # 如果第一次执行结果中没有中断字段，说明流程已经顺利结束
    if "__interrupt__" not in first_chunk:
        # 直接返回最终结果，不需要进入 HITL 循环
        return first_chunk

    # 2) 如果存在中断，则开始循环处理所有后续中断
    # 支持像 get_location -> get_weather 这样的工具链式中断场景
    result = first_chunk
    while "__interrupt__" in result:
        # 从结果中取出所有中断请求列表
        hitl_requests = result["__interrupt__"]
        # 简化处理：只取第一个中断请求进行处理
        hitl_req = hitl_requests[0]
        # 从中断请求中获取所有需要人工审核的工具调用
        action_requests = hitl_req.value["action_requests"]
        # 获取每个工具调用对应的审核配置（可选决策等）
        review_configs = hitl_req.value["review_configs"]

        # 初始化一个列表，用于收集本轮所有工具调用的人工决策
        decisions = []
        # 遍历每个工具调用请求，i 为序号，ar 为具体请求内容
        for i, ar in enumerate(action_requests):
            # 从请求中读取工具名称
            name = ar.get("name")
            # 兼容两种参数字段：优先使用 args，没有则退回到 arguments
            args = ar.get("args", ar.get("arguments"))

            # 从对应的审核配置中取出当前工具允许的决策类型
            allowed = review_configs[i]["allowed_decisions"]

            # 打印需要人工审核的提示信息
            print("\n=== 工具调用需要人工审核 ===")
            # 打印工具名称
            print(f"工具名：{name}")
            # 打印工具调用参数
            print(f"参数：{args}")
            # 打印允许的决策类型列表
            print(f"允许的决策类型：{allowed}")

            # 提示人工输入对该工具调用的决策，并去掉前后空白
            decision_type = input("请输入决策(approve/edit/reject)：").strip()
            # 如果输入不在允许列表中，则循环要求重新输入
            while decision_type not in allowed:
                decision_type = input(
                    f"非法决策，请重新输入({','.join(allowed)})："
                ).strip()

            # 如果选择 edit，表示需要修改工具调用参数后再让代理执行
            if decision_type == "edit":
                # 提示输入修改后的参数，要求为 JSON 字符串格式
                new_args_str = input("请输入修改后的 args(JSON 字符串)：").strip()
                # 将 JSON 字符串解析为 Python 字典对象
                new_args = json.loads(new_args_str)

                # 构造编辑后的动作对象：保持 name 不变，仅更新 args 字段
                edited_action = {
                    "name": name,
                    "args": new_args,
                }
                # 将编辑类型的决策和编辑后的动作加入决策列表
                decisions.append({
                    "type": "edit",
                    "edited_action": edited_action,
                })
            else:
                # 对于 approve 或 reject 等非编辑类型，只记录决策类型
                decisions.append({"type": decision_type})

        # 3) 携带本轮人工决策信息，使用恢复指令再次开启一轮流式执行
        print("\n=== 带人工决策继续执行 ===")
        result = _stream_once(
            # 使用恢复命令，将 decisions 传回给代理，继续后续流程
            Command(resume={"decisions": decisions})
        )

    # 当跳出 while 循环时，说明所有中断都已处理完毕，不再有新的中断
    # 此时 result 即为最后一轮流式执行结束时的最终状态
    return result

# 创建一个基于数据库的检查点存储器，用于保存对话状态，实现短期记忆与多轮会话关联
with (
    PostgresStore.from_conn_string(Config.DB_URI) as store,
    PostgresSaver.from_conn_string(Config.DB_URI) as checkpointer
):
    # 初始化检查点保存器的数据库表结构
    checkpointer.setup()

    # 初始化长期记忆的数据库表结构
    store.setup()

    # 使用 LangChain 的 create_agent 创建一个 Agent 实例
    # - model: 指定使用的对话 LLM 模型
    # - system_prompt: 指定系统级提示词，约束 Agent 行为
    # - tools: 传入可供 Agent 调用的工具列表
    # - middleware：传入SummarizationMiddleware中间件进行消息修剪,hitl_middleware中间件用于人工介入审核
    # - context_schema: 指定上下文对象的 Pydantic（或类似）schema，用于扩展状态信息（如 user_id）
    # - response_format: 使用 ToolStrategy + ResponseFormat 定义结构化输出格式，支持从 Agent 状态中读取 structured_response 字段
    # - checkpointer: 传入 PostgresSaver，使 Agent 具备按线程维度存储和恢复对话状态的能力
    # - store: 传入 PostgresStore，使 Agent 具备按用户维度存储和查找长期记忆(如用户偏好设置)的能力
    agent = create_agent(
        model=llm_chat,
        system_prompt=system_prompt,
        tools=tools,
        middleware=[
            SummarizationMiddleware(model=llm_chat, trigger=("tokens", 4000), keep=("messages", 3)),
            hitl_middleware
        ],
        context_schema=Context,
        response_format=ToolStrategy(ResponseFormat),
        checkpointer=checkpointer,
        store=store
    )

    # 读取长期记忆内容
    def read_long_term_info(user_id: str):
        # 定义用于查询的命名空间，通常用 (类别, 用户ID) 这种形式做分区
        namespace = ("memories", user_id)

        # 在指定命名空间下搜索所有记忆数据，这里 query="" 意味着不带语义过滤，取全部或由实现决定
        memories = store.search(namespace, query="")

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
    def write_long_term_info(user_id: str, memory_info: str):
        # 定义命名空间，用于把某个用户的记忆归到 ("memories", user_id) 这个层级路径下
        namespace = ("memories", user_id)

        # 生成一个全局唯一的记忆ID，作为该条记忆在命名空间内的 key
        memory_id = str(uuid.uuid4())

        # 调用存储接口，将记忆写入存储：
        # - namespace: 命名空间，用于分层组织数据
        # - key: 该命名空间下唯一的主键（这里用随机UUID）
        # - value: 实际存储内容，这里用 dict 包一层，字段名为 "data"
        result = store.put(
            namespace=namespace,
            key=memory_id,
            value={"data": memory_info}
        )

        # 记录日志，说明为该用户成功写入了一条长期记忆，并打印记忆ID，便于排查
        logger.info(f"成功为用户ID: {user_id} 存储记忆，记忆ID: {memory_id}")

        # 返回给上层一个简单的成功提示文案
        return "记忆存储成功"

    # 写入长期记忆
    write_long_term_info("user_001", "gcs")

    # 定义调用配置，其中 configurable.thread_id 用于标识一段对话的唯一“线程 ID”
    # configurable.user_id 用于标识唯一“用户 ID”
    # 不同 thread_id 之间状态隔离，相同 thread_id 则共享对话上下文与短期记忆
    # 不同 user_id 之间数据隔离，相同 user_id 则共享长期记忆
    config = {
        "configurable": {
            "thread_id": "01",
            "user_id": "user_001",
        }
    }
    # 定义原始用户问题
    raw_question = "今天天气怎么样？"
    # 获取长期记忆内容，用户偏好设置(用户名称)
    name = read_long_term_info("user_001")

    # 使用 chat_prompt.format_messages 方法,将模板中的 {question}、{name} 等占位符
    # 替换为实际变量,生成一组完整的对话消息列表(messages)
    messages = chat_prompt.format_messages(question=raw_question, name=name)
    # 取出消息列表中的最后一条消息,通常对应 user(用户)消息,作为本轮要发送给 Agent 的用户提示
    human_msg = messages[-1]
    # 打印最终生成的人类提示内容,便于调试查看模板渲染后的实际文案
    print(f'用户的问题是: {human_msg.content} \n')
    # 将用户提示内容写入日志,方便后续排查问题或重现对话
    logger.info(f"用户的问题是: {human_msg.content}")

    # 调用 Agent 进行对话
    # - messages: 传入用户消息列表，这里用户问“外面的天气怎么样？”
    # - config: 传入包含 thread_id 的配置，用于绑定会话上下文
    # - context: 传入自定义的 Context 对象（如包含 user_id 等业务相关信息）
    # 使用带 HITL 审核的封装函数 run_with_hitl_stream
    response = run_with_hitl_stream(
        agent=agent,
        user_content=human_msg.content,
        config=config,
        context=Context(user_id="user_001")
    )