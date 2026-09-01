# 导入项目自定义配置模块，用于读取 BASE_URL 等
from utils.config import Config
# 导入 requests 库，用于向 FastAPI 服务发送 HTTP 请求
import requests
# 导入 json 模块，用于 JSON 的序列化与反序列化
import json
# 导入 time 模块，用于记录请求耗时与生成 thread_id
import time
# 导入 sys 模块，用于程序退出（sys.exit）及 stderr 调试输出
import sys
# 导入 argparse，用于解析命令行参数（如 --stream、--debug、--question）
import argparse

# api_test_plus：在 api_test 基础上，支持用自然语言经 Agent API 覆盖浏览器工具

# 自然语言测试指令：与 system_prompt_tmpl.md 中七个浏览器工具一一对应，走 /ask 或 /ask/stream 由模型自行调用工具
NATURAL_LANGUAGE_BROWSER_TOOLS_TEST = """请你作为测试助手，必须通过可用的浏览器工具真实完成下列步骤（不要编造工具结果；若遇人工审核请按流程处理）。测试站点统一使用 https://juejin.cn/ 。

1）打开该网址。
2）查看并说明当前页面 URL。
3）提取当前页可见文本，并在回复里用一两句话概括。
4）用合适的选择器查看页面主标题区域（例如 h1）的文本内容。
5）点击页面上的人工智能分类进入第一篇文章页面。

最后在回复末尾用简短列表汇总每一步实际做了什么、工具返回的关键信息；若某步失败请写明原因。"""


# 从配置中读取 FastAPI 服务的基础 URL（如 http://localhost:8203）
BASE_URL = Config.API_BASE_URL
# 固定用户 ID，测试用
USER_ID = "user_001"
# 每次运行生成基于时间戳的 thread_id，用于隔离会话
THREAD_ID = f"thread_{int(time.time())}"


# 定义向 /ask 接口发送问题并处理响应的函数（非流式）
def ask_question(question: str):
    # 构造请求体：用户 ID、线程 ID、问题内容
    payload = {
        "user_id": USER_ID,
        "thread_id": THREAD_ID,
        "question": question
    }

    # 打印分隔线，便于阅读
    print("\n" + "=" * 70)
    # 打印本次发送的问题
    print(f"发送问题：{question}")
    # 打印短分隔线
    print("-" * 70)

    # 记录请求开始时间，用于后续计算耗时
    start_time = time.time()

    # 使用 try-except 捕获网络或 JSON 解析异常
    try:
        # 向 /ask 发送 POST 请求，超时 90 秒
        resp = requests.post(f"{BASE_URL}/ask", json=payload, timeout=90)
        # 打印 HTTP 状态码
        print(f"状态码: {resp.status_code}")

        # 若状态码非 200，打印错误内容并返回 None
        if resp.status_code != 200:
            print("错误响应：")
            print(resp.text)
            return None

        # 将响应体解析为 JSON 字典
        data = resp.json()
        # 打印“返回结果”标题
        print("返回结果：")
        # 格式化打印完整 JSON（ensure_ascii=False 以正确显示中文）
        print(json.dumps(data, ensure_ascii=False, indent=2))

        # 若状态为需要人工介入（中断）
        if data.get("status") == "interrupted":
            print("\n【检测到需要人工介入】")
            print("中断详情：")
            # 打印中断详情（如 action_requests、review_configs）
            print(json.dumps(data.get("interrupt_details"), ensure_ascii=False, indent=2))
            # 返回中断详情，供主流程进入审核循环
            return data.get("interrupt_details")

        # 若状态为已完成
        elif data.get("status") == "completed":
            print("\nAgent 完整回答：")
            # 打印最终回答文本（去除首尾空白）
            print(data.get("result", "").strip())
            return None  # 无需介入

    # 捕获任意异常并打印
    except Exception as e:
        print(f"请求异常：{e}")
        return None

    # 无论成功或失败，在 finally 中打印本次请求耗时
    finally:
        # 计算经过的秒数
        elapsed = time.time() - start_time
        # 打印耗时（保留两位小数）
        print(f"耗时：{elapsed:.2f} 秒")


# 定义提交人工决策的函数，调用 /intervene 接口（非流式）
def intervene_with_decisions(thread_id: str, user_id: str, decisions: list):
    # 构造请求体：线程 ID、用户 ID、决策列表
    payload = {
        "thread_id": thread_id,
        "user_id": user_id,
        "decisions": decisions
    }

    # 打印分隔线
    print("\n" + "=" * 70)
    # 打印“提交人工决策”提示
    print("提交人工决策...")
    # 格式化打印本次提交的决策内容
    print(json.dumps(decisions, ensure_ascii=False, indent=2))

    try:
        # 向 /intervene 发送 POST 请求，超时 60 秒
        resp = requests.post(f"{BASE_URL}/intervene", json=payload, timeout=60)
        # 打印状态码
        print(f"状态码: {resp.status_code}")

        # 若状态码异常，打印错误并返回 None
        if resp.status_code != 200:
            print("错误：")
            print(resp.text)
            return None

        # 解析响应 JSON
        data = resp.json()
        # 打印“介入后结果”标题
        print("介入后结果：")
        # 格式化打印完整响应
        print(json.dumps(data, ensure_ascii=False, indent=2))

        # 若已完成，打印最终回答并返回 None
        if data.get("status") == "completed":
            print("\n最终回答：")
            print(data.get("result", "").strip())
            return None  # 已完成

        # 若仍有中断，返回新的中断详情供下一轮审核
        elif data.get("status") == "interrupted":
            print("\n【仍有新的中断】")
            return data.get("interrupt_details")

        # 其他未知状态
        else:
            print("未知状态")
            return None

    except Exception as e:
        print(f"介入请求异常：{e}")
        return None


# 支持多轮 HITL 的完整测试主流程（非流式：/ask + /intervene）
def run_multi_hitl_test(question: str, auto_approve=False):
    # 打印当前测试问题
    print(f"\n开始测试问题：{question}\n")

    # 首次发送问题到 /ask，得到中断详情或 None
    interrupt_info = ask_question(question)

    # 只要还有中断，就循环：审核 → 提交决策 → 再请求直至完成或退出
    while interrupt_info:
        if auto_approve:
            # 自动全部通过：不进入交互，直接构造全部 approve 的决策
            print("\n自动全部 approve...")
            decisions = [{"type": "approve"} for _ in interrupt_info.get("action_requests", [])]
        else:
            # 交互式审核：展示列表并等待用户输入，得到决策列表
            decisions = interactive_review(interrupt_info)
            # 若用户选择退出，interactive_review 可能 exit，否则这里若返回 None 则结束循环
            if decisions is None:
                return

        # 将决策提交到 /intervene，得到新的中断详情或 None（完成）
        interrupt_info = intervene_with_decisions(THREAD_ID, USER_ID, decisions)

    # 所有中断处理完毕，打印结束提示
    print("\n" + "=" * 70)
    print("测试流程结束（已完成或已退出）")


# 流式请求 /ask/stream：解析 SSE，按 token 打印；若 interrupted 返回 interrupt_details，若 completed 或异常返回 None
def ask_question_stream(question: str, debug: bool = False):
    # 构造请求体（与 /ask 一致）
    payload = {"user_id": USER_ID, "thread_id": THREAD_ID, "question": question}
    # 打印分隔线
    print("\n" + "=" * 70)
    # 打印流式请求的提示与问题
    print(f"流式请求 /ask/stream：{question}")
    # 若开启调试，在 stderr 提示将打印每个 token 的长度
    if debug:
        print("[调试] 已开启，每收到一个 token 将在 stderr 打印 [token 长度=N]", file=sys.stderr)
    # 打印短分隔线
    print("-" * 70)
    # 记录开始时间
    start_time = time.time()
    # 统计收到的 token 事件个数（用于调试）
    token_count = 0
    try:
        # 向 /ask/stream 发送 POST，stream=True 表示流式接收响应体
        resp = requests.post(f"{BASE_URL}/ask/stream", json=payload, timeout=120, stream=True)
        # 打印 HTTP 状态码
        print(f"状态码: {resp.status_code}")
        # 若状态码非 200，打印错误并返回 None
        if resp.status_code != 200:
            print("错误响应：", resp.text)
            return None
        # 按行迭代响应体（decode_unicode=True 得到字符串）
        for line in resp.iter_lines(decode_unicode=True):
            # 跳过空行或仅空白行
            if not line or not line.strip():
                continue
            # 只处理 SSE 数据行（以 "data: " 开头）
            if line.startswith("data: "):
                try:
                    # 去掉 "data: " 前缀并解析 JSON 得到事件对象
                    event = json.loads(line[6:].strip())
                except json.JSONDecodeError:
                    continue
                # 取出事件类型：token / tool_output / completed / interrupted
                t = event.get("type")
                # 若为 token 事件，将内容追加打印（实现逐字/逐段输出）
                if t == "token":
                    piece = event.get("content", "")
                    if piece:
                        if debug:
                            token_count += 1
                            sys.stderr.write(f"[token #{token_count} 长度={len(piece)}] ")
                            sys.stderr.flush()
                        print(piece, end="", flush=True)
                        sys.stdout.flush()
                # 若为工具节点输出
                elif t == "tool_output":
                    piece = event.get("content", "")
                    if piece:
                        if debug:
                            sys.stderr.write(f"[tool_output 长度={len(piece)}] ")
                            sys.stderr.flush()
                        print("\n[工具输出]\n", end="", flush=True)
                        print(piece, end="", flush=True)
                        print("\n[/工具输出]\n", end="", flush=True)
                        sys.stdout.flush()
                # 若为 completed，打印最终结果并返回 None 结束
                elif t == "completed":
                    if debug:
                        sys.stderr.write(f"\n[调试] 共收到 {token_count} 个 token 事件\n")
                        sys.stderr.flush()
                    result = event.get("result", "")
                    print("\n\n[SSE completed] result 长度:", len(result))
                    if result:
                        print("\n--- 最终完整结果 ---\n")
                        print(result)
                        print("\n--- 以上为最终完整结果 ---")
                    print(f"耗时：{time.time() - start_time:.2f} 秒")
                    return None
                # 若为 interrupted，打印中断信息并返回 interrupt_details 以进入多轮 HITL
                elif t == "interrupted":
                    print("\n\n[SSE interrupted] 需要人工审核，将进入多轮 HITL 交互。")
                    print("interrupt_details:")
                    print(json.dumps(event.get("interrupt_details", {}), ensure_ascii=False, indent=2))
                    print(f"耗时：{time.time() - start_time:.2f} 秒")
                    return event.get("interrupt_details", {})
        # 流结束但未收到 completed/interrupted 时打印耗时并返回 None
        print(f"耗时：{time.time() - start_time:.2f} 秒")
        return None
    except Exception as e:
        print(f"流式请求异常：{e}")
        return None


# 流式提交决策：请求 /intervene/stream，按 token 流式输出恢复后的回答；若再次中断则返回 interrupt_details
def intervene_stream(thread_id: str, user_id: str, decisions: list, debug: bool = False):
    # 构造请求体（与 /intervene 一致）
    payload = {"thread_id": thread_id, "user_id": user_id, "decisions": decisions}
    # 打印分隔线
    print("\n" + "=" * 70)
    # 打印流式恢复的提示
    print("提交人工决策（流式恢复 /intervene/stream）...")
    # 格式化打印决策内容
    print(json.dumps(decisions, ensure_ascii=False, indent=2))
    # 统计 token 事件数（调试用）
    token_count = 0
    try:
        # 向 /intervene/stream 发送 POST，stream=True 流式接收
        resp = requests.post(f"{BASE_URL}/intervene/stream", json=payload, timeout=120, stream=True)
        # 打印状态码
        print(f"状态码: {resp.status_code}")
        # 若状态码非 200，打印错误并返回 None
        if resp.status_code != 200:
            print("错误：", resp.text)
            return None
        # 按行迭代响应体
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.strip():
                continue
            if line.startswith("data: "):
                try:
                    event = json.loads(line[6:].strip())
                except json.JSONDecodeError:
                    continue
                t = event.get("type")
                # token：逐段打印并刷新 stdout
                if t == "token":
                    piece = event.get("content", "")
                    if piece:
                        if debug:
                            token_count += 1
                            sys.stderr.write(f"[token #{token_count} 长度={len(piece)}] ")
                            sys.stderr.flush()
                        print(piece, end="", flush=True)
                        sys.stdout.flush()
                # 工具节点输出
                elif t == "tool_output":
                    piece = event.get("content", "")
                    if piece:
                        if debug:
                            sys.stderr.write(f"[tool_output 长度={len(piece)}] ")
                            sys.stderr.flush()
                        print("\n[工具输出]\n", end="", flush=True)
                        print(piece, end="", flush=True)
                        print("\n[/工具输出]\n", end="", flush=True)
                        sys.stdout.flush()
                # completed：打印最终结果并返回 None
                elif t == "completed":
                    if debug:
                        sys.stderr.write(f"\n[调试] 共收到 {token_count} 个 token 事件\n")
                        sys.stderr.flush()
                    result = event.get("result", "")
                    print("\n\n[intervene/stream completed] result 长度:", len(result))
                    if result:
                        print("\n--- 最终完整结果 ---\n")
                        print(result)
                        print("\n--- 以上为最终完整结果 ---")
                    return None
                # interrupted：打印新中断信息并返回 interrupt_details 以继续多轮审核
                elif t == "interrupted":
                    print("\n\n[intervene/stream interrupted] 仍有新的中断，需继续审核。")
                    print("interrupt_details:")
                    print(json.dumps(event.get("interrupt_details", {}), ensure_ascii=False, indent=2))
                    return event.get("interrupt_details", {})
        return None
    except Exception as e:
        print(f"流式介入请求异常：{e}")
        return None


# 交互式人工审核：根据 interrupt_details 展示待审核工具列表，等待用户输入并返回决策列表
def interactive_review(interrupt_details):
    # 若无中断详情，直接返回 None
    if not interrupt_details:
        return None

    # 从中断详情中取出待审核的工具调用列表
    action_requests = interrupt_details.get("action_requests", [])
    # 若列表为空，提示并返回 None
    if not action_requests:
        print("没有待审核的工具调用")
        return None

    # 打印“待审核工具调用列表”标题
    print("\n待审核工具调用列表：")
    # 遍历每个待审核项，打印序号、工具名、参数
    for i, action in enumerate(action_requests):
        name = action.get("name", "未知工具")
        args = action.get("args", action.get("arguments", {}))
        print(f"  [{i}] 工具: {name}")
        print(f"      参数: {json.dumps(args, ensure_ascii=False, indent=2)}")
        print("-" * 50)

    # 打印操作选项说明
    print("\n操作选项：")
    print("  approve    → 全部 approve")
    print("  reject    → 全部 reject")
    print("  edit N  → 编辑第 N 个工具的参数（N 从 0 开始）")
    print("  quit    → 退出测试")

    # 循环等待用户输入直到得到有效决策
    while True:
        # 读取用户输入并去除首尾空白、转小写
        choice = input("\n请输入你的选择 (approve/reject/edit N/quit): ").strip().lower()

        # 用户选择退出则直接退出程序
        if choice == "quit":
            print("用户选择退出测试")
            sys.exit(0)

        # 全部通过：返回全部 approve 的决策列表
        if choice == "approve":
            return [{"type": "approve"} for _ in action_requests]

        # 全部拒绝：返回全部 reject 的决策列表
        if choice == "reject":
            return [{"type": "reject"} for _ in action_requests]

        # 编辑第 N 个工具的参数
        if choice.startswith("edit "):
            try:
                # 从输入中解析出索引 N（如 "edit 1" 中的 1）
                idx = int(choice.split()[1])
                # 若索引在有效范围内
                if 0 <= idx < len(action_requests):
                    print(f"\n当前参数（第 {idx} 个）：")
                    # 取出当前参数并打印
                    current_args = action_requests[idx].get("args", action_requests[idx].get("arguments", {}))
                    print(json.dumps(current_args, ensure_ascii=False, indent=2))

                    # 提示用户输入新 JSON 参数，回车则保持原样
                    new_args_str = input("请输入新的 JSON 参数（直接回车保持原样）：").strip()
                    if new_args_str:
                        new_args = json.loads(new_args_str)
                    else:
                        new_args = current_args

                    # 构造决策列表：被编辑项为 edit，其余为 approve
                    decisions = []
                    for i in range(len(action_requests)):
                        if i == idx:
                            decisions.append({
                                "type": "edit",
                                "edited_action": {
                                    "name": action_requests[i]["name"],
                                    "args": new_args
                                }
                            })
                        else:
                            decisions.append({"type": "approve"})

                    return decisions
                else:
                    print("索引超出范围")
            except (ValueError, json.JSONDecodeError) as e:
                print(f"输入错误：{e}")
            continue

        # 未识别的输入，提示重新选择
        print("无效输入，请重新选择")



# 主程序入口
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="api_test_plus：测试 Agent API（/ask、/intervene、/ask/stream）；--playwright 使用自然语言触发浏览器七类能力联测"
    )
    # 是否使用流式模式（/ask/stream 与 /intervene/stream）
    parser.add_argument("--stream", action="store_true", help="请求 /ask/stream 并打印 SSE 事件（按 token 流式输出）")
    # 是否在 stderr 打印每个 token 的序号与长度（便于调试）
    parser.add_argument("--debug", action="store_true", help="与 --stream 同用时，在 stderr 打印每个 token 的序号与长度，便于确认是否逐 token 收到")
    # 测试问题内容（默认值可改）
    # parser.add_argument("--question", type=str, default="今天天气怎么样？", help="测试问题")
    parser.add_argument("--question", type=str, default="使用 translate 技能，把「今天天气很好」翻译成英文。并返回翻译后的结果。", help="测试问题")
    parser.add_argument(
        "--playwright",
        action="store_true",
        help="使用内置自然语言长指令，经 Agent 依次覆盖打开网页、当前 URL、抓文本、链、元素、点击、后退等（需已启动 agent_api；建议配合 --stream）",
    )
    # 解析命令行参数
    args = parser.parse_args()

    if args.playwright:
        args.question = NATURAL_LANGUAGE_BROWSER_TOOLS_TEST

    # 打印测试开始信息
    print("开始测试 ReAct Agent API（api_test_plus.py：自然语言联调浏览器能力）...\n")
    # 打印后端地址、用户 ID、会话 ID
    print(f"后端: {BASE_URL}  用户: {USER_ID}  会话: {THREAD_ID}\n")

    if args.stream:
        # 流式模式：先请求 /ask/stream，若中断则进入多轮 HITL，恢复时用 /intervene/stream 保持流式输出
        interrupt_info = ask_question_stream(args.question, debug=args.debug)
        # print(f"interrupt_info: {interrupt_info}")
        # 若有中断，循环：审核 → 流式恢复，直到完成或用户退出
        while interrupt_info:
            decisions = interactive_review(interrupt_info)
            if decisions is None:
                break
            interrupt_info = intervene_stream(THREAD_ID, USER_ID, decisions, debug=args.debug)
        # 打印流程结束提示
        print("\n" + "=" * 70)
        print("测试流程结束（已完成或已退出）")
    else:
        # 非流式模式：走多轮 HITL 主流程（/ask + /intervene）
        run_multi_hitl_test(args.question)
