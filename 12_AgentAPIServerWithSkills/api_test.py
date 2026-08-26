# 导入项目自定义配置模块
from utils.config import Config
# 导入 requests 库，用于发送 HTTP 请求到 FastAPI 服务
import requests
# 导入 json 模块，用于处理 JSON 数据（序列化/反序列化）
import json
# 导入 time 模块，用于记录请求耗时
import time
# 导入 sys 模块，用于程序退出（sys.exit）
import sys

# 从配置中读取 FastAPI 服务的基础 URL
BASE_URL = Config.API_BASE_URL
# 固定用户 ID（测试用）
USER_ID = "user_001"
# 每次运行使用基于时间戳的 thread_id
THREAD_ID = f"thread_{int(time.time())}"

# 定义发送问题给 /ask 接口的函数
def ask_question(question: str):
    # 构造请求体（payload），包含用户 ID、线程 ID 和问题内容
    payload = {
        "user_id": USER_ID,
        "thread_id": THREAD_ID,
        "question": question
    }

    # 打印分隔线，美化输出
    print("\n" + "=" * 70)
    # 打印本次发送的问题内容
    print(f"发送问题：{question}")
    print("-" * 70)

    # 记录请求开始时间，用于计算耗时
    start_time = time.time()

    # 使用 try-except 捕获可能的网络或解析异常
    try:
        # 向 /ask 接口发送 POST 请求，超时设为 90 秒
        resp = requests.post(f"{BASE_URL}/ask", json=payload, timeout=90)
        # 打印 HTTP 状态码
        print(f"状态码: {resp.status_code}")

        # 如果状态码不是 200，打印错误响应内容并返回 None
        if resp.status_code != 200:
            print("错误响应：")
            print(resp.text)
            return None

        # 将响应体解析为 JSON
        data = resp.json()
        # 打印完整的返回结果（格式化输出，便于阅读）
        print("返回结果：")
        print(json.dumps(data, ensure_ascii=False, indent=2))

        # 判断是否需要人工介入（中断状态）
        if data.get("status") == "interrupted":
            print("\n【检测到需要人工介入】")
            print("中断详情：")
            # 打印中断细节（action_requests 等）
            print(json.dumps(data.get("interrupt_details"), ensure_ascii=False, indent=2))
            # 返回中断详情供后续处理
            return data.get("interrupt_details")

        # 如果是已完成状态，打印最终回答并返回 None（表示无需介入）
        elif data.get("status") == "completed":
            print("\nAgent 完整回答：")
            print(data.get("result", "").strip())
            return None  # 无需介入

    # 捕获所有异常并打印
    except Exception as e:
        print(f"请求异常：{e}")
        return None

    # 无论成功与否，都打印本次请求耗时
    finally:
        elapsed = time.time() - start_time
        print(f"耗时：{elapsed:.2f} 秒")

# 定义提交人工决策的函数（调用 /intervene 接口）
def intervene_with_decisions(thread_id: str, user_id: str, decisions: list):
    # 构造请求体，包含线程 ID、用户 ID 和人工决策列表
    payload = {
        "thread_id": thread_id,
        "user_id": user_id,
        "decisions": decisions
    }

    # 打印分隔线
    print("\n" + "=" * 70)
    print("提交人工决策...")
    # 打印本次提交的决策内容（格式化）
    print(json.dumps(decisions, ensure_ascii=False, indent=2))

    try:
        # 向 /intervene 接口发送 POST 请求，超时 60 秒
        resp = requests.post(f"{BASE_URL}/intervene", json=payload, timeout=60)
        print(f"状态码: {resp.status_code}")

        # 如果状态码异常，打印错误并返回 None
        if resp.status_code != 200:
            print("错误：")
            print(resp.text)
            return None

        # 解析响应 JSON
        data = resp.json()
        # 打印介入后的返回结果
        print("介入后结果：")
        print(json.dumps(data, ensure_ascii=False, indent=2))

        # 如果完成，打印最终回答并返回 None
        if data.get("status") == "completed":
            print("\n最终回答：")
            print(data.get("result", "").strip())
            return None  # 已完成

        # 如果仍有中断，返回新的中断详情
        elif data.get("status") == "interrupted":
            print("\n【仍有新的中断】")
            return data.get("interrupt_details")

        # 其他未知状态
        else:
            print("未知状态")
            return None

    # 捕获异常
    except Exception as e:
        print(f"介入请求异常：{e}")
        return None

# 交互式人工审核函数：显示待审核工具列表并让用户选择决策
def interactive_review(interrupt_details):
    # 如果没有中断详情，直接返回 None
    if not interrupt_details:
        return None

    # 获取待审核的工具调用请求列表
    action_requests = interrupt_details.get("action_requests", [])
    if not action_requests:
        print("没有待审核的工具调用")
        return None

    # 打印待审核工具列表
    print("\n待审核工具调用列表：")
    for i, action in enumerate(action_requests):
        name = action.get("name", "未知工具")
        args = action.get("args", action.get("arguments", {}))
        print(f"  [{i}] 工具: {name}")
        print(f"      参数: {json.dumps(args, ensure_ascii=False, indent=2)}")
        print("-" * 50)

    # 打印操作提示
    print("\n操作选项：")
    print("  approve    → 全部 approve")
    print("  reject    → 全部 reject")
    print("  edit N  → 编辑第 N 个工具的参数（N 从 0 开始）")
    print("  quit    → 退出测试")

    # 循环等待用户输入
    while True:
        # 获取用户输入并转换为小写
        choice = input("\n请输入你的选择 (approve/reject/edit N/quit): ").strip().lower()

        # 用户选择退出程序
        if choice == "quit":
            print("用户选择退出测试")
            sys.exit(0)

        # 全部通过
        if choice == "approve":
            return [{"type": "approve"} for _ in action_requests]

        # 全部拒绝
        if choice == "reject":
            return [{"type": "reject"} for _ in action_requests]

        # 编辑指定工具的参数
        if choice.startswith("edit "):
            try:
                # 解析索引
                idx = int(choice.split()[1])
                if 0 <= idx < len(action_requests):
                    print(f"\n当前参数（第 {idx} 个）：")
                    current_args = action_requests[idx].get("args", action_requests[idx].get("arguments", {}))
                    print(json.dumps(current_args, ensure_ascii=False, indent=2))

                    # 让用户输入新的 JSON 参数
                    new_args_str = input("请输入新的 JSON 参数（直接回车保持原样）：").strip()
                    if new_args_str:
                        new_args = json.loads(new_args_str)
                    else:
                        new_args = current_args

                    # 构造决策列表：被编辑的用 edit，其他默认 approve
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

        # 无效输入，提示重新选择
        print("无效输入，请重新选择")

# 支持多轮 HITL 的完整测试主流程函数
def run_multi_hitl_test(question: str, auto_approve=False):
    # 打印测试开始信息
    print(f"\n开始测试问题：{question}\n")

    # 第一次发送问题
    interrupt_info = ask_question(question)

    # 只要还有中断，就进入循环处理
    while interrupt_info:
        if auto_approve:
            # 自动全部通过模式
            print("\n自动全部 approve...")
            decisions = [{"type": "approve"} for _ in interrupt_info.get("action_requests", [])]
        else:
            # 交互式审核
            decisions = interactive_review(interrupt_info)
            # 如果用户选择退出，则直接结束
            if decisions is None:
                return

        # 提交决策，继续执行
        interrupt_info = intervene_with_decisions(THREAD_ID, USER_ID, decisions)

    # 测试流程结束
    print("\n" + "=" * 70)
    print("测试流程结束（已完成或已退出）")


# 主程序入口
if __name__ == "__main__":
    print("开始测试 ReAct Agent API（含 Skill）...\n")
    print(f"后端: {BASE_URL}  用户: {USER_ID}  会话: {THREAD_ID}\n")

    # 测试用例
    # run_multi_hitl_test("外面的天气怎么样？")
    # run_multi_hitl_test("使用 translate 技能，把「今天天气很好」翻译成英文")
    # run_multi_hitl_test("使用 summarize 技能，总结一下：人工智能正在改变我们的生活和工作方式。", auto_approve=True)
    run_multi_hitl_test(
        "使用 text_inspect 技能，切记不要使用search_documents工具。帮我分析一下下面这段文字：统计字数、行数、段落数，再提取几个关键词，"
        "看看里面有没有网址或邮箱，用清晰的结构化结果告诉我。\n\n"
        "Hello world.\n\n"
        "这是一个测试文本：https://example.com\n"
        "联系邮箱 test@example.com\n"
    )