import gradio as gr
import requests
import json
import re
from typing import List, Dict, Any
from utils.config import Config
from utils.logger import LoggerManager

# =============================================
# 配置区
# =============================================
BASE_URL = Config.BASE_URL
THREAD_ID = Config.THREAD_ID
USER_ID = Config.USER_ID

current_interrupt_info: Dict[str, Any] = None

# 获取项目统一的日志记录器实例
logger = LoggerManager.get_logger()

def send_message(message: str, history: List[Dict[str, str]]):
    global current_interrupt_info

    if not message.strip():
        history.append({"role": "assistant", "content": "请输入内容"})
        return history

    history.append({"role": "user", "content": message})
    lower_msg = message.lower().strip()

    approve_keywords = ["同意", "通过", "yes", "ok", "允许", "确认", "y", "approve"]
    reject_keywords = ["拒绝", "no", "不允许", "禁止", "否", "n", "reject"]
    exit_keywords = ["退出", "stop", "结束", "cancel", "q"]
    help_keywords = ["帮助", "help"]
    edit_keywords = ["编辑", "edit", "修改", "改", "change"]

    is_approve = any(kw in lower_msg for kw in approve_keywords)
    is_reject = any(kw in lower_msg for kw in reject_keywords)
    is_exit = any(kw in lower_msg for kw in exit_keywords)
    is_help = any(kw in lower_msg for kw in help_keywords)
    is_edit = any(kw in lower_msg for kw in edit_keywords)

    if current_interrupt_info:
        actions = current_interrupt_info.get("action_requests", [])
        if not actions:
            history.append({"role": "assistant", "content": "当前没有待审核的工具调用"})
            current_interrupt_info = None
            return history

        if is_exit:
            history.append({"role": "assistant", "content": "已退出本次对话。需要重新开始请清空会话或输入新问题。"})
            current_interrupt_info = None
            return history

        if is_help:
            help_text = """
                        **审核帮助：**
                        - `同意` / `通过` / `yes` / `ok` → 全部允许执行
                        - `拒绝` / `no` → 全部禁止执行
                        - `编辑 N {新参数JSON}` → 编辑第 N 个工具的参数（N 从 0 开始）
                          示例：编辑 1 {"city": "上海"}
                        - `退出` / `stop` → 结束对话
                        - `帮助` → 显示此帮助
                        """
            history.append({"role": "assistant", "content": help_text})
            return history

        decisions = []
        edit_found = False
        edit_pattern = r"(?:编辑|edit|修改|改|change)\s*(?:第)?\s*(\d+)\s*(?:个|工具)?\s*(?:参数)?\s*({.*})"
        matches = re.findall(edit_pattern, message, re.IGNORECASE | re.DOTALL)
        logger.info(f"编辑匹配结果: {matches}")
        if matches or is_edit:
            edit_found = True
            edited_indices = set()
            for idx_str, new_args_str in matches:
                try:
                    idx = int(idx_str.strip())
                    if 0 <= idx -1 < len(actions):
                        new_args = json.loads(new_args_str.strip())
                        edited_action = {"name": actions[idx -1]["name"], "args": new_args}
                        decisions.append({"type": "edit", "edited_action": edited_action})
                        edited_indices.add(idx -1)
                        history.append({"role": "assistant",
                                        "content": f"已记录编辑：第 {idx} 个工具参数改为 {json.dumps(new_args, ensure_ascii=False)}"})
                    else:
                        history.append(
                            {"role": "assistant", "content": f"索引 {idx} 超出范围（共 {len(actions)} 个工具）"})
                except (ValueError, json.JSONDecodeError) as e:
                    history.append({"role": "assistant", "content": f"编辑参数解析失败：{str(e)}\n请使用合法 JSON 格式"})

            for i in range(len(actions)):
                if i not in edited_indices:
                    decisions.append({"type": "approve"})

        if not edit_found:
            if is_approve:
                decisions = [{"type": "approve"} for _ in actions]
                action_msg = "**您选择了：全部同意** 正在继续执行..."
            elif is_reject:
                decisions = [{"type": "reject"} for _ in actions]
                action_msg = "**您选择了：全部拒绝** 正在处理..."
            else:
                history.append(
                    {"role": "assistant", "content": "未识别到有效指令。请回复「同意」「拒绝」「编辑 N {参数}」「帮助」等"})
                return history
            history.append({"role": "assistant", "content": action_msg})

        payload = {"thread_id": THREAD_ID, "user_id": USER_ID, "decisions": decisions}

        try:
            resp = requests.post(f"{BASE_URL}/intervene", json=payload, timeout=60)
            if resp.status_code != 200:
                history.append({"role": "assistant", "content": f"审核提交失败：{resp.status_code} {resp.text}"})
                current_interrupt_info = None
                return history

            data = resp.json()
            if data.get("status") == "completed":
                result = data.get("result", "").strip()
                history.append({"role": "assistant", "content": f"**执行完成！**\n\n最终回答：\n{result}"})
                current_interrupt_info = None
                return history

            elif data.get("status") == "interrupted":
                current_interrupt_info = data.get("interrupt_details", {})
                actions = current_interrupt_info.get("action_requests", [])
                interrupt_text = "**⚠️ 仍有新的待审核项**\n\nAgent 继续执行后，又遇到需要您确认的工具调用。\n\n**本次待审核：**\n\n"
                for i, action in enumerate(actions):
                    name = action.get("name", "未知工具")
                    args_str = json.dumps(action.get("args", action.get("arguments", {})), ensure_ascii=False, indent=2)
                    interrupt_text += f"**{i + 1}.** 工具：`{name}`  \n参数：\n```json\n{args_str}\n```\n\n"
                interrupt_text += "\n**请直接回复以下任一指令继续：**\n\n✅ `同意` / `yes` / `ok`\n\n❌ `拒绝` / `no`\n\n✏️ `编辑 N {新参数JSON}`\n\n🛑 `退出` / `帮助`"
                history.append({"role": "assistant", "content": interrupt_text})
                return history
            else:
                history.append({"role": "assistant", "content": "未知状态，审核已结束"})
                current_interrupt_info = None
                return history
        except Exception as e:
            history.append({"role": "assistant", "content": f"审核执行异常：{str(e)}"})
            current_interrupt_info = None
            return history

    else:
        payload = {"user_id": USER_ID, "thread_id": THREAD_ID, "question": message}
        history.append({"role": "assistant", "content": "思考中..."})

        try:
            resp = requests.post(f"{BASE_URL}/ask", json=payload, timeout=90)
            if resp.status_code != 200:
                history[-1]["content"] = f"API 错误 {resp.status_code}: {resp.text}"
                return history

            data = resp.json()
            if data.get("status") == "completed":
                answer = data.get("result", "无返回内容").strip()
                history[-1]["content"] = answer
                return history

            elif data.get("status") == "interrupted":
                current_interrupt_info = data.get("interrupt_details", {})
                actions = current_interrupt_info.get("action_requests", [])
                interrupt_text = "**⚠️ 需要人工确认（安全审核）**\n\n为了保护您的隐私与安全，Agent 即将调用外部工具，此操作需要您的许可。\n\n**待审核的工具调用：**\n\n"
                for i, action in enumerate(actions):
                    name = action.get("name", "未知工具")
                    args_str = json.dumps(action.get("args", action.get("arguments", {})), ensure_ascii=False, indent=2)
                    interrupt_text += f"**{i + 1}.** 工具：`{name}`  \n参数：\n```json\n{args_str}\n```\n\n"
                interrupt_text += "\n**请直接回复以下任一指令继续：**\n\n✅ `同意` / `通过` / `yes`\n\n❌ `拒绝` / `no`\n\n✏️ `编辑 N {新参数JSON}`\n\n🛑 `退出` / `帮助`"
                history[-1]["content"] = interrupt_text
                return history
            else:
                history[-1]["content"] = "未知响应状态"
                return history
        except Exception as e:
            history[-1]["content"] = f"请求异常：{str(e)}"
            return history


def clear_chat():
    global current_interrupt_info
    current_interrupt_info = None
    return []


# ────────────────────────────────────────────────
#  配色方案说明（60-30-10 法则 + WCAG 标准）
# ────────────────────────────────────────────────
# 60% 主色调（背景）：#f1f5f9（slate-100）极浅冷灰，冷静不刺眼
# 30% 辅助色（卡片）：#ffffff 纯白，靠灰度边框 #e2e8f0 区分层级
# 10% 强调色（交互）：#2563eb（blue-600）仅用于按钮、链接、徽章
#
# 对比度合规：
# - 正文 #0f172a on #ffffff = 15.3:1  （≥4.5 ✓）
# - 正文 #1e293b on #ffffff = 9.8:1   （≥4.5 ✓）
# - 次级 #475569 on #ffffff = 5.3:1   （≥4.5 ✓）
# - 用户消息 #f8fafc on #334155 = 11.4:1 （≥4.5 ✓）
# - 大标题 #0f172a on #f1f5f9 = 12.6:1 （≥3 ✓）
#
# 情绪：技术后台，蓝灰系，零高饱和大面积铺色
# ────────────────────────────────────────────────

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Inter'), 'system-ui', '-apple-system', 'sans-serif'],
).set(
    body_background_fill="#f1f5f9",
    body_text_color="#0f172a",
    body_text_size="15px",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_hover="#f8fafc",
    button_secondary_text_color="#334155",
    input_background_fill="#ffffff",
    input_border_color="#cbd5e1",
)

custom_css = """
/* ========== 1. 全局基础（60% 主色调） ========== */
.gradio-container {
    max-width: 1100px !important;
    width: 95% !important;
    margin: 0 auto;
    padding: 2rem 1.5rem;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: #f1f5f9 !important; /* slate-100：60% 主色调，极浅冷灰 */
}

/* ========== 2. 标题区域（30% 辅助色卡片） ========== */
.header-section {
    text-align: center;
    margin-bottom: 1.5rem;
    padding: 1.5rem 2rem;
    background: #ffffff; /* 30% 辅助色：纯白卡片 */
    border-radius: 16px;
    border: 1px solid #e2e8f0; /* slate-200：灰度边框区分层级，不用彩色 */
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 4px 12px rgba(15, 23, 42, 0.03);
}

.header-title {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: #0f172a !important; /* slate-900：对比度 15.3:1，远超 WCAG 大标题 3:1 标准 */
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.02em;
}

.header-subtitle {
    font-size: 0.875rem !important;
    color: #475569 !important; /* slate-600：对比度 5.3:1，满足正文 4.5:1 */
    font-weight: 500;
}

.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.75rem;
    padding: 0.35rem 1rem;
    background: #eff6ff; /* blue-50：极低饱和蓝，属于 10% 强调色的弱化背景 */
    border-radius: 100px;
    font-size: 0.8rem;
    color: #1d4ed8; /* blue-700：强调色文字，仅小面积使用 */
    font-weight: 600;
    border: 1px solid #dbeafe; /* blue-100 */
}

/* ========== 3. 聊天框主体（30% 辅助色） ========== */
#chatbot {
    border-radius: 16px !important;
    overflow: hidden;
    border: 1px solid #e2e8f0; /* slate-200 灰度边框 */
    box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.04), 0 10px 20px -3px rgba(15, 23, 42, 0.03);
    background: #ffffff !important; /* 纯白卡片 */
    height: 600px !important;
}

/* ========== 4. 消息层级 ========== */
.message {
    padding: 0.85rem 1.15rem !important;
    margin: 0.5rem 0 !important;
    border-radius: 16px !important;
    max-width: 85% !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    animation: messageAppear 0.3s ease-out forwards;
    opacity: 0;
    transform: translateY(6px);
}

@keyframes messageAppear {
    to { opacity: 1; transform: translateY(0); }
}

/* 用户消息：冷静深灰蓝（非高饱和），符合技术后台情绪 */
.message.user {
    background: #334155 !important; /* slate-700：低饱和、沉稳、克制 */
    color: #f8fafc !important; /* slate-50：接近白但不刺眼，对比度 11.4:1 */
    margin-left: auto !important;
    margin-right: 0.5rem !important;
    border-radius: 16px 16px 4px 16px !important;
    border: none !important;
}

.message.user .message-content {
    color: #f8fafc !important;
}

/* 助手消息：纯白卡片，靠灰度边框与背景区分层级 */
.message.bot {
    background: #ffffff !important;
    color: #1e293b !important; /* slate-800：正文对比度 9.8:1 */
    margin-right: auto !important;
    margin-left: 0.5rem !important;
    border-radius: 16px 16px 16px 4px !important;
    border: 1px solid #e2e8f0 !important; /* slate-200 */
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}

.message.bot .message-content {
    color: #1e293b !important;
}

/* ========== 5. 消息内富文本 ========== */
.message.bot pre {
    background: #1e293b !important; /* slate-800 代码块背景 */
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
    margin: 0.5rem 0 !important;
    overflow-x: auto;
    border: 1px solid #334155; /* slate-700 边框增加层级 */
}

.message.bot code {
    font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace !important;
    font-size: 0.85rem !important;
    line-height: 1.5 !important;
    color: #e2e8f0 !important; /* slate-200：代码文字高对比 */
}

.message.bot p code {
    background: #f1f5f9 !important; /* 行内代码用浅灰底 */
    color: #1d4ed8 !important; /* blue-700：10% 强调色 */
    padding: 0.15rem 0.4rem !important;
    border-radius: 4px !important;
    font-size: 0.85em !important;
    border: 1px solid #e2e8f0;
}

/* 链接：强调色 + hover 状态区分 */
.message.bot a {
    color: #2563eb !important; /* blue-600：10% 强调色 */
    text-decoration: underline;
    text-underline-offset: 2px;
    transition: color 0.15s ease;
}
.message.bot a:hover {
    color: #1e40af !important; /* blue-800：hover 加深 */
}

/* ========== 6. 输入框区域（30% 辅助色卡片） ========== */
.input-area {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.25rem;
    margin-top: 1rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}

.textbox {
    font-size: 0.95rem !important;
    border-radius: 12px !important;
    border: 1.5px solid #cbd5e1 !important; /* slate-300：输入框边框 */
    background: #ffffff !important;
    padding: 0.85rem 1.1rem !important;
    transition: all 0.2s ease !important;
    line-height: 1.5 !important;
    color: #0f172a !important; /* slate-900 输入文字 */
}

.textbox:focus {
    border-color: #2563eb !important; /* blue-600：10% 强调色聚焦环 */
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
    outline: none !important;
}

.textbox::placeholder {
    color: #94a3b8 !important; /* slate-400：占位符，对比度较低但不影响输入 */
}

/* ========== 7. 按钮四态（10% 强调色） ========== */
.button {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    padding: 0.7rem 1.8rem !important;
    border-radius: 10px !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.01em;
    border: none !important;
}

/* 主按钮：默认 / hover / active / disabled */
.button.primary {
    background: #2563eb !important; /* blue-600：10% 强调色 */
    color: #ffffff !important;
    box-shadow: 0 1px 2px rgba(37, 99, 235, 0.15);
}
.button.primary:hover {
    background: #1d4ed8 !important; /* blue-700：hover 加深 */
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    transform: translateY(-1px);
}
.button.primary:active {
    background: #1e40af !important; /* blue-800：active 再加深 */
    transform: translateY(0);
    box-shadow: 0 1px 2px rgba(37, 99, 235, 0.15);
}
.button.primary:disabled,
.button.primary[disabled] {
    background: #e2e8f0 !important; /* slate-200：禁用灰 */
    color: #94a3b8 !important; /* slate-400：禁用文字 */
    box-shadow: none;
    cursor: not-allowed;
    transform: none;
}

/* 次按钮：灰度层级，四态完整 */
.button.secondary {
    background: #ffffff !important;
    color: #334155 !important; /* slate-700 */
    border: 1px solid #cbd5e1 !important; /* slate-300 */
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.button.secondary:hover {
    background: #f8fafc !important; /* slate-50：hover 微亮 */
    border-color: #94a3b8 !important; /* slate-400 */
    color: #0f172a !important; /* slate-900 */
}
.button.secondary:active {
    background: #f1f5f9 !important; /* slate-100：active 再深 */
    border-color: #64748b !important; /* slate-500 */
}
.button.secondary:disabled,
.button.secondary[disabled] {
    background: #f1f5f9 !important;
    color: #cbd5e1 !important; /* slate-300 */
    border-color: #e2e8f0 !important;
    cursor: not-allowed;
}

/* ========== 8. 提示文字 ========== */
.hint-text {
    font-size: 0.8rem !important;
    color: #64748b !important; /* slate-500：提示文字，对比度 4.6:1 临界，仅作辅助信息 */
    text-align: left !important;
    margin: 0.75rem 0.25rem 0.5rem 0.25rem !important;
    line-height: 1.5;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ========== 9. 滚动条（灰度层级） ========== */
#chatbot::-webkit-scrollbar { width: 6px; }
#chatbot::-webkit-scrollbar-track { background: transparent; }
#chatbot::-webkit-scrollbar-thumb {
    background: #cbd5e1; /* slate-300 */
    border-radius: 100px;
}
#chatbot::-webkit-scrollbar-thumb:hover {
    background: #94a3b8; /* slate-400：hover 加深 */
}

/* ========== 10. 深色模式（色彩逻辑统一，禁用纯白刺眼文字） ========== */
.dark .gradio-container {
    background: #0f172a !important; /* slate-900：深色背景 */
}

.dark .header-section {
    background: #1e293b !important; /* slate-800：卡片层 */
    border-color: #334155 !important; /* slate-700 边框 */
}

.dark .header-title {
    color: #f1f5f9 !important; /* slate-100：主文字，非纯白避免刺眼 */
}

.dark .header-subtitle {
    color: #94a3b8 !important; /* slate-400：次级文字 */
}

.dark .header-badge {
    background: #172554 !important; /* blue-950：极深蓝背景 */
    color: #93c5fd !important; /* blue-300：柔和强调 */
    border-color: #1e3a8a !important;
}

.dark #chatbot {
    background: #1e293b !important; /* slate-800 卡片 */
    border-color: #334155 !important;
}

.dark .message.user {
    background: #475569 !important; /* slate-600：用户消息降低亮度 */
    color: #f1f5f9 !important;
}

.dark .message.bot {
    background: #1e293b !important;
    color: #e2e8f0 !important; /* slate-200：正文，非纯白 */
    border-color: #334155 !important;
}

.dark .message.bot .message-content {
    color: #e2e8f0 !important;
}

.dark .message.bot pre {
    background: #0f172a !important; /* slate-900 代码块 */
    border-color: #334155;
}

.dark .message.bot p code {
    background: #334155 !important;
    color: #93c5fd !important; /* blue-300：柔和强调 */
    border-color: #475569;
}

.dark .input-area {
    background: #1e293b !important;
    border-color: #334155 !important;
}

.dark .textbox {
    background: #0f172a !important;
    border-color: #475569 !important;
    color: #e2e8f0 !important;
}

.dark .textbox:focus {
    border-color: #3b82f6 !important; /* blue-500 深色模式强调 */
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
}

.dark .button.secondary {
    background: #1e293b !important;
    color: #cbd5e1 !important;
    border-color: #475569 !important;
}
.dark .button.secondary:hover {
    background: #334155 !important;
    color: #f1f5f9 !important;
}

/* ========== 11. 响应式 ========== */
@media (max-width: 768px) {
    .gradio-container { width: 100% !important; padding: 1rem !important; }
    .header-title { font-size: 1.35rem !important; }
    #chatbot { height: 450px !important; }
    .message { max-width: 92% !important; font-size: 0.9rem !important; }
}
"""

with gr.Blocks(title="Agent 智能对话（人机协同审核）") as demo:
    with gr.Column(elem_classes="header-section"):
        gr.Markdown("## Agent 智能对话（人机协同审核）", elem_classes="header-title")
        gr.Markdown("支持安全审核、工具调用确认与参数编辑", elem_classes="header-subtitle")
        gr.Markdown(
            f"用户：`{USER_ID}`　|　会话：`{THREAD_ID}`",
            elem_classes="header-badge"
        )

    chatbot = gr.Chatbot(
        value=[],
        height=600,
        show_label=False,
        elem_id="chatbot",
        render_markdown=True,
    )

    with gr.Column(elem_classes="input-area"):
        msg = gr.Textbox(
            placeholder="输入问题，或在审核时回复「同意」「拒绝」「编辑 N {参数}」...",
            label="",
            lines=2,
            autofocus=True,
            elem_classes="textbox",
            show_label=False,
        )

        gr.Markdown(
            "出现工具调用审核时，直接在输入框回复指令即可，无需额外按钮",
            elem_classes="hint-text"
        )

        with gr.Row():
            send_btn = gr.Button("发送消息", variant="primary", scale=3, elem_classes="button primary")
            clear_btn = gr.Button("清空对话", variant="secondary", scale=1, elem_classes="button secondary")

    send_btn.click(
        fn=send_message,
        inputs=[msg, chatbot],
        outputs=chatbot
    ).then(
        fn=lambda: "",
        inputs=None,
        outputs=msg
    )

    clear_btn.click(
        clear_chat,
        outputs=chatbot
    )

if __name__ == "__main__":
    print(f"启动 Gradio 界面... FastAPI 后端地址：{BASE_URL}")
    print(f"当前会话 thread_id: {THREAD_ID}\n")
    print("审核提示：支持 '同意' '拒绝' '编辑 N {JSON}' '退出' '帮助' 等指令\n")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=theme,
        css=custom_css,
        inbrowser=True,
        share=False
    )
