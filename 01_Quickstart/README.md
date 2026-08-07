# LangChain 最新版本 V1.x 快速入门

## 1、介绍

用例的核心功能包含：

- 各厂商LLM大模型集成
- System Prompt 定义
- Tools 工具定义
- Models 数据模型定义
- Agent 定义和运行
- Agent 短期记忆(内存)
- Agent 工具调用
- Agent 结构化输出
- LangSmith 跟踪观测

### 什么是 Agents        

Agents 将 LLM 大语言模型与 Tools 工具结合，创建能够推理任务、决定使用哪些工具并迭代寻找解决方案的系统                
Agent遵循 ReAct ("推理 + 行动")模式，在满足停止条件前循环运行工具以实现目标             

官方介绍链接:https://docs.langchain.com/oss/python/langchain/agents              

#### 核心组件

1. 模型(Model) 

- 支持静态模型配置         
- 支持动态模型选择(通过中间件在运行时根据对话复杂度等因素切换模型)         

2. 工具(Tools) 

- 为Agent提供执行操作的能力
- 支持多个工具顺序调用、并行调用、动态工具选择及错误处理
- 可通过 @tool 装饰器自定义工具属性

3. 系统Prompt(System Prompt)
- 可使用字符串或 SystemMessage 定义Agent行为方式
- 支持动态系统Prompt

#### 高级功能

1. 结构化输出(2种策略)
- ToolStrategy: 通过人工工具调用生成结构化输出,适用于所有支持工具调用的模型
- ProviderStrategy: 使用模型提供商的原生结构化输出功能,更可靠但依赖提供商支持
- with_structured_output: 为Agent添加结构化输出功能,支持自定义输出格式

2. 记忆(Memory)
- 通过消息状态自动维护对话历史
- 支持自定义状态模式以存储额外信息,作为Agent的短期记忆

3. 流式传输(Streaming)
- 支持通过 stream 方法实时返回中间步骤和消息

4. 中间件(Middleware)
- 在执行的不同阶段自定义Agent行为
- 可用于消息修剪、内容过滤、错误处理、动态模型选择等场景                  

## 3、功能测试   

- 运行脚本 `python agent.py`                                   