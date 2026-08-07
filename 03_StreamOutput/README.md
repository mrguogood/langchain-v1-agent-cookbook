# Agent 的三种主要调用方式

## 1、介绍

核心功能包含：    

- invoke,同步调用,返回完整响应 
- stream,流式输出,实时显示生成内容 
- batch,批量处理多个独立请求 

### invoke

invoke 是 LangChain 聊天模型最基础、最常用的同步调用方法,用于一次性发送请求并拿到完整回复        

**invoke 做什么**    

接受一条消息(字符串)或一组消息(对话历史),送入模型推理           
等模型生成完整答案后,返回一个 AIMessage 对象,里面包含文本内容等信息                  
适合大多数“问一句、拿一个完整答复”的场景,比如问答、翻译、写文案等                

### stream

stream 是聊天模型的流式输出调用方式,能在模型生成回复的过程中,把内容一段一段“边产边发”给你    

**stream 做什么**      

调用模型,返回一个迭代器,每次迭代拿到一个 AIMessageChunk, 里面是一小段文本或内容块      
非常适合长回复、需要“打字机效果”的场景,显著提升用户体验     
多个 chunk 可以累加成一个完整的 AIMessage, 之后像 invoke 的结果一样继续用于上下文    

**三种模式**      

(1)updates:Agent步骤级进度流              

在每一次Agent(step)执行后,把当前状态更新(state update)流给前端,而不是按 token 粒度输出      
如果同一个步骤里有多个节点(如 LLM 节点、Tool 节点)运行,每个节点的状态更新会单独发送一次       

典型应用场景:        

- 多工具/多步骤Agent: 想要看到 “调用了哪个工具 → 工具结果 → 最终回复” 这种步骤级时间线,而不是 token 级别细节
- 业务侧进度条/步骤可视化: 用于在 UI 上展示“当前正在执行: 模型推理 / 调用工具 / 汇总结果”等状态  
- Debug 与观测: 在 LangGraph / LangSmith 上调试复杂 agent 流程时,用 updates 看每次 state 变化最直观

(2)messages:LLM token 级流式输出           

从所有调用 LLM 的图节点里,以 (token, metadata) 元组的形式, 按生成顺序流式传输消息碎片(message chunks)        
对于工具调用,会先流出一连串 tool_call_chunk(部分JSON),再流出最终自然语言回复的 token 序列         

典型应用场景:       

- 聊天 UI 实时打字效果: 想要一边生成一边展示模型回答    
- 工具调用 JSON 增量可视化: 想在前端看到函数调用参数是如何逐渐被补全的  
- 多 LLM / 子Agent token 源区分: 配合 metadata["tags"] 或 subgraphs=True 可以标记 “当前 token 来自哪个子Agent/子图”  

常见组合用法:           
stream_mode=["messages", "updates"]                     
- 用 messages 流 token 供 UI 打字效果         
- 用 updates 获取完整、已解析的 AIMessage/ToolMessage(比如完整工具调用、工具结果),避免自己在前端拼 chunk            

(3)custom:任意自定义数据流         

在节点或工具内部通过 get_stream_writer() 获取一个流写入器(writer),把任意字符串或结构化数据推送到流中          
流出来的数据形式是你在代码里 writer(...) 写入的内容,跟 LLM message/状态无关              

典型应用场景:          

- 细粒度任务进度: 如 “正在拉取第 10/100 条记录”、“已完成向量检索,开始调用外部 API” 等业务日志              
- 模型长耗时操作回报: 比如爬虫、批量数据库查询、外部微服务调用,用 custom 给用户持续反馈        
- 高级 Guardrail / 中间评估结果曝光: 例如安全模型对主模型回答打分、结构化评估结果等,可直接通过 writer(result) 推给前端   

常见组用法:          
stream_mode=["updates", "custom"] 或 ["messages", "custom"]            
前者用于步骤+业务日志,后者用于 token+业务日志                   

**和 invoke 的区别(直观理解)**           

invoke: 一次性等模型“想完、说完”, 返回一个完整 AIMessage  
stream: 模型边“想”边“说”, 用 for 循环一块块接收 AIMessageChunk, 自己决定如何展示和是否合并       

想要“有打字进度条”的体验用 stream, 想要“结果直接给我”用 invoke            

### Batch

batch 是聊天模型的批量调用接口,用来一次性并行处理多条彼此独立的请求,从而提升吞吐量、降低总体等待时间和成本         

**batch 做什么**          

接收一个「输入列表」(每个元素是一条 prompt 或一段对话),并行发送给同一个模型             
返回一个「响应列表」,每个位置的响应对应输入列表中相同索引的那条输入     
适用于大量独立任务: 多条问答、多段文案生成、多条分类/抽取请求等          

**batch_as_completed 与并发控制**            

batch() 默认在所有请求完成后一次性返回整个列表         
想要“谁先算完先拿谁的结果”,可以用 batch_as_completed() 来流式拿回每条的结果,注意返回顺序可能与输入顺序不同,每个结果里会带上原始索引用于还原顺序            
处理很多请求时,可以通过 config = {"max_concurrency": N} 来限制最多并行多少个调用,从而避免打爆 provider 的限速或本地资源           

一条就用 invoke / stream, 一堆独立请求就用 batch / batch_as_completed                

## 4、功能测试   

运行脚本测试                               
```bash
python agent_invoke.py
python agent_stream.py 
python agent_batch.py
```