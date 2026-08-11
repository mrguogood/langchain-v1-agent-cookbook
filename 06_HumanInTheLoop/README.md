# Agent 的人机交互

## 1、介绍

实现Agent的Human-in-the-Loop (HITL)，也就是人机协作、人在环中、人工介入循环                                                                          

核心功能包含：    

- 在invoke中使用HITL
- 在stream中使用HITL

### Human-in-the-Loop

HITL中间件可以在模型执行敏感操作(如写入文件或执行SQL)前暂停执行并等待人工审核           
该系统通过检查每个工具调用是否符合预设策略来决定是否需要干预,当需要时会发出中断信号暂停执行,利用LangGraph的持久化层保存图状态           

**人工审核者可以对中断做出三种响应：**     

- 批准(approve): 按原样执行操作,不做任何更改
- 编辑(edit): 修改参数后再执行工具调用
- 拒绝(reject): 拒绝该操作并提供反馈说明

**配置方式：**          

- 使用HITL需要在创建 Agent 时添加 HumanInTheLoopMiddleware 中间件,并配置 interrupt_on 映射来指定哪些工具需要人工审核以及允许的决策类型               
- 该功能必须配置检查点保存器(checkpointer)来持久化图状态                                     

## 2、功能测试   

### 2.1 使用Docker方式运行PostgreSQL数据库     

进入官网 https://www.docker.com/ 下载安装Docker Desktop软件并安装，安装完成后打开软件                      

打开命令行终端，`cd 04_ShortTermMemory/postgresql` 文件夹下                     
- 进入到 postgresql 下执行 `docker-compose up -d` 运行 PostgreSQL 服务                            

运行成功后可在Docker Desktop软件中进行管理操作或使用命令行操作或使用指令                                 

### 2.2 运行脚本测试            

```bash
python agent_invoke_hitl.py
python agent_stream_hitl.py
```
