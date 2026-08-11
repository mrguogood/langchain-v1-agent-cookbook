# Agent 的长期记忆

## 1、介绍

实现Agent的长期记忆(跨对话线程持久化存储)功能，包括长期记忆的写入和查询                                                      

核心功能包含：    

- PostgresStore，基于数据库的长期记忆持久化存储       
- 写入长期记忆        
- 读取长期记忆  

### 长期记忆
      
短期记忆允许应用程序在单个对话线程(thread)内记住之前的交互。对话历史是最常见的短期记忆形式          
长期记忆，指的是跨对话、跨会话都能复用的、持久化存储的用户或应用信息，而不是只跟当前对话轮次绑定的“上下文窗口”         

**长期记忆是什么：**    

- 存什么：用户画像（名字、偏好、历史决策）、业务配置、历史任务结果等，需要“下次还能记得”的信息
- 作用范围：可以跨 thread / 会话使用，同一个用户在不同对话里都能被读出来，而不是只在单个聊天线程中可见

**存储位置与形式：**      

- LangChain / LangGraph 把长期记忆抽象为一个 Store（键值存储），可以是内存、Postgres、向量库等，实现统一接口 put / get / search
- 数据通常按 namespace + key 组织，比如 ["memories", user_id] 作为命名空间，再用 uuid 作为具体记忆条目的 key，value 是包含实际信息的字典

            
## 2、功能测试   
                                
### 2.1 使用Docker方式运行PostgreSQL数据库     

进入官网 https://www.docker.com/ 下载安装Docker Desktop软件并安装，安装完成后打开软件                      

打开命令行终端，`cd 04_ShortTermMemory/postgresql` 文件夹下                     
- 进入到 postgresql 下执行 `docker-compose up -d` 运行 PostgreSQL 服务                            

运行成功后可在Docker Desktop软件中进行管理操作或使用命令行操作或使用指令                       
 
### 2.2 运行脚本测试            

```bash
python agent_PostgresStore.py
```