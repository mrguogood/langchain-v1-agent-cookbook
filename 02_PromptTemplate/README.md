# 定义和使用prompt模版      

## 1、介绍

实现将 prompt 以模版文件(txt、md)形式单独进行维护，并在Agent中加载模版使用                 


核心功能包含：

- 系统 prompt模版文件定义  
- 用户 prompt模版文件定义
- prompt 模版定义变量
- prompt 变量动态传参
- Agent 加载Prompt模版使用

## 2、项目结构

```
02_PromptTemplate/
├── agent.py                          # 主入口：创建 Agent、加载提示词、执行对话
├── README.md                         # 项目说明文档
├── prompt/                           # 提示词模板目录
│   ├── system_prompt_tmpl.md         # 系统提示词模板（角色、工作流程、工具规则）--新增
│   └── human_prompt_tmpl.md          # 用户提示词模板（{name}、{question} 占位符）--新增
└── utils/                            # 工具模块目录
    ├── config.py                     # 配置类（模型类型、提示词文件路径等）
    ├── llms.py                       # LLM 工厂方法（获取对话模型和嵌入模型）
    ├── logger.py                     # 日志管理器（统一日志输出）
    ├── models.py                     # 数据模型（Context、ResponseFormat）
    ├── tools.py                      # 工具定义（get_weather_for_location、get_user_location）               
```

## 3、功能测试   

- 运行脚本 `python agent.py`                                  