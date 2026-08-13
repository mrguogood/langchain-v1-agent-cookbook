# 从自定义配置模块导入 Config 类，用于读取模型类型等配置
from utils.config import Config
# 从自定义 LLM 工具模块导入 get_llm 方法，用于获取对话模型和向量模型实例
from utils.llms import get_llm
# 从 langchain_chroma 包中导入 Chroma，用于构建和使用基于 Chroma 的向量数据库/向量存储
from langchain_chroma import Chroma
# 从 langchain_community.document_loaders 中导入 PyPDFLoader，用于加载和解析 PDF 文档为可用的文本数据
from langchain_community.document_loaders import PyPDFLoader
# 从 langchain 导入递归字符文本分割器，用于将长文档切分成小块
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 从自定义日志模块导入 LoggerManager，用于获取日志记录器实例
from utils.logger import LoggerManager

# 获取全局日志记录器，用于输出运行过程中的日志信息
logger = LoggerManager.get_logger()

# 根据配置中指定的 LLM 类型，获取对话模型 llm_chat 和嵌入模型 llm_embedding 实例
llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

# 使用嵌入模型实例化内存向量数据库实例，用于存储文档向量
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=llm_embedding,
    persist_directory="./chroma_langchain_db",
)

# 1、加载文档
# 指定 PDF 文件的路径，这里文件名为“健康档案.pdf”，位于当前目录下
file_path = "./健康档案.pdf"
# 使用 PyPDFLoader 创建一个加载器，用于读取该 PDF 文件
loader = PyPDFLoader(file_path)
# 调用 load() 方法，将整份 PDF 加载为由多个 Document 对象组成的列表
docs = loader.load()
# 取出列表中的第一个 Document，一般对应 PDF 的第一页
first_doc = docs[0]
# 使用 len() 计算第一页文本内容的长度（字符数），并打印“文档字数总计
print(f"文档字数总计: {len(first_doc.page_content)} \n")
logger.info(f"文档字数总计: {len(first_doc.page_content)}")
# 截取第一页文本内容的前 100 个字符，用于快速预览文档内容，并打印出来
print(f"文档前100个字符：{first_doc.page_content[:100]} \n")
logger.info(f"文档前100个字符：{first_doc.page_content[:100]}")
# 打印第一页对应的元数据信息（如作者、标题、创建时间、页码等），便于了解 PDF 的结构和属性
print(f"文档元数据信息：{first_doc.metadata} \n")
logger.info(f"文档元数据信息：{first_doc.metadata}")

# 2、切分文档
# 创建递归字符文本分割器实例，设置切分参数
# 每个文档块的大小为 1000 个字符
# 文档块之间重叠 200 个字符，确保上下文连贯性
# 添加起始索引，跟踪每个块在原文档中的位置
# 定义分隔符优先级列表，按从高到低的优先级依次尝试分割文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    add_start_index=True,
    separators=[
        # 优先级最高：双换行符（段落分隔）
        "\n\n",
        # 单换行符（行分隔）
        "\n",
        # 中文句号
        "。",
        # 中文感叹号
        "！",
        # 中文问号
        "？",
        # 英文感叹号
        "!",
        # 英文问号
        "?",
        # 英文句号
        ".",
        # 中文分号
        "；",
        # 英文分号
        ";",
        # 中文逗号
        "，",
        # 英文逗号
        ",",
        # 中文冒号
        "：",
        # 英文冒号
        ":",
        # 空格
        " ",
        # 优先级最低：空字符串（强制按字符切分）
        ""
    ]
)
# 使用文本分割器将文档切分成多个小块
all_splits = text_splitter.split_documents(docs)
# 打印切分后的文档块数量
print(f"文档一共切分了 {len(all_splits)} 个文本块 \n")
logger.info(f"文档一共切分了 {len(all_splits)} 个文本块")

# 3、创建索引并写入向量数据库
# 将切分后的文档块添加到向量数据库中，并返回每个文档的 ID
document_ids = vector_store.add_documents(documents=all_splits)
# 打印前 3 个文档的 ID
print(f"前3个文档的ID：{document_ids[:3]}")
logger.info(f"前3个文档的ID：{document_ids[:3]}")

