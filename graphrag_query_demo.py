# graphrag_query_demo.py
from graphrag import GraphRag

def main():
    # 替换为你的实际项目根目录
    root_dir = "/autodl-tmp/graftmind"

    # 初始化 GraphRAG 实例
    rag = GraphRag(root=root_dir)

    # 示例 Local 查询
    question_local = "What is the main function of the Shared Context mechanism?"
    response_local = rag.query_local(question_local)
    print("\n[Local Query Result]")
    print(response_local.answer)

    # 示例 Global 查询
    question_global = "Summarize the main features of the GraftMind system."
    response_global = rag.query_global(question_global)
    print("\n[Global Query Result]")
    print(response_global.answer)

if __name__ == "__main__":
    main()
