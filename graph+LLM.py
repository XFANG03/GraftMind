# 知识图谱代码（智能更新的精简版）
import json
import ast
from neo4j import GraphDatabase
from openai import OpenAI
import time
from typing import List, Dict
import requests
import os


class ContentFocusedKG:
    def __init__(self, uri, username, password, openai_api_key=None, tongyi_api_key=None):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

        # 设置OpenAI API
        if openai_api_key:
            self.client = OpenAI(
                api_key=openai_api_key,
                base_url="https://api.chatanywhere.tech/v1"
            )
        else:
            self.client = None

        # 设置通义千问API
        self.tongyi_api_key = tongyi_api_key

    def close(self):
        self.driver.close()

    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            return session.run(query, parameters).data()

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def parse_dialog(self, dialog_str):
        """解析对话字符串，支持多种格式"""
        if not dialog_str or dialog_str == "[]":
            return []

        try:
            # 尝试JSON格式
            parsed = json.loads(dialog_str)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            try:
                # 尝试Python字面量格式（处理单引号的情况）
                parsed = ast.literal_eval(dialog_str)
                return parsed if isinstance(parsed, list) else []
            except (ValueError, SyntaxError):
                return []

    def summarize_content_with_dialogs(self, content: str, dialog_messages: List[Dict]) -> str:
        """使用LLM总结Content和所有Dialog的综合信息，生成100字总结"""
        # 构建完整的内容和对话文本
        full_text = f"笔记内容: {content}\n\n"

        if dialog_messages:
            full_text += "相关对话:\n"
            for i, msg in enumerate(dialog_messages, 1):
                role = msg.get('role', 'user')
                msg_content = msg.get('content', '')
                role_name = "用户" if role == 'user' else "AI助手"
                full_text += f"{role_name}: {msg_content}\n"

        # 调用OpenAI API进行综合总结
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "请对以下笔记内容和相关对话进行综合总结，重点概括核心主题、关键观点和讨论要点。总结应该是中文，控制在100字以内，要完整体现内容的核心价值。"
                },
                {
                    "role": "user",
                    "content": full_text
                }
            ],
            max_tokens=150,
            temperature=0.3
        )

        summary = response.choices[0].message.content.strip()
        time.sleep(0.1)
        return summary

    def calculate_tongyi_similarity(self, text1: str, text2: str, max_retries: int = 3) -> float:
        """使用通义千问HTTP API计算两个文本的相似度（带重试机制）"""
        url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        headers = {
            "Authorization": f"Bearer {self.tongyi_api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "gte-rerank",
            "input": {
                "query": text1,
                "documents": [text2]
            },
            "parameters": {
                "return_documents": True,
                "top_n": 1
            }
        }

        response = requests.post(url, headers=headers, json=data, timeout=30)
        result = response.json()

        results = result["output"]["results"]
        similarity = float(results[0]["relevance_score"])
        return max(0.0, min(1.0, similarity))

    def calculate_all_nodes_similarity_with_tongyi(self, session):
        """使用通义千问rerank API计算所有节点之间的相似度"""
        content_nodes = self.execute_query("""
            MATCH (c:Content)
            RETURN c.id as id, c.summary as content, 'Content' as node_type
        """)

        all_nodes = content_nodes

        if len(all_nodes) < 2:
            return

        valid_nodes = []
        for node in all_nodes:
            content_text = node.get('content', '').strip()
            if content_text:
                valid_nodes.append({
                    'id': node.get('id'),
                    'type': node.get('node_type'),
                    'content': content_text
                })

        if len(valid_nodes) < 2:
            return

        similarity_count = 0

        for i in range(len(valid_nodes)):
            for j in range(i + 1, len(valid_nodes)):
                node1 = valid_nodes[i]
                node2 = valid_nodes[j]

                similarity_score = self.calculate_tongyi_similarity(
                    node1['content'],
                    node2['content']
                )

                if similarity_score > 0.5:
                    distance_weight = 1.0 - similarity_score

                    self.create_similarity_relationship(
                        session, node1, node2, similarity_score, distance_weight
                    )
                    similarity_count += 1

                # 增加延迟以避免API限流
                time.sleep(0.2)

    def create_similarity_relationship(self, session, node1, node2, similarity_score, distance_weight):
        """根据节点类型创建相似性关系"""
        node1_id = node1['id']
        node2_id = node2['id']

        # 现在只有Content to Content关系
        query = """
            MATCH (n1:Content {id: $node1_id}), (n2:Content {id: $node2_id})
            MERGE (n1)-[:SIMILAR_CONTENT {
                similarity: $similarity,
                weight: $weight,
                distance: $distance,
                type: 'content_to_content',
                method: 'tongyi_rerank'
            }]-(n2)
        """

        session.run(query,
                    node1_id=node1_id,
                    node2_id=node2_id,
                    similarity=similarity_score,
                    weight=distance_weight,
                    distance=distance_weight)

    def update_graph(self, json_file_path, force_full_rebuild=False):
        """智能更新知识图谱（自动判断全量导入或增量更新）"""

        # 读取新数据
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        new_notes = data.get('output', [])

        if not new_notes:
            print("📭 没有新内容需要更新")
            return False  # ✅ 明确返回 False

        # 检查数据库是否为空或强制重建
        existing_count = self.execute_query("MATCH (c:Content) RETURN count(c) as count")[0]['count']
        is_database_empty = existing_count == 0

        if is_database_empty or force_full_rebuild:
            # 全量导入模式
            print(f"🔄 执行全量导入 - 数据库{'为空' if is_database_empty else '强制重建'}")

            if not is_database_empty:
                self.clear_database()

            with self.driver.session() as session:
                for note in new_notes:
                    self.process_content_node(session, note)
                print(f"✅ 创建了 {len(new_notes)} 个节点")
                self.calculate_all_nodes_similarity_with_tongyi(session)

            print("✅ 全量导入完成！")
            return True  # ✅ 添加返回值
        else:
            # 增量更新模式
            print("🔄 执行增量更新")

            # 获取数据库中已存在的Content ID
            existing_ids = set()
            existing_content_query = self.execute_query("MATCH (c:Content) RETURN c.id as id")
            for item in existing_content_query:
                existing_ids.add(item['id'])

            # 只处理新增数据，跳过已存在的数据
            new_notes_to_add = []
            skipped_count = 0

            for note in new_notes:
                note_id = note.get('note_id')
                if note_id in existing_ids:
                    skipped_count += 1
                else:
                    new_notes_to_add.append(note)

            print(f"🆕 新增: {len(new_notes_to_add)} 条，⏭️ 跳过已存在: {skipped_count} 条")

            if not new_notes_to_add:
                print("📭 没有新内容需要添加")
                print("✅ 增量更新完成！")
                return False  # ✅ 明确返回 False

            # 只创建新节点
            with self.driver.session() as session:
                for note in new_notes_to_add:
                    self.process_content_node(session, note)

            # 只为新增节点计算相似度
            print("⚡ 开始为新增节点计算相似度...")

            # 获取所有节点（包括新增的）
            all_nodes = self.execute_query("""
                MATCH (c:Content)
                RETURN c.id as id, c.summary as content
            """)

            # 分类节点
            new_node_ids = set(note.get('note_id') for note in new_notes_to_add)
            new_nodes = []
            existing_nodes = []

            for node in all_nodes:
                content_text = node.get('content', '').strip()
                if content_text:
                    node_info = {
                        'id': node.get('id'),
                        'type': 'Content',
                        'content': content_text
                    }

                    if node.get('id') in new_node_ids:
                        new_nodes.append(node_info)
                    else:
                        existing_nodes.append(node_info)

            similarity_count = 0

            with self.driver.session() as similarity_session:
                # 计算新增节点与已存在节点的相似度
                for new_node in new_nodes:
                    for existing_node in existing_nodes:
                        similarity_score = self.calculate_tongyi_similarity(
                            new_node['content'], existing_node['content']
                        )

                        if similarity_score > 0.5:
                            distance_weight = 1.0 - similarity_score
                            self.create_similarity_relationship(
                                similarity_session, new_node, existing_node, similarity_score, distance_weight
                            )
                            similarity_count += 1

                        time.sleep(0.2)

                # 计算新增节点之间的相似度
                for i in range(len(new_nodes)):
                    for j in range(i + 1, len(new_nodes)):
                        similarity_score = self.calculate_tongyi_similarity(
                            new_nodes[i]['content'], new_nodes[j]['content']
                        )

                        if similarity_score > 0.5:
                            distance_weight = 1.0 - similarity_score
                            self.create_similarity_relationship(
                                similarity_session, new_nodes[i], new_nodes[j], similarity_score, distance_weight
                            )
                            similarity_count += 1

                        time.sleep(0.2)

                print(f"⚡ 新建 {similarity_count} 个相似性关系")

            print("✅ 增量更新完成！")
            return True

    def process_content_node(self, session, note):
        """处理内容节点 - 将Content和Dialog合并总结"""
        content = note.get('content', '')
        dialog = self.parse_dialog(note.get('dialog', '[]'))
        note_id = note.get('note_id')

        # 生成Content和Dialog的综合总结（100字）
        comprehensive_summary = self.summarize_content_with_dialogs(content, dialog)

        # 创建内容节点，使用综合总结作为主要内容
        session.run("""
            CREATE (c:Content {
                id: $note_id,
                content: $original_content,
                summary: $comprehensive_summary,
                dialog_count: $dialog_count,
                user_id: $user_id,
                group_id: $group_id,
                position_x: $x_pos,
                position_y: $y_pos,
                deleted: $deleted,
                history_content: $history
            })
        """, note_id=note_id,
                    original_content=content,
                    comprehensive_summary=comprehensive_summary,
                    dialog_count=len(dialog),
                    user_id=note.get('user_id'),
                    group_id=note.get('group_id'),
                    x_pos=note.get('position', {}).get('x', 0),
                    y_pos=note.get('position', {}).get('y', 0),
                    deleted=note.get('deleted', False),
                    history=json.dumps(note.get('history_content', []), ensure_ascii=False))

    def analyze_similarity_with_llm(self, content1_summary: str, content2_summary: str, similarity_score: float) -> str:
        """使用LLM分析两个内容之间的相似性关系"""
        if not self.client:
            return f"两个内容的相似度为{similarity_score:.3f}，存在一定程度的语义关联。"

        analysis_prompt = f"""
        内容1: {content1_summary}
        内容2: {content2_summary}
        相似度分数: {similarity_score:.3f}

        请分析这两个内容之间的相似性，说明它们在哪些方面相关联，为什么会有这样的相似度。
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "请分析两个内容之间的相似性关系，重点说明它们的关联点和相似原因。回复应该是中文，控制在50字以内，要具体且有分析价值。"
                },
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ],
            max_tokens=80,
            temperature=0.3
        )

        analysis = response.choices[0].message.content.strip()
        time.sleep(0.1)
        return analysis

    def incremental_export_knowledge_graph(self, output_dir="./output/"):
        """增量导出知识图谱（只分析新增的相似性关系）"""

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 检查是否已存在导出文件
        main_file = os.path.join(output_dir, 'knowledge_graph_complete.json')
        existing_relationships = set()

        if os.path.exists(main_file):
            try:
                with open(main_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    for rel in existing_data.get('relationships', []):
                        # 使用排序后的ID对作为关系的唯一标识
                        id1, id2 = sorted([rel['content1_id'], rel['content2_id']])
                        existing_relationships.add(f"{id1}_{id2}")
                print(f"📄 检测到已有导出文件，包含 {len(existing_relationships)} 个已分析的关系")
            except Exception as e:
                print(f"⚠️ 读取已有文件失败: {e}")
                existing_relationships = set()

        # 获取所有Content节点
        content_nodes = self.execute_query("""
            MATCH (c:Content)
            RETURN c.id as id, c.content as content, c.summary as summary,
                   c.dialog_count as dialog_count, c.user_id as user_id,
                   c.group_id as group_id, c.position_x as position_x,
                   c.position_y as position_y, c.deleted as deleted,
                   c.history_content as history_content
            ORDER BY c.id
        """)

        # 获取所有相似性关系
        all_relationships = self.execute_query("""
            MATCH (c1:Content)-[r:SIMILAR_CONTENT]-(c2:Content)
            WHERE c1.id < c2.id  // 避免重复
            RETURN c1.id as content1_id, c1.summary as content1_summary,
                   c2.id as content2_id, c2.summary as content2_summary,
                   r.similarity as similarity, r.weight as weight,
                   r.distance as distance, r.type as type, r.method as method
            ORDER BY r.similarity DESC
        """)

        # 识别新增的关系
        new_relationships = []
        existing_analyzed_relationships = []

        for rel in all_relationships:
            id1, id2 = sorted([rel['content1_id'], rel['content2_id']])
            rel_key = f"{id1}_{id2}"

            if rel_key in existing_relationships:
                # 这是已经分析过的关系，需要保留原有分析
                existing_analyzed_relationships.append(rel)
            else:
                # 这是新的关系，需要进行LLM分析
                new_relationships.append(rel)

        if not new_relationships:
            print("📭 没有新的相似性关系需要分析")
            return

        print(f"🆕 发现 {len(new_relationships)} 个新的相似性关系需要分析...")
        print(f"📋 保留 {len(existing_analyzed_relationships)} 个已分析的关系")

        # 只为新关系生成LLM分析
        enhanced_new_relationships = []
        for i, rel in enumerate(new_relationships):
            print(f"分析进度: {i + 1}/{len(new_relationships)}")

            llm_analysis = self.analyze_similarity_with_llm(
                rel['content1_summary'],
                rel['content2_summary'],
                rel['similarity']
            )

            enhanced_rel = {
                'content1_id': rel['content1_id'],
                'content2_id': rel['content2_id'],
                'similarity': rel['similarity'],
                'weight': rel['weight'],
                'distance': rel['distance'],
                'type': rel['type'],
                'method': rel['method'],
                'llm_analysis': llm_analysis,
                'content1_summary': rel['content1_summary'],
                'content2_summary': rel['content2_summary']
            }
            enhanced_new_relationships.append(enhanced_rel)

        # 如果有已存在的分析结果，需要读取并合并
        all_enhanced_relationships = []

        if existing_relationships and os.path.exists(main_file):
            try:
                with open(main_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    all_enhanced_relationships = existing_data.get('relationships', [])
                print(f"📋 加载了 {len(all_enhanced_relationships)} 个已有分析结果")
            except Exception as e:
                print(f"⚠️ 加载已有分析结果失败: {e}")

        # 添加新分析的关系
        all_enhanced_relationships.extend(enhanced_new_relationships)

        # 构建完整的图谱数据结构
        knowledge_graph = {
            'nodes': content_nodes,
            'relationships': all_enhanced_relationships
        }

        # 导出主文件
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_graph, f, ensure_ascii=False, indent=2)

        # 导出节点文件
        nodes_file = os.path.join(output_dir, 'nodes.json')
        with open(nodes_file, 'w', encoding='utf-8') as f:
            json.dump(content_nodes, f, ensure_ascii=False, indent=2)

        # 导出关系文件
        relationships_file = os.path.join(output_dir, 'relationships.json')
        with open(relationships_file, 'w', encoding='utf-8') as f:
            json.dump(all_enhanced_relationships, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 增量导出完成！")
        print(f"📁 输出目录: {output_dir}")
        print(
            f"📄 完整版本: knowledge_graph_complete.json ({len(content_nodes)} 节点, {len(all_enhanced_relationships)} 关系)")
        print(f"🆕 新增分析: {len(enhanced_new_relationships)} 个关系")
        print(f"📄 节点文件: nodes.json")
        print(f"📄 关系文件: relationships.json")


def main():
    NEO4J_URI = "neo4j+s://9cf25794.databases.neo4j.io"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "NRuAyaDCUhFxXImdnWHBszPdyW5Ux6MHDBl9jwnOj5g"
    OPENAI_API_KEY = "sk-guFVSLItjWTGMkkeSjwKTFQqSa23H5lsLqbYdHZqBboz6k6D"
    TONGYI_API_KEY = ("sk-3160bc2566fe45ba9348a789c5316bbc")

    kg = ContentFocusedKG(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_KEY, TONGYI_API_KEY)

    # 智能更新使用方式：
    # 1. 自动判断模式（推荐）- 数据库为空时全量导入，否则增量更新
    has_updates = kg.update_graph('./data/sample.json')

    # 2. 强制全量重建
    # has_updates = kg.update_graph('./data/sample.json', force_full_rebuild=True)

    # 3. 增量导出知识图谱（只在有更新时进行导出，且只分析新增的关系）
    if has_updates:
        print("🔄 开始增量导出...")
        kg.incremental_export_knowledge_graph("./knowledge_graph_output/")
    else:
        print("⏭️ 无更新内容，跳过导出")

    kg.close()


if __name__ == "__main__":
    main()