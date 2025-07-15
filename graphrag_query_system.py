#!/usr/bin/env python3
"""
GraphRAG查询验证系统
实现全局搜索和局部搜索功能
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import List, Dict, Any
from collections import Counter
import json

class GraphRAGQueryEngine:
    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.load_data()
    
    def load_data(self):
        """加载所有GraphRAG输出数据"""
        try:
            self.entities_df = pd.read_parquet(self.output_dir / "entities.parquet")
            self.relationships_df = pd.read_parquet(self.output_dir / "relationships.parquet")
            self.communities_df = pd.read_parquet(self.output_dir / "communities.parquet")
            self.community_reports_df = pd.read_parquet(self.output_dir / "community_reports.parquet")
            self.text_units_df = pd.read_parquet(self.output_dir / "text_units.parquet")
            self.documents_df = pd.read_parquet(self.output_dir / "documents.parquet")
            print("✅ 所有数据加载成功")
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
    
    def global_search(self, query: str) -> Dict[str, Any]:
        """
        全局搜索：基于社区报告进行查询
        类似于GraphRAG的global search模式
        """
        print(f"\n🌍 全局搜索: '{query}'")
        print("=" * 60)
        
        # 将查询转换为关键词
        query_keywords = self._extract_keywords(query)
        print(f"🔍 提取关键词: {query_keywords}")
        
        # 搜索社区报告
        relevant_reports = []
        for idx, report in self.community_reports_df.iterrows():
            relevance_score = self._calculate_relevance(
                query_keywords, 
                str(report.get('full_content', '')) + " " + str(report.get('summary', ''))
            )
            
            if relevance_score > 0:
                relevant_reports.append({
                    'community_id': report.get('community', 'N/A'),
                    'title': report.get('title', 'N/A'),
                    'summary': report.get('summary', 'N/A'),
                    'full_content': report.get('full_content', 'N/A'),
                    'relevance_score': relevance_score
                })
        
        # 按相关性排序
        relevant_reports.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # 生成全局答案
        global_answer = self._generate_global_answer(query, relevant_reports)
        
        return {
            'query': query,
            'method': 'global',
            'relevant_communities': len(relevant_reports),
            'reports': relevant_reports,
            'answer': global_answer
        }
    
    def local_search(self, query: str, entity_name: str = None) -> Dict[str, Any]:
        """
        局部搜索：基于特定实体及其邻居进行查询
        类似于GraphRAG的local search模式
        """
        print(f"\n🎯 局部搜索: '{query}'")
        if entity_name:
            print(f"🎯 聚焦实体: '{entity_name}'")
        print("=" * 60)
        
        # 如果没有指定实体，从查询中提取
        if not entity_name:
            entity_name = self._find_relevant_entity(query)
            print(f"🔍 自动识别实体: '{entity_name}'")
        
        # 找到目标实体
        target_entity = self._get_entity_info(entity_name)
        if not target_entity:
            print(f"❌ 未找到实体: {entity_name}")
            return {'error': f'Entity not found: {entity_name}'}
        
        # 获取实体的邻居和关系
        neighbors = self._get_entity_neighbors(entity_name)
        relationships = self._get_entity_relationships(entity_name)
        
        # 获取相关文本块
        relevant_texts = self._get_relevant_text_units(query, entity_name)
        
        # 生成局部答案
        local_answer = self._generate_local_answer(query, target_entity, neighbors, relationships, relevant_texts)
        
        return {
            'query': query,
            'method': 'local',
            'target_entity': target_entity,
            'neighbors': neighbors,
            'relationships': relationships,
            'relevant_texts': relevant_texts,
            'answer': local_answer
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        # 简单的关键词提取
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        # 过滤停用词
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'when', 'where', 'why'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def _calculate_relevance(self, keywords: List[str], text: str) -> float:
        """计算关键词与文本的相关性分数"""
        text_lower = text.lower()
        score = 0
        for keyword in keywords:
            # 精确匹配
            if keyword in text_lower:
                score += 1
            # 部分匹配
            elif any(keyword in word for word in text_lower.split()):
                score += 0.5
        return score
    
    def _find_relevant_entity(self, query: str) -> str:
        """从查询中找到最相关的实体"""
        query_keywords = self._extract_keywords(query)
        best_entity = None
        best_score = 0
        
        for idx, entity in self.entities_df.iterrows():
            entity_text = f"{entity.get('title', '')} {entity.get('description', '')}"
            score = self._calculate_relevance(query_keywords, entity_text)
            if score > best_score:
                best_score = score
                best_entity = entity.get('title', '')
        
        return best_entity or "LLMS"  # 默认返回LLMs
    
    def _get_entity_info(self, entity_name: str) -> Dict[str, Any]:
        """获取实体详细信息"""
        entity_row = self.entities_df[self.entities_df['title'].str.contains(entity_name, case=False, na=False)]
        if entity_row.empty:
            return None
        
        entity = entity_row.iloc[0]
        return {
            'name': entity.get('title', ''),
            'type': entity.get('type', ''),
            'description': entity.get('description', ''),
            'community_ids': entity.get('community_ids', [])
        }
    
    def _get_entity_neighbors(self, entity_name: str) -> List[Dict[str, Any]]:
        """获取实体的邻居节点"""
        neighbors = []
        
        # 从关系中找到连接的实体
        related_rels = self.relationships_df[
            (self.relationships_df['source'].str.contains(entity_name, case=False, na=False)) |
            (self.relationships_df['target'].str.contains(entity_name, case=False, na=False))
        ]
        
        neighbor_names = set()
        for idx, rel in related_rels.iterrows():
            source = rel.get('source', '')
            target = rel.get('target', '')
            if entity_name.lower() in source.lower():
                neighbor_names.add(target)
            else:
                neighbor_names.add(source)
        
        # 获取邻居实体的详细信息
        for neighbor_name in neighbor_names:
            neighbor_info = self._get_entity_info(neighbor_name)
            if neighbor_info:
                neighbors.append(neighbor_info)
        
        return neighbors
    
    def _get_entity_relationships(self, entity_name: str) -> List[Dict[str, Any]]:
        """获取实体的所有关系"""
        relationships = []
        
        related_rels = self.relationships_df[
            (self.relationships_df['source'].str.contains(entity_name, case=False, na=False)) |
            (self.relationships_df['target'].str.contains(entity_name, case=False, na=False))
        ]
        
        for idx, rel in related_rels.iterrows():
            relationships.append({
                'source': rel.get('source', ''),
                'target': rel.get('target', ''),
                'description': rel.get('description', ''),
                'weight': rel.get('weight', 1.0)
            })
        
        return relationships
    
    def _get_relevant_text_units(self, query: str, entity_name: str) -> List[Dict[str, Any]]:
        """获取与查询和实体相关的文本块"""
        query_keywords = self._extract_keywords(query)
        relevant_texts = []
        
        for idx, text_unit in self.text_units_df.iterrows():
            text_content = str(text_unit.get('text', ''))
            
            # 计算相关性（同时考虑查询和实体）
            query_score = self._calculate_relevance(query_keywords, text_content)
            entity_score = 1 if entity_name.lower() in text_content.lower() else 0
            total_score = query_score + entity_score
            
            if total_score > 0:
                relevant_texts.append({
                    'text': text_content,
                    'document_ids': text_unit.get('document_ids', []),
                    'relevance_score': total_score
                })
        
        # 按相关性排序
        relevant_texts.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_texts[:5]  # 返回前5个最相关的
    
    def _generate_global_answer(self, query: str, reports: List[Dict]) -> str:
        """基于社区报告生成全局答案"""
        if not reports:
            return "未找到相关信息。"
        
        # 合并所有相关报告的内容
        all_content = []
        for report in reports[:3]:  # 使用前3个最相关的报告
            all_content.append(f"社区 {report['community_id']}: {report['summary']}")
        
        # 简单的答案生成（实际应用中可以使用LLM）
        answer = f"基于社区分析，关于 '{query}' 的主要发现：\n\n"
        for i, content in enumerate(all_content, 1):
            answer += f"{i}. {content}\n\n"
        
        return answer.strip()
    
    def _generate_local_answer(self, query: str, entity: Dict, neighbors: List, relationships: List, texts: List) -> str:
        """基于局部图信息生成答案"""
        answer = f"关于实体 '{entity['name']}' 在 '{query}' 方面的信息：\n\n"
        
        # 实体描述
        answer += f"📝 实体描述：{entity['description']}\n\n"
        
        # 相关关系
        if relationships:
            answer += "🔗 相关关系：\n"
            for rel in relationships[:3]:
                answer += f"  • {rel['source']} → {rel['target']}: {rel['description']}\n"
            answer += "\n"
        
        # 邻居实体
        if neighbors:
            answer += "🌐 关联实体：\n"
            for neighbor in neighbors[:3]:
                answer += f"  • {neighbor['name']} ({neighbor['type']}): {neighbor['description'][:100]}...\n"
            answer += "\n"
        
        # 相关文本
        if texts:
            answer += "📄 相关文本片段：\n"
            for text in texts[:2]:
                answer += f"  • {text['text'][:150]}...\n"
        
        return answer

    def run_comprehensive_test(self):
        """运行综合测试"""
        print("🧪 GraphRAG 查询系统综合测试")
        print("=" * 80)
        
        # 测试用例
        test_cases = [
            {
                'type': 'global',
                'query': 'What are the main types of biases in LLMs?',
                'description': '全局搜索：LLM中的主要偏见类型'
            },
            {
                'type': 'global', 
                'query': 'How can bias in language models be evaluated?',
                'description': '全局搜索：如何评估语言模型中的偏见'
            },
            {
                'type': 'local',
                'query': 'What biases are associated with LLMs?',
                'entity': 'LLMS',
                'description': '局部搜索：LLMs实体的偏见相关信息'
            },
            {
                'type': 'local',
                'query': 'How does training data affect model bias?',
                'entity': 'TRAINING DATA', 
                'description': '局部搜索：训练数据如何影响模型偏见'
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 测试 {i}: {test_case['description']}")
            print("-" * 50)
            
            try:
                if test_case['type'] == 'global':
                    result = self.global_search(test_case['query'])
                else:
                    result = self.local_search(test_case['query'], test_case.get('entity'))
                
                # 显示结果
                print(f"🎯 查询: {result.get('query', 'N/A')}")
                print(f"📊 方法: {result.get('method', 'N/A')}")
                print(f"📄 答案:\n{result.get('answer', 'N/A')}")
                
                results.append(result)
                
            except Exception as e:
                print(f"❌ 测试失败: {e}")
                results.append({'error': str(e)})
        
        return results

def main():
    """主函数"""
    try:
        # 初始化查询引擎
        query_engine = GraphRAGQueryEngine()
        
        # 运行综合测试
        results = query_engine.run_comprehensive_test()
        
        print("\n" + "=" * 80)
        print("✅ 测试完成！")
        print(f"📊 成功执行 {len([r for r in results if 'error' not in r])} 个查询")
        print(f"❌ 失败 {len([r for r in results if 'error' in r])} 个查询")
        
        # 提供交互式查询选项
        print("\n💡 您可以尝试以下交互式查询:")
        print("query_engine.global_search('your query here')")
        print("query_engine.local_search('your query here', 'ENTITY_NAME')")
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")

if __name__ == "__main__":
    main()