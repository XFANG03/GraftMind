#!/usr/bin/env python3
"""
GraphRAG查询脚本
直接使用GraphRAG API进行查询，避免CLI问题
"""

import os
import pandas as pd
import asyncio
from pathlib import Path

def check_output_files():
    """检查GraphRAG输出文件"""
    output_dir = Path("output")
    if not output_dir.exists():
        print("❌ output目录不存在")
        return False
    
    print("📁 Output目录内容:")
    files = list(output_dir.glob("*.parquet"))
    for file in files:
        size = file.stat().st_size / 1024  # KB
        print(f"  📄 {file.name} ({size:.1f} KB)")
    
    required_files = [
        "entities.parquet",
        "relationships.parquet", 
        "communities.parquet",
        "community_reports.parquet"
    ]
    
    missing = []
    for req_file in required_files:
        if not (output_dir / req_file).exists():
            missing.append(req_file)
    
    if missing:
        print(f"❌ 缺少必要文件: {missing}")
        return False
    
    print("✅ 所有必要文件都存在")
    return True

def load_and_explore_data():
    """加载并探索GraphRAG生成的数据"""
    output_dir = Path("output")
    
    print("\n🔍 数据探索:")
    
    # 加载实体数据
    try:
        entities_df = pd.read_parquet(output_dir / "entities.parquet")
        print(f"📊 实体数量: {len(entities_df)}")
        print("📝 实体示例:")
        print(entities_df[['title', 'type', 'description']].head())
        
        # 统计实体类型
        if 'type' in entities_df.columns:
            print("\n📈 实体类型分布:")
            print(entities_df['type'].value_counts())
            
    except Exception as e:
        print(f"❌ 加载实体数据失败: {e}")
    
    # 加载关系数据  
    try:
        relationships_df = pd.read_parquet(output_dir / "relationships.parquet")
        print(f"\n🔗 关系数量: {len(relationships_df)}")
        print("📝 关系示例:")
        print(relationships_df[['source', 'target', 'description']].head())
        
    except Exception as e:
        print(f"❌ 加载关系数据失败: {e}")
    
    # 加载社区数据
    try:
        communities_df = pd.read_parquet(output_dir / "communities.parquet")
        print(f"\n🏘️ 社区数量: {len(communities_df)}")
        
        # 社区大小分布
        if 'level' in communities_df.columns:
            print("📈 社区层级分布:")
            print(communities_df['level'].value_counts().sort_index())
            
    except Exception as e:
        print(f"❌ 加载社区数据失败: {e}")
    
    # 加载社区报告
    try:
        reports_df = pd.read_parquet(output_dir / "community_reports.parquet")
        print(f"\n📋 社区报告数量: {len(reports_df)}")
        
        if len(reports_df) > 0:
            print("📝 社区报告示例:")
            print("=" * 50)
            sample_report = reports_df.iloc[0]
            print(f"社区ID: {sample_report.get('community', 'N/A')}")
            print(f"标题: {sample_report.get('title', 'N/A')}")
            print(f"摘要: {sample_report.get('summary', 'N/A')[:200]}...")
            print("=" * 50)
            
    except Exception as e:
        print(f"❌ 加载社区报告失败: {e}")

def simple_global_search():
    """简单的全局搜索实现"""
    try:
        output_dir = Path("output")
        reports_df = pd.read_parquet(output_dir / "community_reports.parquet")
        
        print("\n🔍 基于社区报告的全局搜索:")
        print("=" * 60)
        
        query_keywords = ["bias", "llm", "evaluation", "perspective", "model"]
        
        relevant_reports = []
        for idx, report in reports_df.iterrows():
            summary = str(report.get('summary', ''))
            title = str(report.get('title', ''))
            full_content = f"{title} {summary}".lower()
            
            # 简单关键词匹配
            matches = sum(1 for keyword in query_keywords if keyword in full_content)
            if matches > 0:
                relevant_reports.append({
                    'community': report.get('community', 'N/A'),
                    'title': title,
                    'summary': summary,
                    'relevance_score': matches
                })
        
        # 按相关性排序
        relevant_reports.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        print(f"找到 {len(relevant_reports)} 个相关社区报告:")
        print()
        
        for i, report in enumerate(relevant_reports[:3]):  # 显示前3个最相关的
            print(f"🏘️ 社区 {report['community']} (相关性: {report['relevance_score']})")
            print(f"📋 标题: {report['title']}")
            print(f"📄 摘要: {report['summary'][:300]}...")
            print("-" * 40)
            
        # 生成简单的全局答案
        if relevant_reports:
            print("\n📊 关于LLM偏见的主要主题:")
            all_summaries = " ".join([r['summary'] for r in relevant_reports[:3]])
            
            # 简单的主题提取（基于关键词频率）
            themes = {
                'bias evaluation': ['evaluation', 'evaluate', 'measure', 'assessment'],
                'cultural perspectives': ['cultural', 'western', 'perspective', 'diversity'],
                'model transparency': ['transparent', 'open-source', 'inspection'],
                'bias mitigation': ['reduce', 'mitigate', 'prevent', 'address']
            }
            
            found_themes = []
            for theme, keywords in themes.items():
                score = sum(1 for kw in keywords if kw in all_summaries.lower())
                if score > 0:
                    found_themes.append((theme, score))
            
            found_themes.sort(key=lambda x: x[1], reverse=True)
            
            for theme, score in found_themes:
                print(f"  • {theme.title()} (提及度: {score})")
        
    except Exception as e:
        print(f"❌ 全局搜索失败: {e}")

def analyze_user_perspectives():
    """分析不同用户的观点差异"""
    try:
        output_dir = Path("output")
        documents_df = pd.read_parquet(output_dir / "documents.parquet")
        
        print("\n👥 用户观点分析:")
        print("=" * 40)
        
        # 分析文档中的用户信息
        user_docs = {}
        for idx, doc in documents_df.iterrows():
            doc_id = str(doc.get('id', ''))
            title = str(doc.get('title', ''))
            
            # 从ID中提取用户信息
            if 'bias-' in doc_id:
                user_id = doc_id.split('-')[1].split('_')[0]
                if user_id not in user_docs:
                    user_docs[user_id] = []
                user_docs[user_id].append({
                    'id': doc_id,
                    'title': title
                })
        
        print(f"发现 {len(user_docs)} 个不同用户的对话:")
        for user_id, docs in user_docs.items():
            print(f"  👤 用户 {user_id}: {len(docs)} 个对话")
        
        return user_docs
        
    except Exception as e:
        print(f"❌ 用户分析失败: {e}")
        return {}

def main():
    """主函数"""
    print("🚀 GraphRAG 数据分析与查询")
    print("=" * 50)
    
    # 1. 检查文件
    if not check_output_files():
        print("请确保GraphRAG索引已成功完成")
        return
    
    # 2. 探索数据
    load_and_explore_data()
    
    # 3. 用户分析
    user_docs = analyze_user_perspectives()
    
    # 4. 简单全局搜索
    simple_global_search()
    
    print("\n✅ 分析完成!")
    print("\n💡 接下来可以:")
    print("  1. 基于这些结果开发更复杂的查询逻辑")
    print("  2. 实现知识图谱可视化")
    print("  3. 构建用户协作创意系统")

if __name__ == "__main__":
    main()