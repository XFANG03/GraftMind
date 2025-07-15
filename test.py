import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def analyze_relationships_file(file_path):
    """
    分析GraphRAG的relationships.parquet文件
    
    Args:
        file_path: relationships.parquet文件路径
    """
    
    # 读取parquet文件
    try:
        df = pd.read_parquet(file_path)
        print("✅ 成功读取relationships.parquet文件")
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None
    
    # 基本信息
    print(f"\n📊 基本统计信息:")
    print(f"关系总数: {len(df)}")
    print(f"列数: {len(df.columns)}")
    print(f"文件大小: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # 显示列名和数据类型
    print(f"\n📋 列结构:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    # 显示前几行数据
    print(f"\n🔍 前5行数据:")
    print(df.head())
    
    # 检查是否有权重/相似度相关的列
    print(f"\n🔗 关系权重/相似度分析:")
    weight_columns = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in ['weight', 'score', 'similarity', 'confidence', 'strength'])]
    
    if weight_columns:
        print(f"找到权重/相似度相关列: {weight_columns}")
        for col in weight_columns:
            print(f"\n  {col} 统计:")
            print(f"    最小值: {df[col].min()}")
            print(f"    最大值: {df[col].max()}")
            print(f"    平均值: {df[col].mean():.4f}")
            print(f"    标准差: {df[col].std():.4f}")
    else:
        print("未找到明显的权重/相似度列")
    
    # 分析关系类型
    if 'relationship' in df.columns:
        print(f"\n📊 关系类型分布:")
        rel_counts = df['relationship'].value_counts()
        print(rel_counts.head(10))
        
        # 绘制关系类型分布图
        plt.figure(figsize=(12, 6))
        rel_counts.head(15).plot(kind='bar')
        plt.title('Top 15 关系类型分布')
        plt.xlabel('关系类型')
        plt.ylabel('频次')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # 分析实体
    if 'source' in df.columns and 'target' in df.columns:
        print(f"\n🎯 实体分析:")
        unique_entities = set(df['source'].unique()) | set(df['target'].unique())
        print(f"唯一实体数量: {len(unique_entities)}")
        
        # 计算实体度分布
        entity_degrees = {}
        for _, row in df.iterrows():
            source, target = row['source'], row['target']
            entity_degrees[source] = entity_degrees.get(source, 0) + 1
            entity_degrees[target] = entity_degrees.get(target, 0) + 1
        
        degrees = list(entity_degrees.values())
        print(f"平均度: {np.mean(degrees):.2f}")
        print(f"度的标准差: {np.std(degrees):.2f}")
        
        # 找出度最高的实体
        top_entities = sorted(entity_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n🏆 度最高的10个实体:")
        for entity, degree in top_entities:
            print(f"  {entity}: {degree}")
    
    # 分析描述字段
    if 'description' in df.columns:
        print(f"\n📝 描述字段分析:")
        desc_lengths = df['description'].str.len()
        print(f"描述平均长度: {desc_lengths.mean():.2f} 字符")
        print(f"描述长度标准差: {desc_lengths.std():.2f}")
        print(f"最长描述: {desc_lengths.max()} 字符")
        print(f"最短描述: {desc_lengths.min()} 字符")
    
    # 检查其他可能的数值列
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    print(f"\n🔢 数值型列分析:")
    for col in numeric_columns:
        if col not in weight_columns:  # 避免重复分析
            print(f"  {col}:")
            print(f"    范围: {df[col].min()} - {df[col].max()}")
            print(f"    平均值: {df[col].mean():.4f}")
    
    # 如果有权重列，绘制分布图
    if weight_columns:
        plt.figure(figsize=(12, 4))
        for i, col in enumerate(weight_columns[:3]):  # 最多显示3个权重列
            plt.subplot(1, len(weight_columns[:3]), i+1)
            plt.hist(df[col].dropna(), bins=30, alpha=0.7)
            plt.title(f'{col} 分布')
            plt.xlabel(col)
            plt.ylabel('频次')
        plt.tight_layout()
        plt.show()
    
    return df

def search_similar_relationships(df, entity_name, top_n=10):
    """
    搜索与特定实体相关的关系
    """
    if df is None:
        return
    
    # 搜索包含特定实体的关系
    related_rels = df[(df['source'].str.contains(entity_name, case=False, na=False)) | 
                     (df['target'].str.contains(entity_name, case=False, na=False))]
    
    print(f"\n🔍 与 '{entity_name}' 相关的关系 (前{top_n}个):")
    if len(related_rels) > 0:
        # 如果有权重列，按权重排序
        weight_columns = [col for col in related_rels.columns if any(keyword in col.lower() 
                         for keyword in ['weight', 'score', 'similarity', 'confidence'])]
        
        if weight_columns:
            related_rels = related_rels.sort_values(weight_columns[0], ascending=False)
        
        for idx, row in related_rels.head(top_n).iterrows():
            print(f"  {row['source']} -> {row['target']}")
            if 'relationship' in row:
                print(f"    关系: {row['relationship']}")
            if weight_columns:
                print(f"    权重: {row[weight_columns[0]]}")
            if 'description' in row:
                print(f"    描述: {row['description'][:100]}...")
            print()
    else:
        print(f"未找到与 '{entity_name}' 相关的关系")

# 使用示例
if __name__ == "__main__":
    # 请将此路径替换为您的relationships.parquet文件路径
    file_path = "output/relationships.parquet"
    
    print("🚀 开始分析GraphRAG关系文件...")
    df = analyze_relationships_file(file_path)
    
    if df is not None:
        # 可以进一步搜索特定实体的关系
        # search_similar_relationships(df, "输入您想搜索的实体名称")
        
        print("\n✨ 分析完成! 您可以使用以下代码进行进一步探索:")
        print("search_similar_relationships(df, '实体名称')")
        print("df.head()  # 查看数据")
        print("df.columns  # 查看所有列名")
        print("df.describe()  # 查看数值列的描述性统计")