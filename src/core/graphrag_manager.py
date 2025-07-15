import graphrag.api as api
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path

class GraphRAGManager:
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent.parent
        self.config = self._load_config()
        self._init_cache()
    
    async def process_messages(self, messages: List[Dict]):
        """处理消息批次"""
        # 准备数据
        df = self._prepare_dataframe(messages)
        
        # 保存到input目录
        filepath = self._save_to_input(df)
        
        # 触发增量更新
        await self._update_index()
    
    async def query(self, question: str, method: str = "local", filters: Dict = None):
        """执行查询"""
        # 构建查询参数
        query_params = {
            "question": question,
            "method": method
        }
        
        if filters:
            # 应用过滤器（如用户ID、时间范围等）
            query_params["context"] = self._build_context(filters)
        
        # 执行查询
        result = await api.query(
            config=self.config,
            **query_params
        )
        
        # 格式化响应
        return self._format_response(result)
    
    async def get_user_insights(self, user_id: str):
        """获取用户洞察"""
        # 查询用户相关的实体和关系
        query = f"What are the main ideas and themes from user {user_id}?"
        result = await self.query(query, method="local", filters={"user_id": user_id})
        
        return {
            "user_id": user_id,
            "main_themes": result["themes"],
            "key_ideas": result["ideas"],
            "collaboration_score": self._calculate_collaboration_score(user_id)
        }
    
    async def get_recommendations(self, user_id: str, limit: int = 5):
        """获取推荐"""
        # 基于图谱的推荐逻辑
        query = f"What ideas from other users might be relevant to {user_id}'s work?"
        result = await self.query(query, method="global")
        
        recommendations = []
        for item in result["related_ideas"][:limit]:
            recommendations.append({
                "idea": item["content"],
                "relevance_score": item["score"],
                "source_user": item["user_id"],
                "reason": item["connection"]
            })
        
        return recommendations