from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
from datetime import datetime

from src.core.graphrag_manager import GraphRAGManager
from src.graftmind.ideation_processor import IdeationProcessor

app = FastAPI(title="GraftMind GraphRAG API")

# CORS配置（允许前端访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请设置具体的前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局实例
graphrag_manager = GraphRAGManager()
message_queue = asyncio.Queue()

# 请求/响应模型
class Message(BaseModel):
    user_id: str
    content: str
    space_type: str = "private"
    metadata: Optional[Dict] = None

class QueryRequest(BaseModel):
    question: str
    method: str = "local"  # local or global
    user_id: Optional[str] = None
    context_filter: Optional[Dict] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    confidence: float
    timestamp: datetime

class IndexStatus(BaseModel):
    status: str
    last_updated: datetime
    total_entities: int
    total_relationships: int

# API端点
@app.post("/api/messages")
async def add_message(message: Message, background_tasks: BackgroundTasks):
    """接收新的对话消息"""
    msg_data = {
        "message_id": f"{message.user_id}_{datetime.now().timestamp()}",
        "user_id": message.user_id,
        "timestamp": datetime.now().isoformat(),
        "space_type": message.space_type,
        "content": {
            "text": message.content,
            "type": "ideation"
        }
    }
    
    # 添加到队列
    await message_queue.put(msg_data)
    
    # 后台处理
    background_tasks.add_task(process_message_batch)
    
    return {"status": "accepted", "message_id": msg_data["message_id"]}

@app.post("/api/query", response_model=QueryResponse)
async def query_graph(request: QueryRequest):
    """查询知识图谱"""
    try:
        result = await graphrag_manager.query(
            question=request.question,
            method=request.method,
            filters=request.context_filter
        )
        
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            confidence=result.get("confidence", 0.8),
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status", response_model=IndexStatus)
async def get_status():
    """获取系统状态"""
    status = await graphrag_manager.get_status()
    return IndexStatus(**status)

@app.post("/api/index/rebuild")
async def rebuild_index(background_tasks: BackgroundTasks):
    """重建索引（管理员功能）"""
    background_tasks.add_task(graphrag_manager.rebuild_index)
    return {"status": "rebuilding started"}

@app.get("/api/insights/{user_id}")
async def get_user_insights(user_id: str):
    """获取用户创意洞察"""
    insights = await graphrag_manager.get_user_insights(user_id)
    return insights

@app.get("/api/recommendations/{user_id}")
async def get_recommendations(user_id: str, limit: int = 5):
    """获取个性化推荐"""
    recommendations = await graphrag_manager.get_recommendations(
        user_id=user_id,
        limit=limit
    )
    return recommendations

# WebSocket支持（实时通信）
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    try:
        while True:
            # 接收消息
            data = await websocket.receive_json()
            
            # 处理消息
            if data["type"] == "message":
                await add_message(Message(**data["payload"]), BackgroundTasks())
            
            elif data["type"] == "query":
                result = await query_graph(QueryRequest(**data["payload"]))
                await websocket.send_json({
                    "type": "response",
                    "data": result.dict()
                })
                
    except WebSocketDisconnect:
        print(f"User {user_id} disconnected")

# 后台任务
async def process_message_batch():
    """批量处理消息"""
    messages = []
    
    # 收集消息
    while not message_queue.empty() and len(messages) < 50:
        messages.append(await message_queue.get())
    
    if messages:
        await graphrag_manager.process_messages(messages)