import os
import time
import json
import logging
from typing import Dict, List, Optional, Union, Any, Literal
from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from solution import tool_map
from prompt import instruction_template

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# OpenAI compatible API
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    max_tokens: int = 512
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Dict[str, int]


# Initialize FastAPI app and inference engine
app = FastAPI(
    title="OpenAI Compatible API",
    description="An API compatible with OpenAI's API, powered by your custom LLM",
    version="1.0.0",
)


AVAILABLE_MODELS = ["S1-Base", ]


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion"""
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

    model = ChatOpenAI(
        model=os.getenv("ROUTE_MODEL_NAME"),
        base_url=os.getenv("ROUTE_MODEL_BASE_URL"),
        api_key=os.getenv("ROUTE_MODEL_BASE_API_KEY"),
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )

    route_input = []

    for rm in request.messages:
        if rm.role == 'user':
            route_input.append({"role": rm.role, "content": instruction_template.format(user_query=rm.content)})
        else:
            route_input.append({"role": rm.role, "content": rm.content})

    route_response = model.invoke(route_input)
    content = route_response.content.replace('<think>\n\n</think>\n\n', '')

    route = json.loads(content)

    from langgraph.prebuilt import create_react_agent

    agent_model = ChatOpenAI(
        model=os.getenv("BASE_MODEL_NAME"),
        base_url=os.getenv("BASE_MODEL_BASE_URL"),
        api_key=os.getenv("BASE_MODEL_BASE_API_KEY"),
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )


    tools = [tool_map.get(route.get('intent'))] if tool_map.get(route.get('intent')) else []
    agent = create_react_agent(model=agent_model, tools=tools)
    agent_input = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    response = agent.invoke({'messages': agent_input})
    result_content = []
    token_usage = {
        'completion_tokens': 0,
        'prompt_tokens': 0,
        'total_tokens': 0,
    }
    for message in response.get('messages', []):
        if not isinstance(message, AIMessage):
            continue
        result_content.append(message.content)
        token_usage['total_tokens'] += message.response_metadata.get('token_usage', {}).get('total_tokens', 0)
        token_usage['completion_tokens'] += message.response_metadata.get('token_usage', {}).get('completion_tokens', 0)
        token_usage['prompt_tokens'] += message.response_metadata.get('token_usage', {}).get('prompt_tokens', 0)

    choice = ChatCompletionResponseChoice(
        index=0,
        message=Message(role="assistant", content=''.join(result_content)),
        finish_reason="stop"
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time() * 1000)}",
        created=int(time.time()),
        model=request.model,
        choices=[choice],
        usage=token_usage,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
