"""
A2A protocol schema (subset targeting the April 2025+ spec).

We implement the minimum needed for:
- Spec-compliant Agent Cards at /.well-known/agent.json
- SendMessage JSON-RPC method with text-only parts
- Task lifecycle with TASK_STATE_* enum

Omitted for MVP (all optional per spec):
- Streaming (SSE / bidi)
- Push notifications
- File / data parts (text only for now)
- gRPC and HTTP+JSON bindings (JSONRPC only)
- Signatures
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class _A2AModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="allow",
    )


class TaskState(str, Enum):
    SUBMITTED = "TASK_STATE_SUBMITTED"
    WORKING = "TASK_STATE_WORKING"
    INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
    COMPLETED = "TASK_STATE_COMPLETED"
    CANCELED = "TASK_STATE_CANCELED"
    FAILED = "TASK_STATE_FAILED"


class TextPart(_A2AModel):
    text: str
    kind: Literal["text"] = "text"


Part = TextPart  # MVP: text only; spec allows file/data parts as a union


class Message(_A2AModel):
    role: str
    parts: List[Part]
    message_id: str = Field(default_factory=lambda: f"msg-{uuid4()}")
    context_id: Optional[str] = None
    reference_task_ids: Optional[List[str]] = None


class Artifact(_A2AModel):
    artifact_id: str = Field(default_factory=lambda: f"artifact-{uuid4()}")
    name: Optional[str] = None
    description: Optional[str] = None
    parts: List[Part]


class TaskStatus(_A2AModel):
    state: TaskState
    message: Optional[str] = None


class Task(_A2AModel):
    id: str = Field(default_factory=lambda: f"task-{uuid4()}")
    context_id: str = Field(default_factory=lambda: f"ctx-{uuid4()}")
    status: TaskStatus
    artifacts: List[Artifact] = Field(default_factory=list)
    history: List[Message] = Field(default_factory=list)


class AgentInterface(_A2AModel):
    url: str
    protocol_binding: Literal["JSONRPC", "GRPC", "HTTP+JSON"] = "JSONRPC"
    protocol_version: str = "1.0"


class AgentCapabilities(_A2AModel):
    streaming: bool = False
    push_notifications: bool = False
    state_transition_history: bool = False
    extended_agent_card: bool = False


class AgentSkill(_A2AModel):
    id: str
    name: str
    description: str
    tags: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)
    input_modes: List[str] = Field(default_factory=lambda: ["text/plain"])
    output_modes: List[str] = Field(default_factory=lambda: ["text/plain"])


class AgentProvider(_A2AModel):
    organization: str
    url: Optional[str] = None


class AgentCard(_A2AModel):
    name: str
    description: str
    version: str = "0.1.0"
    provider: Optional[AgentProvider] = None
    icon_url: Optional[str] = None
    documentation_url: Optional[str] = None
    supported_interfaces: List[AgentInterface]
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    security_schemes: Dict[str, Any] = Field(default_factory=dict)
    security_requirements: List[Dict[str, List[str]]] = Field(default_factory=list)
    default_input_modes: List[str] = Field(default_factory=lambda: ["text/plain"])
    default_output_modes: List[str] = Field(default_factory=lambda: ["text/plain"])
    skills: List[AgentSkill] = Field(default_factory=list)

    def all_tags(self) -> List[str]:
        tags: List[str] = []
        for skill in self.skills:
            tags.extend(skill.tags)
        return sorted(set(tags))


class JsonRpcRequest(_A2AModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Union[str, int, None] = None
    method: str
    params: Dict[str, Any] = Field(default_factory=dict)


class JsonRpcError(_A2AModel):
    code: int
    message: str
    data: Optional[Any] = None


class JsonRpcResponse(_A2AModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Union[str, int, None] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[JsonRpcError] = None


def text_message(role: str, text: str) -> Message:
    return Message(role=role, parts=[TextPart(text=text)])


def text_artifact(name: str, text: str, description: Optional[str] = None) -> Artifact:
    return Artifact(name=name, description=description, parts=[TextPart(text=text)])
