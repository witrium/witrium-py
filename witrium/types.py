from typing import Any, Optional, List
from pydantic import BaseModel


class FileUpload(BaseModel):
    """
    File upload schema.
    Args:
        filename: The name of the file.
        data: The base64 encoded file content.
    """

    filename: str
    data: str  # base64 encoded file content


class WorkflowRunSubmittedSchema(BaseModel):
    workflow_id: str
    run_id: str
    status: str


class AgentExecutionSchema(BaseModel):
    status: str
    instruction_order: int
    instruction: str
    result: Optional[dict | list] = None
    result_format: Optional[str] = None
    error_message: Optional[str] = None


class WorkflowRunExecutionSchema(BaseModel):
    instruction_id: str
    instruction: str
    result: Optional[dict | list] = None
    result_format: Optional[str] = None
    message: Optional[str] = None
    status: str
    error_message: Optional[str] = None


class WorkflowRunResultsSchema(BaseModel):
    workflow_id: str
    run_id: str
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    message: Optional[str] = None
    executions: Optional[List[AgentExecutionSchema]] = None
    result: Optional[dict | list] = None
    result_format: Optional[str] = None
    error_message: Optional[str] = None


class WorkflowSchema(BaseModel):
    uuid: str
    name: str
    description: Optional[str] = None


class WorkflowRunSchema(BaseModel):
    uuid: str
    session_id: Optional[str] = None  # browser_session id
    workflow: WorkflowSchema
    run_type: str
    triggered_by: str
    status: str
    session_active: bool
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    executions: Optional[List[WorkflowRunExecutionSchema]] = None


class WorkflowRunStatus:
    """Constants for workflow run statuses."""

    PENDING = "P"
    RUNNING = "R"
    COMPLETED = "C"
    FAILED = "F"
    CANCELLED = "X"

    # Terminal statuses that should stop polling
    TERMINAL_STATUSES = [COMPLETED, FAILED, CANCELLED]

    # Reverse mapping for human-readable status names
    STATUS_NAMES = {
        PENDING: "pending",
        RUNNING: "running",
        COMPLETED: "completed",
        FAILED: "failed",
        CANCELLED: "cancelled",
    }

    @classmethod
    def get_status_name(cls, status_code: str) -> str:
        """Get human-readable status name from status code."""
        return cls.STATUS_NAMES.get(status_code, status_code)


class AgentExecutionStatus:
    """Constants for agent execution statuses."""

    PENDING = "P"
    RUNNING = "R"
    COMPLETED = "C"
    FAILED = "F"
    CANCELLED = "X"

    STATUS_NAMES = {
        PENDING: "pending",
        RUNNING: "running",
        COMPLETED: "completed",
        FAILED: "failed",
        CANCELLED: "cancelled",
    }

    @classmethod
    def get_status_name(cls, status_code: str) -> str:
        """Get human-readable status name from status code."""
        return cls.STATUS_NAMES.get(status_code, status_code)


class TalentResultSchema(BaseModel):
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    message: Optional[str] = None
    result: Optional[Any] = None
    result_format: Optional[str] = None
    error_message: Optional[str] = None
