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


class WorkflowRunOptionsSchema(BaseModel):
    """
    Options for running a workflow.

    Note: If browser_session_id is provided, the use_states parameter will be ignored.
    The browser session's use_states (set during session creation) will be used instead.
    """

    args: Optional[dict[str, str | int | float]] = None
    files: Optional[List[FileUpload]] = None
    use_states: Optional[List[str]] = None
    preserve_state: Optional[str] = None
    no_intelligence: bool = False
    record_session: bool = False
    browser_session_id: Optional[str] = None
    skip_goto_url_instruction: bool = False


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


class WorkflowRunResultSchema(BaseModel):
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


class TalentRunOptionsSchema(BaseModel):
    """
    Options for running a talent.

    Note: If browser_session_id is provided, the use_states parameter will be ignored.
    The browser session's use_states (set during session creation) will be used instead.
    """

    args: Optional[dict[str, Any]] = None
    files: Optional[List[FileUpload]] = None
    use_states: Optional[List[str]] = None  # Ignored if browser_session_id is provided
    preserve_state: Optional[str] = None
    browser_session_id: Optional[str] = (
        None  # If provided, uses existing browser session
    )


class TalentResultSchema(BaseModel):
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    message: Optional[str] = None
    result: Optional[Any] = None
    result_format: Optional[str] = None
    error_message: Optional[str] = None


class WaitUntilStateOptionsSchema(BaseModel):
    """
    Options for waiting until a workflow reaches a specific state.

    Args:
        all_instructions_executed: If True, wait until all instructions are executed.
        min_wait_time: Minimum time to wait before starting to poll (seconds).
        polling_interval: Time between polls (seconds).
        timeout: Maximum time to wait (seconds). None means no timeout (poll forever).
    """

    all_instructions_executed: bool = False
    min_wait_time: int = 0
    polling_interval: int = 2
    timeout: Optional[float] = None


class RunWorkflowAndWaitOptionsSchema(WorkflowRunOptionsSchema):
    """
    Options for running a workflow and waiting for results.

    Inherits all WorkflowRunOptionsSchema fields plus:
        polling_interval: Time between polls (seconds).
        timeout: Maximum time to wait (seconds). None means no timeout (poll forever).
        return_intermediate_results: If True, return all intermediate poll results.
        on_progress: Optional callback function called on each poll with results.
    """

    polling_interval: int = 5
    timeout: Optional[float] = None
    return_intermediate_results: bool = False
    on_progress: Optional[Any] = (
        None  # Callable, but we'll handle type in the docstring
    )


class BrowserSessionCreateOptions(BaseModel):
    """
    Options for creating a browser session.

    Important: The use_states set here will apply to all workflow and talent runs
    that use this browser session. Individual run options' use_states will be ignored.
    """

    provider: str = "omega"
    use_proxy: bool = False
    proxy_country: str = "us"
    proxy_city: str = "New York"
    use_states: Optional[List[str]] = None  # Applies to all runs using this session
    preserve_state: Optional[str] = None


class BrowserSessionSchema(BaseModel):
    uuid: str
    provider: str
    status: str  # "active" or "closed"
    is_busy: bool
    user_managed: bool
    current_run_type: Optional[str] = None  # "workflow", "talent", or None
    current_run_id: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    last_activity_at: Optional[str] = None
    proxy_country: Optional[str] = None
    proxy_city: Optional[str] = None


class ListBrowserSessionSchema(BaseModel):
    sessions: List[BrowserSessionSchema]
    total_count: int


class BrowserSessionCloseOptions(BaseModel):
    force: bool = False
    preserve_state: Optional[str] = None
