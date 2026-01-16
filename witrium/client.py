import time
import asyncio
import logging
import httpx
from typing import Dict, List, Optional, Union
from witrium.types import (
    BrowserSessionCloseOptions,
    WorkflowRunSubmittedSchema,
    WorkflowRunResultSchema,
    WorkflowRunSchema,
    WorkflowRunStatus,
    AgentExecutionStatus,
    TalentResultSchema,
    WorkflowRunOptionsSchema,
    TalentRunOptionsSchema,
    WaitUntilStateOptionsSchema,
    RunWorkflowAndWaitOptionsSchema,
    BrowserSessionCreateOptions,
    BrowserSessionSchema,
    ListBrowserSessionSchema,
)

# Setup logger
logger = logging.getLogger("witrium_client")


DEFAULT_BASE_URL = "https://api.witrium.com"


class WitriumClientException(Exception):
    """Base exception for Witrium Client errors."""

    pass


class WitriumClient:
    """
    Base class for Witrium API Client.
    Not meant to be used directly - use SyncWitriumClient or AsyncWitriumClient.
    """

    def __init__(
        self,
        api_token: str,
        timeout: Optional[float] = None,
        verify_ssl: bool = True,
    ):
        """
        Initialize the Witrium client.
        Args:
            api_token: The API token for authentication.
            timeout: Timeout in seconds for HTTP requests. None means no timeout.
            verify_ssl: Whether to verify SSL certificates.
        """
        self.base_url = DEFAULT_BASE_URL.rstrip("/")
        self.api_token = api_token
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._headers = {"X-Witrium-Key": api_token, "Content-Type": "application/json"}


class SyncWitriumClient(WitriumClient):
    """Synchronous Witrium API client."""

    def __init__(
        self,
        api_token: str,
        timeout: Optional[float] = None,
        verify_ssl: bool = True,
        session_options: Optional[BrowserSessionCreateOptions] = None,
    ):
        """Initialize the synchronous client.

        Args:
            api_token: The API token for authentication.
            timeout: Timeout in seconds for HTTP requests. None means no timeout (infinite).
            verify_ssl: Whether to verify SSL certificates.
            session_options: Options for automatic browser session creation.
        """
        super().__init__(api_token, timeout, verify_ssl)
        self._client = httpx.Client(
            timeout=self.timeout, verify=self.verify_ssl, headers=self._headers
        )
        self._session_options = session_options or BrowserSessionCreateOptions()
        self.session_id: Optional[str] = None

    def close(self):
        """Close the underlying HTTP client."""
        if self._client:
            self._client.close()

    def __enter__(self):
        session = self.create_browser_session(self._session_options)
        self.session_id = session.uuid
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.session_id:
            try:
                self.close_browser_session(
                    session_id=self.session_id,
                    options=BrowserSessionCloseOptions(
                        force=True, preserve_state=self._session_options.preserve_state
                    ),
                )
            except Exception:
                pass  # Best effort cleanup
            self.session_id = None
        self.close()

    def run_workflow(
        self,
        workflow_id: str,
        options: Optional[WorkflowRunOptionsSchema] = None,
    ) -> WorkflowRunSubmittedSchema:
        """
        Run a workflow by ID.

        Args:
            workflow_id: The ID of the workflow to run.
            options: Optional workflow run options.
                If browser_session_id is provided (or client.session_id is set),
                the use_states from the browser session will be used instead of
                options.use_states.

        Returns:
            Dict containing workflow_id, run_id and status.
        """
        if options is None:
            options = WorkflowRunOptionsSchema()

        url = f"{self.base_url}/v1/workflows/{workflow_id}/run"

        # Build payload with only defined values
        payload = {}
        if options.args is not None:
            payload["args"] = options.args
        if options.files is not None:
            payload["files"] = [file.model_dump() for file in options.files]
        if options.use_states is not None:
            payload["use_states"] = options.use_states
        if options.preserve_state is not None:
            payload["preserve_state"] = options.preserve_state
        if options.no_intelligence:
            payload["no_intelligence"] = options.no_intelligence
        if options.record_session:
            payload["record_session"] = options.record_session
        # Use client's session_id if available and no explicit session provided
        browser_session_id = options.browser_session_id or self.session_id
        if browser_session_id is not None:
            payload["browser_session_id"] = browser_session_id

        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            return WorkflowRunSubmittedSchema.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            error_detail = self._extract_error_detail(e.response)
            raise WitriumClientException(
                f"Error running workflow: {error_detail} (Status code: {e.response.status_code})"
            )
        except Exception as e:
            raise WitriumClientException(f"Error running workflow: {str(e)}")

    def get_workflow_results(self, run_id: str) -> WorkflowRunResultSchema:
        """
        Get workflow run results.

        Args:
            run_id: The ID of the workflow run.

        Returns:
            Dict containing the workflow run results.
        """
        url = f"{self.base_url}/v1/runs/{run_id}/results"

        try:
            response = self._client.get(url)
            response.raise_for_status()
            return WorkflowRunResultSchema.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            error_detail = self._extract_error_detail(e.response)
            raise WitriumClientException(
                f"Error getting workflow results: {error_detail} (Status code: {e.response.status_code})"
            )
        except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            # Transient network errors - provide a descriptive message
            error_type = type(e).__name__
            raise WitriumClientException(
                f"Network error getting workflow results: {error_type} - connection was interrupted. "
                "This is usually a transient error, please retry."
            )
        except Exception as e:
            error_msg = str(e) if str(e) else type(e).__name__
            raise WitriumClientException(f"Error getting workflow results: {error_msg}")

    def run_workflow_and_wait(
        self,
        workflow_id: str,
        options: Optional[RunWorkflowAndWaitOptionsSchema] = None,
    ) -> Union[WorkflowRunResultSchema, List[WorkflowRunResultSchema]]:
        """
        Run a workflow and wait for results by polling until completion.

        Args:
            workflow_id: The ID of the workflow to run.
            options: Optional workflow run and wait options.

        Returns:
            Dict containing the final workflow run results, or if return_intermediate_results=True,
            a list of all polled result dictionaries.
        """
        if options is None:
            options = RunWorkflowAndWaitOptionsSchema()

        # Run the workflow
        run_response = self.run_workflow(
            workflow_id=workflow_id,
            options=WorkflowRunOptionsSchema(
                args=options.args,
                files=options.files,
                use_states=options.use_states,
                preserve_state=options.preserve_state,
                no_intelligence=options.no_intelligence,
                record_session=options.record_session,
                browser_session_id=options.browser_session_id,
            ),
        )

        run_id = run_response.run_id
        start_time = time.time()
        intermediate_results = []
        consecutive_errors = 0
        max_consecutive_errors = 3

        # Poll for results
        while True:
            # Check timeout if specified
            if (
                options.timeout is not None
                and time.time() - start_time >= options.timeout
            ):
                raise WitriumClientException(
                    f"Workflow execution timed out after {options.timeout} seconds"
                )

            try:
                results = self.get_workflow_results(run_id)
                consecutive_errors = 0  # Reset on success
            except WitriumClientException as e:
                # Check if this is a transient network error (retry-able)
                if "Network error" in str(e) or "connection was interrupted" in str(e):
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        raise WitriumClientException(
                            f"Failed to get workflow results after {max_consecutive_errors} "
                            f"consecutive network errors: {str(e)}"
                        )
                    # Exponential backoff: wait longer after each error
                    backoff_time = options.polling_interval * (
                        2 ** (consecutive_errors - 1)
                    )
                    logger.warning(
                        f"Transient error polling workflow results (attempt {consecutive_errors}/{max_consecutive_errors}), "
                        f"retrying in {backoff_time}s: {str(e)}"
                    )
                    time.sleep(backoff_time)
                    continue
                else:
                    # Non-transient error, re-raise immediately
                    raise

            # Store intermediate results if requested
            if options.return_intermediate_results:
                intermediate_results.append(results)

            # Call progress callback if provided
            if options.on_progress:
                options.on_progress(results)

            # Check if workflow has completed
            if results.status in WorkflowRunStatus.TERMINAL_STATUSES:
                return (
                    intermediate_results
                    if options.return_intermediate_results
                    else results
                )

            # Wait before polling again
            time.sleep(options.polling_interval)

    def wait_until_state(
        self,
        run_id: str,
        target_status: str,
        options: Optional[WaitUntilStateOptionsSchema] = None,
    ) -> WorkflowRunResultSchema:
        """
        Wait for a workflow run to reach a specific status by polling.

        Args:
            run_id: The ID of the workflow run to wait for.
            target_status: The status to wait for (e.g., WorkflowRunStatus.RUNNING).
            options: Optional wait options.

        Returns:
            WorkflowRunResultSchema when the target status is reached.

        Raises:
            WitriumClientException: If timeout is reached or workflow reaches an unexpected terminal status.
        """
        if options is None:
            options = WaitUntilStateOptionsSchema()

        # Wait for minimum time before starting to poll
        if options.min_wait_time > 0:
            time.sleep(options.min_wait_time)

        def _check_all_executions_completed(results: WorkflowRunResultSchema) -> bool:
            """Check if all executions have completed status."""
            if not results.executions:
                return False
            return results.executions[-1].status == AgentExecutionStatus.COMPLETED

        start_time = time.time()
        consecutive_errors = 0
        max_consecutive_errors = 3

        # Poll until conditions are met
        while True:
            # Check timeout if specified
            if (
                options.timeout is not None
                and time.time() - start_time >= options.timeout
            ):
                target_status_name = WorkflowRunStatus.get_status_name(target_status)
                condition_msg = f"status '{target_status_name}'"
                if options.all_instructions_executed:
                    condition_msg += " and all instructions executed"
                raise WitriumClientException(
                    f"Workflow run did not reach {condition_msg} within {options.timeout} seconds"
                )

            try:
                results = self.get_workflow_results(run_id)
                consecutive_errors = 0  # Reset on success
            except WitriumClientException as e:
                # Check if this is a transient network error (retry-able)
                if "Network error" in str(e) or "connection was interrupted" in str(e):
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        raise WitriumClientException(
                            f"Failed to get workflow results after {max_consecutive_errors} "
                            f"consecutive network errors: {str(e)}"
                        )
                    # Exponential backoff: wait longer after each error
                    backoff_time = options.polling_interval * (
                        2 ** (consecutive_errors - 1)
                    )
                    logger.warning(
                        f"Transient error polling workflow results (attempt {consecutive_errors}/{max_consecutive_errors}), "
                        f"retrying in {backoff_time}s: {str(e)}"
                    )
                    time.sleep(backoff_time)
                    continue
                else:
                    # Non-transient error, re-raise immediately
                    raise

            # Check if workflow has reached the target status
            status_reached = results.status == target_status
            all_executions_completed = (
                _check_all_executions_completed(results)
                if options.all_instructions_executed
                else True
            )

            if status_reached and all_executions_completed:
                return results

            # Check if workflow has reached a terminal status that's not our target
            if (
                results.status in WorkflowRunStatus.TERMINAL_STATUSES
                and results.status != target_status
            ):
                current_status_name = WorkflowRunStatus.get_status_name(results.status)
                target_status_name = WorkflowRunStatus.get_status_name(target_status)
                raise WitriumClientException(
                    f"Workflow run reached terminal status '{current_status_name}' before reaching target status '{target_status_name}'"
                )

            # Wait before polling again
            time.sleep(options.polling_interval)

    def cancel_run(self, run_id: str) -> WorkflowRunSchema:
        """
        Cancel a workflow run and clean up associated resources.

        Args:
            run_id: The ID of the workflow run to cancel.

        Returns:
            Dict containing the workflow run results.
        """
        url = f"{self.base_url}/v1/runs/{run_id}/cancel"

        try:
            response = self._client.post(url)
            response.raise_for_status()
            return WorkflowRunSchema.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            error_detail = self._extract_error_detail(e.response)
            raise WitriumClientException(
                f"Error cancelling workflow run: {error_detail} (Status code: {e.response.status_code})"
            )
        except Exception as e:
            raise WitriumClientException(f"Error cancelling workflow run: {str(e)}")

    def _extract_error_detail(self, response: httpx.Response) -> str:
        """Extract error detail from response."""
        try:
            error_json = response.json()
            if "detail" in error_json:
                return error_json["detail"]
            return str(error_json)
        except Exception:
            return response.text or "Unknown error"

    def run_talent(
        self,
        talent_id: str,
        options: Optional[TalentRunOptionsSchema] = None,
    ) -> TalentResultSchema:
        """
        Run a talent by ID.

        Args:
            talent_id: The ID of the talent to run.
            options: Optional talent run options.
                If browser_session_id is provided (or client.session_id is set),
                the use_states from the browser session will be used instead of
                options.use_states.

        Returns:
            The result of the talent run.
        """
        if options is None:
            options = TalentRunOptionsSchema()

        url = f"{self.base_url}/v1/talents/{talent_id}/run"

        # Build payload with only defined values
        payload = {}
        if options.args is not None:
            payload["args"] = options.args
        if options.files is not None:
            payload["files"] = [file.model_dump() for file in options.files]
        if options.use_states is not None:
            payload["use_states"] = options.use_states
        if options.preserve_state is not None:
            payload["preserve_state"] = options.preserve_state
        # Use client's session_id if available and no explicit session provided
        browser_session_id = options.browser_session_id or self.session_id
        if browser_session_id is not None:
            payload["browser_session_id"] = browser_session_id

        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            return TalentResultSchema.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            error_detail = self._extract_error_detail(e.response)
            raise WitriumClientException(
                f"Error running talent: {error_detail} (Status code: {e.response.status_code})"
            )
        except Exception as e:
            raise WitriumClientException(f"Error running talent: {str(e)}")

    def create_browser_session(
        self, options: Optional[BrowserSessionCreateOptions] = None
    ) -> BrowserSessionSchema:
        """
        Create a standalone browser session.

        Args:
            options: Optional browser session creation options.
                The use_states set here will apply to all workflow and talent runs
                that use this browser session. Individual run options' use_states
                will be ignored when using this session.

        Returns:
            BrowserSessionSchema containing session details.
        """
        if options is None:
            options = BrowserSessionCreateOptions()

        url = f"{self.base_url}/v1/browser-sessions"
        payload = options.model_dump()

        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            return BrowserSessionSchema.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            error_detail = self._extract_error_detail(e.response)
            raise WitriumClientException(
                f"Error creating browser session: {error_detail} (Status code: {e.response.status_code})"
            )
        except Exception as e:
            raise WitriumClientException(f"Error creating browser session: {str(e)}")

    def list_browser_sessions(self) -> ListBrowserSessionSchema:
        """
        List all active browser sessions.

        Returns:
            BrowserSessionListResponse containing list of sessions and total count.
        """
        url = f"{self.base_url}/v1/browser-sessions"

        try:
            response = self._client.get(url)
            response.raise_for_status()
            return ListBrowserSessionSchema.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            error_detail = self._extract_error_detail(e.response)
            raise WitriumClientException(
                f"Error listing browser sessions: {error_detail} (Status code: {e.response.status_code})"
            )
        except Exception as e:
            raise WitriumClientException(f"Error listing browser sessions: {str(e)}")

    def get_browser_session(self, session_id: str) -> BrowserSessionSchema:
        """
        Get details of a specific browser session.

        Args:
            session_id: The UUID of the browser session.

        Returns:
            BrowserSessionSchema containing session details.
        """
        url = f"{self.base_url}/v1/browser-sessions/{session_id}"

        try:
            response = self._client.get(url)
            response.raise_for_status()
            return BrowserSessionSchema.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            error_detail = self._extract_error_detail(e.response)
            raise WitriumClientException(
                f"Error getting browser session: {error_detail} (Status code: {e.response.status_code})"
            )
        except Exception as e:
            raise WitriumClientException(f"Error getting browser session: {str(e)}")

    def close_browser_session(
        self, session_id: str, options: Optional[BrowserSessionCloseOptions] = None
    ) -> Dict[str, bool]:
        """
        Close a browser session.

        Args:
            session_id: The UUID of the browser session to close.
            options: Optional browser session close options.

        Returns:
            Dict containing success status.
        """

        if options is None:
            options = BrowserSessionCloseOptions()

        url = f"{self.base_url}/v1/browser-sessions/{session_id}"
        payload = options.model_dump()

        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = self._extract_error_detail(e.response)
            raise WitriumClientException(
                f"Error closing browser session: {error_detail} (Status code: {e.response.status_code})"
            )
        except Exception as e:
            raise WitriumClientException(f"Error closing browser session: {str(e)}")


class AsyncWitriumClient(WitriumClient):
    """Asynchronous Witrium API client."""

    def __init__(
        self,
        api_token: str,
        timeout: Optional[float] = None,
        verify_ssl: bool = True,
        session_options: Optional[BrowserSessionCreateOptions] = None,
    ):
        """Initialize the asynchronous client.

        Args:
            api_token: The API token for authentication.
            timeout: Timeout in seconds for HTTP requests. None means no timeout (infinite).
            verify_ssl: Whether to verify SSL certificates.
            session_options: Options for automatic browser session creation.
        """
        super().__init__(api_token, timeout, verify_ssl)
        self._client = httpx.AsyncClient(
            timeout=self.timeout, verify=self.verify_ssl, headers=self._headers
        )
        self._session_options = session_options or BrowserSessionCreateOptions()
        self.session_id: Optional[str] = None

    async def close(self):
        """Close the underlying HTTP client."""
        if self._client:
            await self._client.aclose()

    async def __aenter__(self):
        session = await self.create_browser_session(self._session_options)
        self.session_id = session.uuid
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self.session_id:
            try:
                await self.close_browser_session(
                    session_id=self.session_id,
                    options=BrowserSessionCloseOptions(
                        force=True, preserve_state=self._session_options.preserve_state
                    ),
                )
            except Exception:
                pass  # Best effort cleanup
            self.session_id = None
        await self.close()

    async def run_workflow(
        self,
        workflow_id: str,
        options: Optional[WorkflowRunOptionsSchema] = None,
    ) -> WorkflowRunSubmittedSchema:
        """
        Run a workflow by ID.

        Args:
            workflow_id: The ID of the workflow to run.
            options: Optional workflow run options.
                If browser_session_id is provided (or client.session_id is set),
                the use_states from the browser session will be used instead of
                options.use_states.

        Returns:
            Dict containing workflow_id, run_id and status.
        """
        if options is None:
            options = WorkflowRunOptionsSchema()

        url = f"{self.base_url}/v1/workflows/{workflow_id}/run"

        # Build payload with only defined values
        payload = {}
        if options.args is not None:
            payload["args"] = options.args
        if options.files is not None:
            payload["files"] = [file.model_dump() for file in options.files]
        if options.use_states is not None:
            payload["use_states"] = options.use_states
        if options.preserve_state is not None:
            payload["preserve_state"] = options.preserve_state
        if options.no_intelligence:
            payload["no_intelligence"] = options.no_intelligence
        if options.record_session:
            payload["record_session"] = options.record_session
        # Use client's session_id if available and no explicit session provided
        browser_session_id = options.browser_session_id or self.session_id
        if browser_session_id is not None:
            payload["browser_session_id"] = browser_session_id

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            return WorkflowRunSubmittedSchema.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            error_detail = await self._extract_error_detail(e.response)
            raise WitriumClientException(
                f"Error running workflow: {error_detail} (Status code: {e.response.status_code})"
            )
        except Exception as e:
            raise WitriumClientException(f"Error running workflow: {str(e)}")

    async def get_workflow_results(self, run_id: str) -> WorkflowRunResultSchema:
        """
        Get workflow run results.

        Args:
            run_id: The ID of the workflow run.

        Returns:
            Dict containing the workflow run results.
        """
        url = f"{self.base_url}/v1/runs/{run_id}/results"

        try:
            response = await self._client.get(url)
            response.raise_for_status()
            return WorkflowRunResultSchema.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            error_detail = await self._extract_error_detail(e.response)
            raise WitriumClientException(
                f"Error getting workflow results: {error_detail} (Status code: {e.response.status_code})"
            )
        except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            # Transient network errors - provide a descriptive message
            error_type = type(e).__name__
            raise WitriumClientException(
                f"Network error getting workflow results: {error_type} - connection was interrupted. "
                "This is usually a transient error, please retry."
            )
        except Exception as e:
            error_msg = str(e) if str(e) else type(e).__name__
            raise WitriumClientException(f"Error getting workflow results: {error_msg}")

    async def run_workflow_and_wait(
        self,
        workflow_id: str,
        options: Optional[RunWorkflowAndWaitOptionsSchema] = None,
    ) -> Union[WorkflowRunResultSchema, List[WorkflowRunResultSchema]]:
        """
        Run a workflow and wait for results by polling until completion.

        Args:
            workflow_id: The ID of the workflow to run.
            options: Optional workflow run and wait options.

        Returns:
            Dict containing the final workflow run results, or if return_intermediate_results=True,
            a list of all polled result dictionaries.
        """
        if options is None:
            options = RunWorkflowAndWaitOptionsSchema()

        # Run the workflow
        run_response = await self.run_workflow(
            workflow_id=workflow_id,
            options=WorkflowRunOptionsSchema(
                args=options.args,
                files=options.files,
                use_states=options.use_states,
                preserve_state=options.preserve_state,
                no_intelligence=options.no_intelligence,
                record_session=options.record_session,
                browser_session_id=options.browser_session_id,
            ),
        )

        run_id = run_response.run_id
        start_time = time.time()
        intermediate_results = []
        consecutive_errors = 0
        max_consecutive_errors = 3

        # Poll for results
        while True:
            # Check timeout if specified
            if (
                options.timeout is not None
                and time.time() - start_time >= options.timeout
            ):
                raise WitriumClientException(
                    f"Workflow execution timed out after {options.timeout} seconds"
                )

            try:
                results = await self.get_workflow_results(run_id)
                consecutive_errors = 0  # Reset on success
            except WitriumClientException as e:
                # Check if this is a transient network error (retry-able)
                if "Network error" in str(e) or "connection was interrupted" in str(e):
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        raise WitriumClientException(
                            f"Failed to get workflow results after {max_consecutive_errors} "
                            f"consecutive network errors: {str(e)}"
                        )
                    # Exponential backoff: wait longer after each error
                    backoff_time = options.polling_interval * (
                        2 ** (consecutive_errors - 1)
                    )
                    logger.warning(
                        f"Transient error polling workflow results (attempt {consecutive_errors}/{max_consecutive_errors}), "
                        f"retrying in {backoff_time}s: {str(e)}"
                    )
                    await asyncio.sleep(backoff_time)
                    continue
                else:
                    # Non-transient error, re-raise immediately
                    raise

            # Store intermediate results if requested
            if options.return_intermediate_results:
                intermediate_results.append(results)

            # Call progress callback if provided
            if options.on_progress:
                options.on_progress(results)

            # Check if workflow run has completed
            if results.status in WorkflowRunStatus.TERMINAL_STATUSES:
                return (
                    intermediate_results
                    if options.return_intermediate_results
                    else results
                )

            # Wait before polling again
            await asyncio.sleep(options.polling_interval)

    async def wait_until_state(
        self,
        run_id: str,
        target_status: str,
        options: Optional[WaitUntilStateOptionsSchema] = None,
    ) -> WorkflowRunResultSchema:
        """
        Wait for a workflow run to reach a specific status by polling.

        Args:
            run_id: The ID of the workflow run to wait for.
            target_status: The status to wait for (e.g., WorkflowRunStatus.RUNNING).
            options: Optional wait options.

        Returns:
            WorkflowRunResultSchema when the target status is reached.

        Raises:
            WitriumClientException: If timeout is reached or workflow reaches an unexpected terminal status.
        """
        if options is None:
            options = WaitUntilStateOptionsSchema()

        # Wait for minimum time before starting to poll
        if options.min_wait_time > 0:
            await asyncio.sleep(options.min_wait_time)

        def _check_all_executions_completed(results: WorkflowRunResultSchema) -> bool:
            """Check if all executions have completed status."""
            if not results.executions:
                return False
            return results.executions[-1].status == AgentExecutionStatus.COMPLETED

        start_time = time.time()
        consecutive_errors = 0
        max_consecutive_errors = 3

        # Poll until conditions are met
        while True:
            # Check timeout if specified
            if (
                options.timeout is not None
                and time.time() - start_time >= options.timeout
            ):
                target_status_name = WorkflowRunStatus.get_status_name(target_status)
                condition_msg = f"status '{target_status_name}'"
                if options.all_instructions_executed:
                    condition_msg += " and all instructions executed"
                raise WitriumClientException(
                    f"Workflow run did not reach {condition_msg} within {options.timeout} seconds"
                )

            try:
                results = await self.get_workflow_results(run_id)
                consecutive_errors = 0  # Reset on success
            except WitriumClientException as e:
                # Check if this is a transient network error (retry-able)
                if "Network error" in str(e) or "connection was interrupted" in str(e):
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        raise WitriumClientException(
                            f"Failed to get workflow results after {max_consecutive_errors} "
                            f"consecutive network errors: {str(e)}"
                        )
                    # Exponential backoff: wait longer after each error
                    backoff_time = options.polling_interval * (
                        2 ** (consecutive_errors - 1)
                    )
                    logger.warning(
                        f"Transient error polling workflow results (attempt {consecutive_errors}/{max_consecutive_errors}), "
                        f"retrying in {backoff_time}s: {str(e)}"
                    )
                    await asyncio.sleep(backoff_time)
                    continue
                else:
                    # Non-transient error, re-raise immediately
                    raise

            # Check if workflow has reached the target status
            status_reached = results.status == target_status
            all_executions_completed = (
                _check_all_executions_completed(results)
                if options.all_instructions_executed
                else True
            )

            if status_reached and all_executions_completed:
                return results

            # Check if workflow has reached a terminal status that's not our target
            if (
                results.status in WorkflowRunStatus.TERMINAL_STATUSES
                and results.status != target_status
            ):
                current_status_name = WorkflowRunStatus.get_status_name(results.status)
                target_status_name = WorkflowRunStatus.get_status_name(target_status)
                raise WitriumClientException(
                    f"Workflow run reached terminal status '{current_status_name}' before reaching target status '{target_status_name}'"
                )

            # Wait before polling again
            await asyncio.sleep(options.polling_interval)

    async def cancel_run(self, run_id: str) -> WorkflowRunSchema:
        """
        Cancel a workflow run and clean up associated resources.

        Args:
            run_id: The ID of the workflow run to cancel.

        Returns:
            Dict containing the workflow run results.
        """
        url = f"{self.base_url}/v1/runs/{run_id}/cancel"

        try:
            response = await self._client.post(url)
            response.raise_for_status()
            return WorkflowRunSchema.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            error_detail = await self._extract_error_detail(e.response)
            raise WitriumClientException(
                f"Error cancelling workflow run: {error_detail} (Status code: {e.response.status_code})"
            )
        except Exception as e:
            raise WitriumClientException(f"Error cancelling workflow run: {str(e)}")

    async def _extract_error_detail(self, response: httpx.Response) -> str:
        """Extract error detail from response."""
        try:
            error_json = response.json()
            if "detail" in error_json:
                return error_json["detail"]
            return str(error_json)
        except Exception:
            return response.text or "Unknown error"

    async def run_talent(
        self,
        talent_id: str,
        options: Optional[TalentRunOptionsSchema] = None,
    ) -> TalentResultSchema:
        """
        Run a talent by ID.

        Args:
            talent_id: The ID of the talent to run.
            options: Optional talent run options.
                If browser_session_id is provided (or client.session_id is set),
                the use_states from the browser session will be used instead of
                options.use_states.

        Returns:
            The result of the talent run.
        """
        if options is None:
            options = TalentRunOptionsSchema()

        url = f"{self.base_url}/v1/talents/{talent_id}/run"

        # Build payload with only defined values
        payload = {}
        if options.args is not None:
            payload["args"] = options.args
        if options.files is not None:
            payload["files"] = [file.model_dump() for file in options.files]
        if options.use_states is not None:
            payload["use_states"] = options.use_states
        if options.preserve_state is not None:
            payload["preserve_state"] = options.preserve_state
        # Use client's session_id if available and no explicit session provided
        browser_session_id = options.browser_session_id or self.session_id
        if browser_session_id is not None:
            payload["browser_session_id"] = browser_session_id

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            return TalentResultSchema.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            error_detail = await self._extract_error_detail(e.response)
            raise WitriumClientException(
                f"Error running talent: {error_detail} (Status code: {e.response.status_code})"
            )
        except Exception as e:
            raise WitriumClientException(f"Error running talent: {str(e)}")

    async def create_browser_session(
        self, options: Optional[BrowserSessionCreateOptions] = None
    ) -> BrowserSessionSchema:
        """
        Create a standalone browser session.

        Args:
            options: Optional browser session creation options.
                The use_states set here will apply to all workflow and talent runs
                that use this browser session. Individual run options' use_states
                will be ignored when using this session.

        Returns:
            BrowserSessionSchema containing session details.
        """
        if options is None:
            options = BrowserSessionCreateOptions()

        url = f"{self.base_url}/v1/browser-sessions"
        payload = options.model_dump()

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            return BrowserSessionSchema.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            error_detail = await self._extract_error_detail(e.response)
            raise WitriumClientException(
                f"Error creating browser session: {error_detail} (Status code: {e.response.status_code})"
            )
        except Exception as e:
            raise WitriumClientException(f"Error creating browser session: {str(e)}")

    async def list_browser_sessions(self) -> ListBrowserSessionSchema:
        """
        List all active browser sessions.

        Returns:
            BrowserSessionListResponse containing list of sessions and total count.
        """
        url = f"{self.base_url}/v1/browser-sessions"

        try:
            response = await self._client.get(url)
            response.raise_for_status()
            return ListBrowserSessionSchema.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            error_detail = await self._extract_error_detail(e.response)
            raise WitriumClientException(
                f"Error listing browser sessions: {error_detail} (Status code: {e.response.status_code})"
            )
        except Exception as e:
            raise WitriumClientException(f"Error listing browser sessions: {str(e)}")

    async def get_browser_session(self, session_id: str) -> BrowserSessionSchema:
        """
        Get details of a specific browser session.

        Args:
            session_id: The UUID of the browser session.

        Returns:
            BrowserSessionSchema containing session details.
        """
        url = f"{self.base_url}/v1/browser-sessions/{session_id}"

        try:
            response = await self._client.get(url)
            response.raise_for_status()
            return BrowserSessionSchema.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            error_detail = await self._extract_error_detail(e.response)
            raise WitriumClientException(
                f"Error getting browser session: {error_detail} (Status code: {e.response.status_code})"
            )
        except Exception as e:
            raise WitriumClientException(f"Error getting browser session: {str(e)}")

    async def close_browser_session(
        self, session_id: str, options: Optional[BrowserSessionCloseOptions] = None
    ) -> Dict[str, bool]:
        """
        Close a browser session.

        Args:
            session_id: The UUID of the browser session to close.
            options: Optional browser session close options.

        Returns:
            Dict containing success status.
        """
        if options is None:
            options = BrowserSessionCloseOptions()

        url = f"{self.base_url}/v1/browser-sessions/{session_id}"
        payload = options.model_dump()

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = await self._extract_error_detail(e.response)
            raise WitriumClientException(
                f"Error closing browser session: {error_detail} (Status code: {e.response.status_code})"
            )
        except Exception as e:
            raise WitriumClientException(f"Error closing browser session: {str(e)}")
