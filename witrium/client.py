import time
import asyncio
import logging
import httpx
from typing import Dict, List, Optional, Any, Union, Callable
from tenacity import retry, stop_after_delay, wait_fixed, retry_if_result
from witrium.types import (
    WorkflowRunSubmittedSchema,
    WorkflowRunResultsSchema,
    WorkflowRunSchema,
    FileUpload,
    WorkflowRunStatus,
    AgentExecutionStatus,
    TalentResultSchema,
)

# Setup logger
logger = logging.getLogger("witrium_client")


class WitriumClientException(Exception):
    """Base exception for Witrium Client errors."""

    pass


class WitriumClient:
    """
    Base class for Witrium API Client.
    Not meant to be used directly - use SyncWitriumClient or AsyncWitriumClient.
    """

    def __init__(
        self, base_url: str, api_token: str, timeout: int = 60, verify_ssl: bool = True
    ):
        """
        Initialize the Witrium client.
        Args:
            api_token: The API token for authentication.
        """
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._headers = {"X-Witrium-Key": api_token, "Content-Type": "application/json"}


class SyncWitriumClient(WitriumClient):
    """Synchronous Witrium API client."""

    def __init__(self, api_token: str, timeout: int = 60, verify_ssl: bool = True):
        """Initialize the synchronous client."""
        super().__init__("https://api.witrium.com", api_token, timeout, verify_ssl)
        self._client = httpx.Client(
            timeout=self.timeout, verify=self.verify_ssl, headers=self._headers
        )

    def close(self):
        """Close the underlying HTTP client."""
        if self._client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def run_workflow(
        self,
        workflow_id: str,
        args: Optional[Dict[str, Union[str, int, float]]] = None,
        files: Optional[List[FileUpload]] = None,
        use_states: Optional[List[str]] = None,
        preserve_state: Optional[str] = None,
        no_intelligence: bool = False,
        record_session: Optional[bool] = False,
        keep_session_alive: bool = False,
        use_existing_session: Optional[str] = None,
    ) -> WorkflowRunSubmittedSchema:
        """
        Run a workflow by ID.

        Args:
            workflow_id: The ID of the workflow to run.
            args: Optional arguments to pass to the workflow.
            files: Optional list of files to upload with the workflow.
            use_states: Optional list of state names to use.
            preserve_state: Optional state name to preserve.
            no_intelligence: Whether to run without AI intelligence.
            record_session: Whether to record the session.
            keep_session_alive: Whether to keep the session alive.
            use_existing_session: The ID of the existing session to use.

        Returns:
            Dict containing workflow_id, run_id and status.
        """
        url = f"{self.base_url}/v1/workflows/{workflow_id}/run"
        payload = {
            "args": args,
            "files": [file.model_dump() for file in files] if files else None,
            "use_states": use_states,
            "preserve_state": preserve_state,
            "no_intelligence": no_intelligence,
            "record_session": record_session,
            "keep_session_alive": keep_session_alive,
            "use_existing_session": use_existing_session,
        }

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

    def get_workflow_results(self, run_id: str) -> WorkflowRunResultsSchema:
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
            return WorkflowRunResultsSchema.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            error_detail = self._extract_error_detail(e.response)
            raise WitriumClientException(
                f"Error getting workflow results: {error_detail} (Status code: {e.response.status_code})"
            )
        except Exception as e:
            raise WitriumClientException(f"Error getting workflow results: {str(e)}")

    def run_workflow_and_wait(
        self,
        workflow_id: str,
        args: Optional[Dict[str, Union[str, int, float]]] = None,
        files: Optional[List[FileUpload]] = None,
        use_states: Optional[List[str]] = None,
        preserve_state: Optional[str] = None,
        no_intelligence: bool = False,
        record_session: Optional[bool] = False,
        keep_session_alive: bool = False,
        use_existing_session: Optional[str] = None,
        polling_interval: int = 5,
        timeout: int = 300,
        return_intermediate_results: bool = False,
        on_progress: Optional[Callable[[WorkflowRunResultsSchema], Any]] = None,
    ) -> Union[WorkflowRunResultsSchema, List[WorkflowRunResultsSchema]]:
        """
        Run a workflow and wait for results by polling until completion.

        Args:
            workflow_id: The ID of the workflow to run.
            args: Optional arguments to pass to the workflow.
            files: Optional list of files to upload with the workflow.
            use_states: Optional list of session IDs to use.
            preserve_state: Optional session ID to preserve.
            no_intelligence: Whether to run without AI intelligence.
            record_session: Whether to record the session.
            keep_session_alive: Whether to keep the session alive after the last instruction is run.
            use_existing_session: The ID of the existing session to use. This is useful when you want to run a workflow in an existing session.
            polling_interval: Seconds to wait between polling attempts.
            timeout: Maximum seconds to poll before timing out.
            return_intermediate_results: If True, returns a list of all polled results.
            on_progress: Optional callback function that receives each intermediate result.
                         This is called on each polling iteration with the current results.

        Returns:
            Dict containing the final workflow run results, or if return_intermediate_results=True,
            a list of all polled result dictionaries.
        """
        # Run the workflow
        run_response = self.run_workflow(
            workflow_id=workflow_id,
            args=args,
            files=files,
            use_states=use_states,
            preserve_state=preserve_state,
            no_intelligence=no_intelligence,
            record_session=record_session,
            keep_session_alive=keep_session_alive,
            use_existing_session=use_existing_session,
        )

        run_id = run_response.run_id
        start_time = time.time()
        intermediate_results = []

        # Poll for results
        while time.time() - start_time < timeout:
            results = self.get_workflow_results(run_id)

            # Store intermediate results if requested
            if return_intermediate_results:
                intermediate_results.append(results)

            # Call progress callback if provided
            if on_progress:
                on_progress(results)

            # Check if workflow has completed
            if results.status in WorkflowRunStatus.TERMINAL_STATUSES:
                return intermediate_results if return_intermediate_results else results

            # Wait before polling again
            time.sleep(polling_interval)

        raise WitriumClientException(
            f"Workflow execution timed out after {timeout} seconds"
        )

    def wait_until_state(
        self,
        run_id: str,
        target_status: str,
        all_instructions_executed: bool = False,
        min_wait_time: int = 0,
        polling_interval: int = 2,
        timeout: int = 60,
    ) -> WorkflowRunResultsSchema:
        """
        Wait for a workflow run to reach a specific status by polling.

        Args:
            run_id: The ID of the workflow run to wait for.
            target_status: The status to wait for (e.g., WorkflowRunStatus.RUNNING).
            all_instructions_executed: If True, also wait for all executions to be completed.
            min_wait_time: Minimum time in seconds to wait before starting polling. Useful when you know approximately how long the workflow will take.
            polling_interval: Seconds to wait between polling attempts.
            timeout: Maximum seconds to poll before timing out.

        Returns:
            WorkflowRunResultsSchema when the target status is reached.

        Raises:
            WitriumClientException: If timeout is reached or workflow reaches an unexpected terminal status.
        """

        # Wait for minimum time before starting to poll
        if min_wait_time > 0:
            time.sleep(min_wait_time)

        def _check_all_executions_completed(results: WorkflowRunResultsSchema) -> bool:
            """Check if all executions have completed status."""
            if not results.executions:
                return False
            return results.executions[-1].status == AgentExecutionStatus.COMPLETED

        def _should_continue_polling(results: WorkflowRunResultsSchema) -> bool:
            """Determine if we should continue polling based on target status and execution completion."""
            status_not_reached = results.status != target_status
            terminal_status_reached = (
                results.status in WorkflowRunStatus.TERMINAL_STATUSES
            )

            # If we've reached a terminal status but it's not our target, stop retrying
            if terminal_status_reached and status_not_reached:
                return False

            # If target status is not reached, continue polling
            if status_not_reached:
                return True

            # If target status is reached but we also need all instructions executed
            if all_instructions_executed and not _check_all_executions_completed(
                results
            ):
                return True

            # All conditions met, stop polling
            return False

        @retry(
            stop=stop_after_delay(timeout),
            wait=wait_fixed(polling_interval),
            retry=retry_if_result(_should_continue_polling),
        )
        def _poll_for_status():
            results = self.get_workflow_results(run_id)

            # Check if workflow has reached the target status
            status_reached = results.status == target_status
            all_executions_completed = (
                _check_all_executions_completed(results)
                if all_instructions_executed
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

            # Return results for retry evaluation
            return results

        try:
            return _poll_for_status()
        except Exception as e:
            if "retry" in str(e).lower():
                target_status_name = WorkflowRunStatus.get_status_name(target_status)
                condition_msg = f"status '{target_status_name}'"
                if all_instructions_executed:
                    condition_msg += " and all instructions executed"
                raise WitriumClientException(
                    f"Workflow run did not reach {condition_msg} within {timeout} seconds"
                )
            raise

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
        args: Optional[Dict[str, Union[str, int, float]]] = None,
        files: Optional[List[FileUpload]] = None,
        use_states: Optional[List[str]] = None,
        preserve_state: Optional[str] = None,
        keep_session_alive: bool = False,
        use_existing_session: Optional[str] = None,
    ) -> TalentResultSchema:
        """
        Run a talent by ID.

        Args:
            talent_id: The ID of the talent to run.
            args: Optional arguments to pass to the workflow.
            files: Optional list of files to upload with the workflow.
            use_states: Optional list of state names to use.
            preserve_state: Optional state name to preserve.
            keep_session_alive: Whether to keep the session alive.
            use_existing_session: The ID of the existing session to use.

        Returns:
            The result of the talent run.
        """
        url = f"{self.base_url}/v1/talents/{talent_id}/run"
        payload = {
            "args": args,
            "files": [file.model_dump() for file in files] if files else None,
            "use_states": use_states,
            "preserve_state": preserve_state,
            "keep_session_alive": keep_session_alive,
            "use_existing_session": use_existing_session,
        }
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


class AsyncWitriumClient(WitriumClient):
    """Asynchronous Witrium API client."""

    def __init__(self, api_token: str, timeout: int = 60, verify_ssl: bool = True):
        """Initialize the asynchronous client."""
        super().__init__("https://api.witrium.com", api_token, timeout, verify_ssl)
        self._client = httpx.AsyncClient(
            timeout=self.timeout, verify=self.verify_ssl, headers=self._headers
        )

    async def close(self):
        """Close the underlying HTTP client."""
        if self._client:
            await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    async def run_workflow(
        self,
        workflow_id: str,
        args: Optional[Dict[str, Union[str, int, float]]] = None,
        files: Optional[List[FileUpload]] = None,
        use_states: Optional[List[str]] = None,
        preserve_state: Optional[str] = None,
        no_intelligence: bool = False,
        record_session: Optional[bool] = False,
        keep_session_alive: bool = False,
        use_existing_session: Optional[str] = None,
    ) -> WorkflowRunSubmittedSchema:
        """
        Run a workflow by ID.

        Args:
            workflow_id: The ID of the workflow to run.
            args: Optional arguments to pass to the workflow.
            files: Optional list of files to upload with the workflow.
            use_states: Optional list of state names to use.
            preserve_state: Optional state name to preserve.
            no_intelligence: Whether to run without AI intelligence.
            record_session: Whether to record the session.
            keep_session_alive: Whether to keep the session alive.
            use_existing_session: The ID of the existing session to use.

        Returns:
            Dict containing workflow_id, run_id and status.
        """
        url = f"{self.base_url}/v1/workflows/{workflow_id}/run"
        payload = {
            "args": args,
            "files": [file.model_dump() for file in files] if files else None,
            "use_states": use_states,
            "preserve_state": preserve_state,
            "no_intelligence": no_intelligence,
            "record_session": record_session,
            "keep_session_alive": keep_session_alive,
            "use_existing_session": use_existing_session,
        }

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

    async def get_workflow_results(self, run_id: str) -> WorkflowRunResultsSchema:
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
            return WorkflowRunResultsSchema.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            error_detail = await self._extract_error_detail(e.response)
            raise WitriumClientException(
                f"Error getting workflow results: {error_detail} (Status code: {e.response.status_code})"
            )
        except Exception as e:
            raise WitriumClientException(f"Error getting workflow results: {str(e)}")

    async def run_workflow_and_wait(
        self,
        workflow_id: str,
        args: Optional[Dict[str, Union[str, int, float]]] = None,
        files: Optional[List[FileUpload]] = None,
        use_states: Optional[List[str]] = None,
        preserve_state: Optional[str] = None,
        no_intelligence: bool = False,
        record_session: Optional[bool] = False,
        keep_session_alive: bool = False,
        use_existing_session: Optional[str] = None,
        polling_interval: int = 5,
        timeout: int = 300,
        return_intermediate_results: bool = False,
        on_progress: Optional[Callable[[WorkflowRunResultsSchema], Any]] = None,
    ) -> Union[WorkflowRunResultsSchema, List[WorkflowRunResultsSchema]]:
        """
        Run a workflow and wait for results by polling until completion.

        Args:
            workflow_id: The ID of the workflow to run.
            args: Optional arguments to pass to the workflow.
            files: Optional list of files to upload with the workflow.
            use_states: Optional list of session IDs to use.
            preserve_state: Optional session ID to preserve.
            no_intelligence: Whether to run without AI intelligence.
            record_session: Whether to record the session.
            keep_session_alive: Whether to keep the session alive after the last instruction is run.
            use_existing_session: The ID of the existing session to use. This is useful when you want to run a workflow in an existing session.
            polling_interval: Seconds to wait between polling attempts.
            timeout: Maximum seconds to poll before timing out.
            return_intermediate_results: If True, returns a list of all polled results.
            on_progress: Optional callback function that receives each intermediate result.
                         This is called on each polling iteration with the current results.

        Returns:
            Dict containing the final workflow run results, or if return_intermediate_results=True,
            a list of all polled result dictionaries.
        """
        # Run the workflow
        run_response = await self.run_workflow(
            workflow_id=workflow_id,
            args=args,
            files=files,
            use_states=use_states,
            preserve_state=preserve_state,
            no_intelligence=no_intelligence,
            record_session=record_session,
            keep_session_alive=keep_session_alive,
            use_existing_session=use_existing_session,
        )

        run_id = run_response.run_id
        start_time = time.time()
        intermediate_results = []

        # Poll for results
        while time.time() - start_time < timeout:
            results = await self.get_workflow_results(run_id)

            # Store intermediate results if requested
            if return_intermediate_results:
                intermediate_results.append(results)

            # Call progress callback if provided
            if on_progress:
                on_progress(results)

            # Check if workflow run has completed
            if results.status in WorkflowRunStatus.TERMINAL_STATUSES:
                return intermediate_results if return_intermediate_results else results

            # Wait before polling again
            await asyncio.sleep(polling_interval)

        raise WitriumClientException(
            f"Workflow execution timed out after {timeout} seconds"
        )

    async def wait_until_state(
        self,
        run_id: str,
        target_status: str,
        all_instructions_executed: bool = False,
        min_wait_time: int = 0,
        polling_interval: int = 2,
        timeout: int = 60,
    ) -> WorkflowRunResultsSchema:
        """
        Wait for a workflow run to reach a specific status by polling.

        Args:
            run_id: The ID of the workflow run to wait for.
            target_status: The status to wait for (e.g., WorkflowRunStatus.RUNNING).
            all_instructions_executed: If True, also wait for all executions to be completed.
            min_wait_time: Minimum time in seconds to wait before starting polling. Useful when you know approximately how long the workflow will take.
            polling_interval: Seconds to wait between polling attempts.
            timeout: Maximum seconds to poll before timing out.

        Returns:
            WorkflowRunResultsSchema when the target status is reached.

        Raises:
            WitriumClientException: If timeout is reached or workflow reaches an unexpected terminal status.
        """

        # Wait for minimum time before starting to poll
        if min_wait_time > 0:
            await asyncio.sleep(min_wait_time)

        def _check_all_executions_completed(results: WorkflowRunResultsSchema) -> bool:
            """Check if all executions have completed status."""
            if not results.executions:
                return False
            return results.executions[-1].status == AgentExecutionStatus.COMPLETED

        def _should_continue_polling(results: WorkflowRunResultsSchema) -> bool:
            """Determine if we should continue polling based on target status and execution completion."""
            status_not_reached = results.status != target_status
            terminal_status_reached = (
                results.status in WorkflowRunStatus.TERMINAL_STATUSES
            )

            # If we've reached a terminal status but it's not our target, stop retrying
            if terminal_status_reached and status_not_reached:
                return False

            # If target status is not reached, continue polling
            if status_not_reached:
                return True

            # If target status is reached but we also need all instructions executed
            if all_instructions_executed and not _check_all_executions_completed(
                results
            ):
                return True

            # All conditions met, stop polling
            return False

        @retry(
            stop=stop_after_delay(timeout),
            wait=wait_fixed(polling_interval),
            retry=retry_if_result(_should_continue_polling),
        )
        async def _poll_for_status():
            results = await self.get_workflow_results(run_id)

            # Check if workflow has reached the target status
            status_reached = results.status == target_status
            all_executions_completed = (
                _check_all_executions_completed(results)
                if all_instructions_executed
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

            # Return results for retry evaluation
            return results

        try:
            return await _poll_for_status()
        except Exception as e:
            if "retry" in str(e).lower():
                target_status_name = WorkflowRunStatus.get_status_name(target_status)
                condition_msg = f"status '{target_status_name}'"
                if all_instructions_executed:
                    condition_msg += " and all instructions executed"
                raise WitriumClientException(
                    f"Workflow run did not reach {condition_msg} within {timeout} seconds"
                )
            raise

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
        args: Optional[Dict[str, Union[str, int, float]]] = None,
        files: Optional[List[FileUpload]] = None,
        use_states: Optional[List[str]] = None,
        preserve_state: Optional[str] = None,
        keep_session_alive: bool = False,
        use_existing_session: Optional[str] = None,
    ) -> TalentResultSchema:
        """
        Run a talent by ID.

        Args:
            talent_id: The ID of the talent to run.
            args: Optional arguments to pass to the workflow.
            files: Optional list of files to upload with the workflow.
            use_states: Optional list of state names to use.
            preserve_state: Optional state name to preserve.
            keep_session_alive: Whether to keep the session alive.
            use_existing_session: The ID of the existing session to use.

        Returns:
            The result of the talent run.
        """
        url = f"{self.base_url}/v1/talents/{talent_id}/run"
        payload = {
            "args": args,
            "files": [file.model_dump() for file in files] if files else None,
            "use_states": use_states,
            "preserve_state": preserve_state,
            "keep_session_alive": keep_session_alive,
            "use_existing_session": use_existing_session,
        }
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
