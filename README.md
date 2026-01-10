# Witrium Client

A Python client library for interacting with the Witrium API. Witrium is a cloud-based browser automation platform that allows you to create and execute web automations through a visual interface and control them programmatically via this client.

## How Witrium Works

Witrium operates by spinning up browser instances in the cloud to execute predefined automations that you create through the Witrium UI. Here's the typical workflow:

1. **Create Automations via UI**: You use the Witrium web interface to record and define your automations (workflows)
2. **Execute via API**: You use this Python client to trigger those automations programmatically
3. **Cloud Execution**: Witrium runs your automation in a real browser instance in the cloud
4. **Retrieve Results**: You poll for results and handle the automation outcomes

Each workflow is identified by a unique `workflow_id` and can accept arguments to customize its execution.

## Installation

```bash
pip install witrium
```

## Quick Start

The snippet below shows the **minimum** you need to get up-and-running:

```python
from witrium import SyncWitriumClient, WorkflowRunStatus

# 1. Provide your API endpoint & token (export these as env-vars in production)
api_token = "YOUR_WITRIUM_API_TOKEN"  # Obtain from dashboard

with SyncWitriumClient(api_token=api_token) as client:
    # 2. Kick-off the **login** workflow and keep the browser alive
    login = client.run_workflow(
        workflow_id="login-workflow-id",  # This workflow performs the sign-in steps
        args={"username": "user@example.com", "password": "secretPass!"},
        keep_session_alive=True  # 🔑 keep the browser running after login
    )

    # 3. Block until the browser is *ready for reuse*
    client.wait_until_state(
        run_id=login.run_id,
        target_status=WorkflowRunStatus.RUNNING,  # Wait until browser is alive
        all_instructions_executed=True  # …and the last login step finished
    )

    # 4. Re-use that **same** browser session in a follow-up workflow
    scrape = client.run_workflow(
        workflow_id="dashboard-scrape-workflow-id",
        args={"section": "sales"},
        use_existing_session=login.run_id  # 👈 same browser instance
    )

    # 5. Wait for the scrape to finish and collect the results
    results = client.wait_until_state(
        run_id=scrape.run_id,
        target_status=WorkflowRunStatus.COMPLETED
    )

print("Sales data:", results.result)
```

---

## Workflow Lifecycle & Polling Essentials

`client.run_workflow(...)` **only submits** a job – the real browser work happens asynchronously in the cloud.  Keep these steps in mind whenever you design multi-step automations:

1. **Submit** – your call returns instantly with a `run_id`.
2. **Poll / Wait** – use `wait_until_state()` (or `run_workflow_and_wait()`) to block until the run reaches:
   • `WorkflowRunStatus.RUNNING` – the browser has spun-up and is ready (handy when you enabled `keep_session_alive`).  
   • `WorkflowRunStatus.COMPLETED` – the workflow has finished executing.
3. **Chain or Fetch Results** – once the target state is reached you can either run another workflow (chaining sessions) or read the data via `get_workflow_results()`.

### When to wait for which state?

| Scenario | Recommended `target_status` | Extra flags |
|----------|-----------------------------|-------------|
| You **saved state** using `preserve_state` | `COMPLETED` | – |
| You **kept the session alive** using `keep_session_alive` **and intend to reuse it** | `RUNNING` | `all_instructions_executed=True` |

> ⏳ **Tip:** For very long login flows (e.g. multi-factor auth) combine `min_wait_time` with `polling_interval` to reduce server load.

### Concurrency vs. Serial Execution

• **State Preservation (`preserve_state`)** – Each follow-up workflow spins up its **own** browser.  Scale **horizontally** & run many in parallel.  
• **Session Persistence (`keep_session_alive`)** – All follow-up workflows share **one** browser instance.  Run them **serially** (until multi-tab support lands).

---

## Common Use Cases and Session Management

### The Authentication Challenge

A common pattern in web automation involves authentication: you need to log into a service first, then perform actions in the authenticated session. Witrium provides two powerful approaches to handle this:

#### Approach 1: State Preservation (Concurrent-Friendly)
- Best for: Running multiple post-login automations concurrently
- How it works: Save browser state after login, then restore it in new browser instances

#### Approach 2: Session Persistence (Resource-Efficient)
- Best for: Sequential automations that need to share the exact same browser session
- How it works: Keep the browser alive after login, run subsequent automations in the same instance

## Session Management Patterns

### Pattern 1: Disconnected Sessions with State Preservation

This approach allows you to save the browser state (cookies, localStorage, etc.) after a login workflow and then restore that state in new browser instances for subsequent workflows.

**Advantages:**
- Multiple post-login workflows can run concurrently
- Each workflow gets its own browser instance
- Horizontal scaling of browser instances
- Robust isolation between concurrent executions

**Use Case Example:**

```python
from witrium import SyncWitriumClient, WorkflowRunStatus

with SyncWitriumClient(api_token="your-api-token") as client:
    # Step 1: Run login workflow and preserve the authenticated state
    login_response = client.run_workflow(
        workflow_id="login-workflow-id",
        args={"username": "user@example.com", "password": "secure123"},
        preserve_state="authenticated-session"  # Save state with this name
    )

    # Step 2: Wait for login to complete
    login_results = client.wait_until_state(
        run_id=login_response.run_id,
        target_status=WorkflowRunStatus.COMPLETED
    )

# Step 3: Run multiple post-login workflows concurrently
# Each will spawn a new browser but restore the authenticated state

# Workflow A: Extract data from dashboard
dashboard_response = client.run_workflow(
    workflow_id="dashboard-scraping-workflow-id",
    args={"report_type": "monthly"},
    use_states=["authenticated-session"]  # Restore the saved state
)

# Workflow B: Update user profile (can run concurrently)
profile_response = client.run_workflow(
    workflow_id="profile-update-workflow-id",
    args={"new_email": "newemail@example.com"},
    use_states=["authenticated-session"]  # Same state, different browser instance
)

# Both workflows are now running concurrently in separate browser instances
# but both have access to the authenticated session
```

### Pattern 2: Persistent Session with Keep-Alive

This approach keeps the browser instance alive after the login workflow completes, allowing subsequent workflows to run in the same browser session.

**Advantages:**
- More resource-efficient (reuses same browser instance)
- Maintains exact session continuity
- No need to restore state (session never ends)
- Faster execution for subsequent workflows

**Limitations:**
- Subsequent workflows must run serially (one after another)
- Cannot run multiple post-login workflows concurrently in the same session

**Use Case Example:**

```python
from witrium import SyncWitriumClient, WorkflowRunStatus

with SyncWitriumClient(api_token="your-api-token") as client:
    # Step 1: Run login workflow and keep the browser session alive
    login_response = client.run_workflow(
        workflow_id="login-workflow-id",
        args={"username": "user@example.com", "password": "secure123"},
        keep_session_alive=True  # Keep browser instance running
    )

    # Step 2: Wait for login to complete and start running
    # We wait for RUNNING status because the browser is kept alive
    login_results = client.wait_until_state(
        run_id=login_response.run_id,
        target_status=WorkflowRunStatus.RUNNING,
        all_instructions_executed=True  # Ensure login steps are done
    )

    # Step 3: Run subsequent workflows in the same browser session
    # These must run serially, not concurrently

    # Workflow A: Extract data from dashboard
    dashboard_response = client.run_workflow(
        workflow_id="dashboard-scraping-workflow-id",
        args={"report_type": "monthly"},
        use_existing_session=login_response.run_id  # Use the live session
    )

    # Wait for dashboard workflow to complete before next one
    dashboard_results = client.wait_until_state(
        run_id=dashboard_response.run_id,
        target_status=WorkflowRunStatus.COMPLETED
    )

    # Workflow B: Update user profile (must wait for previous to complete)
    profile_response = client.run_workflow(
        workflow_id="profile-update-workflow-id",
        args={"new_email": "newemail@example.com"},
        use_existing_session=login_response.run_id  # Same live session
    )
```

### Choosing the Right Pattern

| Factor | State Preservation | Session Persistence |
|--------|-------------------|-------------------|
| **Concurrency** | ✅ Multiple workflows can run simultaneously | ❌ Must run serially |
| **Resource Usage** | Higher (multiple browser instances) | ✅ Lower (single browser instance) |
| **Isolation** | ✅ Complete isolation between workflows | ❌ Shared session state |
| **Setup Complexity** | Medium (manage state names) | ✅ Simple (just workflow run IDs) |
| **Use Case** | Bulk data processing, parallel operations | Sequential workflows, state-dependent operations |

## Complete Examples

### Example 1: E-commerce Data Extraction (State Preservation)

```python
from witrium import SyncWitriumClient, WorkflowRunStatus
import concurrent.futures
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_category_data(client, category, state_name):
    """Extract data for a specific product category."""
    try:
        response = client.run_workflow(
            workflow_id="category-scraper-workflow",
            args={"category": category},
            use_states=[state_name]
        )

        results = client.wait_until_state(
            run_id=response.run_id,
            target_status=WorkflowRunStatus.COMPLETED
        )

        return {"category": category, "data": results.result}
    except Exception as e:
        logger.error(f"Failed to extract {category}: {e}")
        return {"category": category, "error": str(e)}


with SyncWitriumClient(api_token="your-api-token") as client:
    # Step 1: Login and save state
    logger.info("Logging into e-commerce platform...")
    login_response = client.run_workflow(
        workflow_id="ecommerce-login-workflow",
        args={"email": "seller@example.com", "password": "secure123"},
        preserve_state="ecommerce-authenticated"
    )

    # Wait for login completion
    client.wait_until_state(
        run_id=login_response.run_id,
        target_status=WorkflowRunStatus.COMPLETED
    )
    logger.info("Login completed, state preserved")

    # Step 2: Extract data from multiple categories concurrently
    categories = ["electronics", "clothing", "home-garden", "books", "sports"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all category extraction tasks concurrently
        future_to_category = {
            executor.submit(extract_category_data, client, category,
                            "ecommerce-authenticated"): category
            for category in categories
        }

        results = []
        for future in concurrent.futures.as_completed(future_to_category):
            result = future.result()
            results.append(result)
            logger.info(f"Completed extraction for {result['category']}")

    logger.info(f"Extracted data from {len(results)} categories")
    for result in results:
        if "error" in result:
            logger.error(f"Error in {result['category']}: {result['error']}")
        else:
            logger.info(f"{result['category']}: {len(result['data'])} items extracted")
```

### Example 2: Banking Workflow (Session Persistence)

```python
from witrium import SyncWitriumClient, WorkflowRunStatus
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with SyncWitriumClient(api_token="your-api-token") as client:
    # Step 1: Secure login with 2FA
    logger.info("Initiating secure banking login...")
    login_response = client.run_workflow(
        workflow_id="bank-login-with-2fa-workflow",
        args={
            "username": "customer123",
            "password": "secure456",
            "phone_number": "+1234567890"  # For 2FA
        },
        keep_session_alive=True  # Keep session for subsequent operations
    )

    # Wait for login and 2FA to complete
    logger.info("Waiting for login and 2FA completion...")
    login_results = client.wait_until_state(
        run_id=login_response.run_id,
        target_status=WorkflowRunStatus.RUNNING,
        all_instructions_executed=True,
        min_wait_time=30  # 2FA usually takes some time
    )
    logger.info("Secure login completed")

    # Step 2: Check account balances
    logger.info("Checking account balances...")
    balance_response = client.run_workflow(
        workflow_id="check-balances-workflow",
        args={"account_types": ["checking", "savings", "credit"]},
        use_existing_session=login_response.run_id
    )

    balance_results = client.wait_until_state(
        run_id=balance_response.run_id,
        target_status=WorkflowRunStatus.COMPLETED
    )
    logger.info(f"Account balances retrieved: {balance_results.result}")

    # Step 3: Download transaction history
    logger.info("Downloading transaction history...")
    transaction_response = client.run_workflow(
        workflow_id="download-transactions-workflow",
        args={
            "date_range": "last_30_days",
            "format": "csv",
            "accounts": ["checking", "savings"]
        },
        use_existing_session=login_response.run_id
    )

    transaction_results = client.wait_until_state(
        run_id=transaction_response.run_id,
        target_status=WorkflowRunStatus.COMPLETED
    )
    logger.info("Transaction history downloaded")

    # Step 4: Generate financial report
    logger.info("Generating financial report...")
    report_response = client.run_workflow(
        workflow_id="generate-financial-report-workflow",
        args={
            "report_type": "monthly_summary",
            "include_charts": True
        },
        use_existing_session=login_response.run_id
    )

    report_results = client.wait_until_state(
        run_id=report_response.run_id,
        target_status=WorkflowRunStatus.COMPLETED
    )

    logger.info("Financial report generated successfully")
    logger.info("All banking operations completed in the same secure session")
```

## Running Talents

In addition to workflows, you can execute "Talents" directly. Talents are pre-defined capabilities or simpler automation units that can be executed with specific arguments.

```python
from witrium import SyncWitriumClient

with SyncWitriumClient(api_token="your-api-token") as client:
    # Run a talent by ID with its execution schema
    result = client.run_talent(
        talent_id="talent-uuid",
        args={"key": "value"},
         # Optional parameters:
         # files=[...],
         # use_states=["state-id"],
         # preserve_state="new-state-name"
    )

    # The result is a TalentResultSchema object
    print(f"Status: {result.status}")
    print(f"Result data: {result.result}")
    if result.error_message:
        print(f"Error: {result.error_message}")
```

## Basic Usage

### Synchronous Client

```python
from witrium import SyncWitriumClient

# Using with context manager (recommended)
with SyncWitriumClient(api_token="your-api-token") as client:
    # Run a workflow and wait for results
    results = client.run_workflow_and_wait(
        workflow_id="workflow-uuid",
        args={"key1": "value1", "key2": 42},
        polling_interval=5,
        timeout=300
    )
    print(f"Workflow completed with status: {results.status}")
    print(f"Results: {results.result}")

    # Or run a workflow without waiting
    response = client.run_workflow(
        workflow_id="workflow-uuid",
        args={"key1": "value1"}
    )
    print(f"Workflow run started: {response.run_id}")

    # Get results later
    results = client.get_workflow_results(
        run_id=response.run_id
    )

    # Wait for workflow to start running
    results = client.wait_until_state(
        run_id=response.run_id,
        target_status=WorkflowRunStatus.RUNNING
    )
    print(f"Workflow is now running: {results.status}")
```

### Asynchronous Client

```python
import asyncio
from witrium import AsyncWitriumClient, WorkflowRunStatus


async def run_workflow():
    # Using with async context manager (recommended)
    async with AsyncWitriumClient(api_token="your-api-token") as client:
        # Run a workflow and wait for results
        results = await client.run_workflow_and_wait(
            workflow_id="workflow-uuid",
            args={"key1": "value1", "key2": 42},
            polling_interval=5,
            timeout=300
        )
        print(f"Workflow completed with status: {results.status}")
        print(f"Results: {results.result}")

        # Or start a workflow and wait for it to begin running
        response = await client.run_workflow(
            workflow_id="workflow-uuid",
            args={"key1": "value1"}
        )

        # Wait until workflow starts running
        results = await client.wait_until_state(
            run_id=response.run_id,
            target_status=WorkflowRunStatus.RUNNING
        )
        print(f"Workflow is now running: {results.status}")


# Run the async function
asyncio.run(run_workflow())
```

## Progress Tracking and Monitoring

### Real-time Progress Tracking

```python
import time
from tqdm import tqdm
from witrium import SyncWitriumClient, WorkflowRunStatus


def create_progress_tracker():
    """Create a progress tracking function."""
    pbar = tqdm(total=100, desc="Workflow Progress")
    last_execution_count = 0

    def update_progress(result):
        nonlocal last_execution_count
        # Get execution count
        executions = result.executions or []
        execution_count = len(executions)

        # Update progress bar only if we have new executions
        if execution_count > last_execution_count:
            pbar.update(execution_count - last_execution_count)
            last_execution_count = execution_count

        # Update description based on status
        pbar.set_description(f"Status: {result.status}")

        # Show individual execution details
        for execution in executions:
            if execution.status == "C":  # Completed
                tqdm.write(f"✅ {execution.instruction}")
            elif execution.status == "F":  # Failed
                tqdm.write(f"❌ {execution.instruction}: {execution.error_message}")

    return update_progress, pbar


with SyncWitriumClient(api_token="your-api-token") as client:
    progress_callback, progress_bar = create_progress_tracker()

    try:
        # Run workflow with progress tracking
        result = client.run_workflow_and_wait(
            workflow_id="workflow-uuid",
            args={"key1": "value1"},
            on_progress=progress_callback
        )
        progress_bar.close()
        print("Workflow completed!")
    except Exception as e:
        progress_bar.close()
        print(f"Workflow failed: {e}")
```

### Using Callbacks for Custom Monitoring

```python
# Define a custom progress callback
def monitor_workflow_progress(result):
    """Custom monitoring function."""
    status = result.status
    executions = result.executions or []
    
    print(f"📊 Status: {status}, Executions: {len(executions)}")
    
    # Log each execution step
    for i, execution in enumerate(executions):
        status_emoji = {
            "P": "⏳",  # Pending
            "R": "🔄",  # Running
            "C": "✅",  # Completed
            "F": "❌",  # Failed
        }.get(execution.status, "❓")
        
        print(f"  {status_emoji} Step {i+1}: {execution.instruction}")
        
        if execution.error_message:
            print(f"    ⚠️  Error: {execution.error_message}")

# Use the callback
with SyncWitriumClient(api_token="your-api-token") as client:
    results = client.run_workflow_and_wait(
        workflow_id="workflow-uuid",
        args={"key1": "value1"},
        on_progress=monitor_workflow_progress
    )
```

## API Reference

### SyncWitriumClient / AsyncWitriumClient

#### Initialization

```python
SyncWitriumClient(
    api_token: str,           # API token for authentication
    timeout: int = 60,        # Request timeout in seconds
    verify_ssl: bool = True   # Whether to verify SSL certificates
)
```

#### Core Methods

##### run_workflow()

Execute a workflow in the Witrium platform.

```python
run_workflow(
    workflow_id: str,                                    # Required: ID of the workflow to run
    args: Optional[Dict[str, Union[str, int, float]]] = None,  # Arguments to pass to the workflow
    use_states: Optional[List[str]] = None,              # List of saved state names to restore
    preserve_state: Optional[str] = None,                # Name to save the browser state as
    no_intelligence: bool = False,                       # Disable AI assistance
    record_session: Optional[bool] = False,              # Record the browser session
    keep_session_alive: bool = False,                    # Keep browser alive after completion
    use_existing_session: Optional[str] = None           # Workflow run ID of existing session to use
) -> WorkflowRunSubmittedSchema
```

**Session Management Parameters:**

- `preserve_state`: Save the browser state with this name after workflow completion. Other workflows can then restore this state using `use_states`.
- `use_states`: List of previously saved state names to restore at the start of this workflow.
- `keep_session_alive`: If True, keeps the browser instance running after workflow completion.
- `use_existing_session`: Run this workflow in an existing browser session (identified by workflow run ID).

##### run_talent()

Run a talent by ID.

```python
run_talent(
    talent_id: str,                      # Required: ID of the talent to run
    args: Optional[Dict[str, Union[str, int, float]]] = None,  # Arguments to pass to the talent
) -> TalentResultSchema
```

##### wait_until_state()

Wait for a workflow run to reach a specific status.

```python
wait_until_state(
    run_id: str,                         # The workflow run ID to wait for
    target_status: str,                  # Target status (e.g., WorkflowRunStatus.RUNNING)
    all_instructions_executed: bool = False,  # Also wait for all executions to complete
    min_wait_time: int = 0,              # Minimum seconds to wait before polling starts
    polling_interval: int = 2,           # Seconds between polling attempts
    timeout: int = 60                    # Maximum seconds to wait
) -> WorkflowRunResultsSchema
```

**Key Parameters:**

- `target_status`: Use `WorkflowRunStatus` constants (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
- `all_instructions_executed`: When True, also waits for all individual execution steps to complete
- `min_wait_time`: Useful for long-running workflows to reduce unnecessary polling

##### Other Methods

- `get_workflow_results(run_id)`: Get current results of a workflow run
- `run_workflow_and_wait(...)`: Run a workflow and poll until completion
- `cancel_run(run_id)`: Cancel a workflow run and clean up associated resources
- `close()`: Close the HTTP client (called automatically with context manager)

### Status Constants

#### WorkflowRunStatus

```python
WorkflowRunStatus.PENDING      # "P" - Workflow is queued
WorkflowRunStatus.RUNNING      # "R" - Workflow is executing
WorkflowRunStatus.COMPLETED    # "C" - Workflow finished successfully
WorkflowRunStatus.FAILED       # "F" - Workflow failed
WorkflowRunStatus.CANCELLED    # "X" - Workflow was cancelled

# Helper lists
WorkflowRunStatus.TERMINAL_STATUSES  # [COMPLETED, FAILED, CANCELLED]
```

#### AgentExecutionStatus

```python
AgentExecutionStatus.PENDING      # "P" - Execution step is queued
AgentExecutionStatus.RUNNING      # "R" - Execution step is running
AgentExecutionStatus.COMPLETED    # "C" - Execution step completed
AgentExecutionStatus.FAILED       # "F" - Execution step failed
AgentExecutionStatus.CANCELLED    # "X" - Execution step cancelled
```

### Response Schemas

#### WorkflowRunSubmittedSchema

```python
{
    "workflow_id": str,
    "run_id": str,  # Use this for polling and session management
    "status": str
}
```

#### WorkflowRunResultsSchema

```python
{
    "workflow_id": str,
    "run_id": str,
    "status": str,
    "started_at": Optional[str],
    "completed_at": Optional[str],
    "message": Optional[str],
    "executions": List[AgentExecutionSchema],  # Individual execution steps
    "result": Optional[dict | list],           # Final workflow result
    "result_format": Optional[str],
    "error_message": Optional[str]
}
```

#### AgentExecutionSchema

```python
{
    "status": str,
    "instruction_order": int,
    "instruction": str,
    "result": Optional[dict | list],
    "result_format": Optional[str],
    "error_message": Optional[str]
}
```

#### TalentExecuteSchema

```python
{
    "args": Optional[dict[str, Any]],
    "files": Optional[List[FileUpload]],
    "use_states": Optional[List[str]],
    "preserve_state": Optional[str]
}
```

### Exception Handling

```python
from witrium import WitriumClientException

try:
    result = client.run_workflow_and_wait(
        workflow_id="my-workflow",
        args={"key": "value"}
    )
except WitriumClientException as e:
    print(f"Witrium API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Cancelling Workflow Runs

You can cancel a workflow run that is in progress:

```python
from witrium import SyncWitriumClient

with SyncWitriumClient(api_token="your-api-token") as client:
    # Start a workflow
    response = client.run_workflow(
        workflow_id="long-running-workflow",
        args={"parameter": "value"}
    )

    # Later, decide to cancel it
    cancel_result = client.cancel_run(run_id=response.run_id)
    print(f"Workflow cancelled with status: {cancel_result.status}")
```

This is particularly useful for:
- Long-running workflows that are no longer needed
- Error recovery scenarios
- Resource management (freeing up browser sessions)
- User-initiated cancellations in interactive applications

## Best Practices

### 1. Always Use Context Managers

```python
# ✅ Good - Automatically closes connections
with SyncWitriumClient(api_token=token) as client:
    results = client.run_workflow(...)

# ❌ Bad - Manual cleanup required
client = SyncWitriumClient(api_token=token)
results = client.run_workflow(...)
client.close()  # Easy to forget!
```

### 2. Choose the Right Session Management Pattern

```python
# ✅ For concurrent operations - use state preservation
for category in categories:
    client.run_workflow(
        workflow_id="scraper",
        args={"category": category},
        use_states=["logged-in-state"]  # Each runs in new browser
    )

# ✅ For sequential operations - use session persistence
login_run_id = client.run_workflow(..., keep_session_alive=True).run_id
client.wait_until_state(..., target_status=WorkflowRunStatus.RUNNING)
client.run_workflow(..., use_existing_session=login_run_id)  # Same browser
```

### 3. Implement Proper Error Handling

```python
def run_workflow_with_retry(client, workflow_id, args, max_retries=3):
    """Run workflow with retry logic."""
    for attempt in range(max_retries):
        try:
            return client.run_workflow_and_wait(
                workflow_id=workflow_id,
                args=args,
                timeout=300
            )
        except WitriumClientException as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff
```

### 4. Use Appropriate Timeouts

```python
# ✅ Adjust timeouts based on workflow complexity
quick_results = client.run_workflow_and_wait(
    workflow_id="simple-data-extraction",
    timeout=60  # Simple workflows
)

complex_results = client.run_workflow_and_wait(
    workflow_id="complex-multi-page-workflow", 
    timeout=600,  # Complex workflows need more time
    min_wait_time=30  # Don't start polling immediately
)
```

### 5. Monitor Progress for Long-Running Workflows

```python
# ✅ Use callbacks for visibility into long-running processes
def log_progress(result):
    completed_steps = sum(1 for ex in result.executions if ex.status == "C")
    total_steps = len(result.executions)
    logger.info(f"Progress: {completed_steps}/{total_steps} steps completed")

client.run_workflow_and_wait(
    workflow_id="long-running-workflow",
    on_progress=log_progress,
    polling_interval=10  # Poll less frequently for long workflows
)
```
