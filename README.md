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
from witrium import SyncWitriumClient, WorkflowRunOptionsSchema

# 1. Provide your API token (export as env-var in production)
api_token = "YOUR_WITRIUM_API_TOKEN"  # Obtain from dashboard

# 2. Use context manager for automatic browser session management
with SyncWitriumClient(api_token=api_token) as client:
    # Browser session is automatically created
    # client.session_id contains the browser session ID
    
    # 3. Run workflows
    login = client.run_workflow(
        workflow_id="login-workflow-id",
        options=WorkflowRunOptionsSchema(
            args={"username": "user@example.com", "password": "secretPass!"}
        )
    )
    
    # 4. Run another workflow in the same browser session
    scrape = client.run_workflow_and_wait(
        workflow_id="dashboard-scrape-workflow-id",
        options=WorkflowRunOptionsSchema(args={"section": "sales"})
    )
    
    # 5. Browser session is automatically closed when exiting context
    print("Sales data:", scrape.result)
```

---

## Browser Session Management

Witrium SDK provides context managers that automatically handle browser lifecycle for you. This is the **recommended approach** for running workflows and/or talents that need to share browser state.

### Automatic Session Management (Recommended)

When you use the context manager, a browser session is automatically created when entering and closed when exiting:

```python
from witrium import AsyncWitriumClient

async with AsyncWitriumClient(api_token="...") as client:
    # Browser session automatically created
    print(f"Session ID: {client.session_id}")
    
    # All workflows automatically use this session
    result1 = await client.run_workflow("workflow-1")
    result2 = await client.run_talent("talent-1")
    
    # Session automatically closed on exit
```

### Custom Session Options

Configure the browser session with specific settings (These are options similar to the ones you'll find in the dashboard):

```python
from witrium import AsyncWitriumClient, BrowserSessionCreateOptions

session_opts = BrowserSessionCreateOptions(
    provider="omega",
    use_proxy=True,
    proxy_country="uk",
    use_states=["my-saved-state"],  # Restore browser state
    preserve_state="updated-state"  # Save state when session closes
)

async with AsyncWitriumClient(
    api_token="...",
    session_options=session_opts
) as client:
    # Browser session created with UK proxy and restored state
    result = await client.run_workflow("workflow-id")
    # State will be automatically saved as "updated-state" when exiting context
```

### Manual Session Management

For advanced use cases, manage browser sessions explicitly:

```python
from witrium import (
    AsyncWitriumClient,
    BrowserSessionCreateOptions,
    BrowserSessionCloseOptions,
    WorkflowRunOptionsSchema
)

client = AsyncWitriumClient(api_token="...")

# Create a browser session
session = await client.create_browser_session(
    BrowserSessionCreateOptions(use_proxy=True)
)
print(f"Created session: {session.uuid}")

# List all active sessions
sessions = await client.list_browser_sessions()
print(f"Active sessions: {sessions.total_count}")

# Get session details
details = await client.get_browser_session(session.uuid)
print(f"Session status: {details.status}")

# Use session explicitly
result = await client.run_workflow(
    "workflow-id",
    options=WorkflowRunOptionsSchema(browser_session_id=session.uuid)
)

# Close session when done (optionally preserve state). This will override the state name passed at creation time
await client.close_browser_session(
    session.uuid,
    options=BrowserSessionCloseOptions(preserve_state="my-saved-state")
)
await client.close()
```

### Important: `use_states` Behavior

**When using a browser session, `use_states` is set at the session level, not the individual run level.**

```python
# use_states in session options applies to ALL runs
session_opts = BrowserSessionCreateOptions(
    use_states=["state-from-session"]  # ✅ This will be used
)

async with AsyncWitriumClient(api_token="...", session_options=session_opts) as client:
    result = await client.run_workflow(
        "workflow-id",
        options=WorkflowRunOptionsSchema(
            use_states=["ignored"]  # ❌ This is IGNORED when using a session
        )
    )
```

If you need different `use_states` for different runs, create separate browser sessions.

### Available Browser Session Methods

- `create_browser_session(options)` - Create a new browser session
- `list_browser_sessions()` - List all active sessions
- `get_browser_session(session_uuid)` - Get session details
- `close_browser_session(session_uuid, options)` - Close a session

---

## Workflow Lifecycle & Polling Essentials

`client.run_workflow(...)` **only submits** a job – the real browser work happens asynchronously in the cloud.  Keep these steps in mind whenever you design multi-step automations:

1. **Submit** – your call returns instantly with a `run_id`.
2. **Poll / Wait** – use `wait_until_state()` (or `run_workflow_and_wait()`) to block until the run reaches:
   • `WorkflowRunStatus.RUNNING` – the browser has spun-up and is ready (handy when you enabled `keep_session_alive`).  
   • `WorkflowRunStatus.COMPLETED` – the workflow has finished executing.
3. **Chain or Fetch Results** – once the target state is reached you can either run another workflow (chaining sessions) or read the data via `get_workflow_results()`.

### When to wait for which state?

| Scenario | Recommended `target_status` |
|----------|-----------------------------|
| You want to wait till the workflow run has completed | `COMPLETED` |
| You want to wait till the workflow has just started its run | `RUNNING` |

### Parallel vs. Serial Execution

• **Parallel Execution** – You can run the same workflow or talent in different context managers and each will spin up a different browser for true parallelism.  
• **Serial Execution** – All workflows in the context manager share **one** browser session. They are to be run serially.

---

## Common Use Cases and Session Management

### The Authentication Challenge

A common pattern in web automation involves authentication: you need to log into a service first, then perform actions in the authenticated session. Witrium provides several approaches:

#### Approach 1: State Preservation (Parallel-Friendly)
- **Best for:** Running multiple post-login automations concurrently
- **How it works:** Save browser state after login, then restore it in new browser instances
- **Advantages:** Horizontal scaling, isolation between runs

#### Approach 2: Custom Session Management (Advanced)
- **Best for:** Complex workflows requiring fine-grained session control
- **How it works:** Manually create, manage, and close browser sessions
- **Advantages:** Full control over session lifecycle

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
from witrium import SyncWitriumClient, WorkflowRunStatus, WorkflowRunOptionsSchema

with SyncWitriumClient(api_token="your-api-token") as client:
    # Step 1: Run login workflow and preserve the authenticated state
    login_response = client.run_workflow(
        workflow_id="login-workflow-id",
        options=WorkflowRunOptionsSchema(
            args={"username": "user@example.com", "password": "secure123"},
            preserve_state="authenticated-session"  # Save state with this name
        )
    )

    # Step 2: Wait for login to complete
    login_results = client.wait_until_state(
        run_id=login_response.run_id,
        target_status=WorkflowRunStatus.COMPLETED
    )

# Step 3: Run multiple post-login workflows concurrently
# Each will spawn a new browser but restore the authenticated state

# Workflow A: Extract data from dashboard
with SyncWitriumClient(api_token="your-api-token") as client:
    dashboard_response = client.run_workflow(
        workflow_id="dashboard-scraping-workflow-id",
        options=WorkflowRunOptionsSchema(
            args={"report_type": "monthly"},
            use_states=["authenticated-session"]  # Restore the saved state
        )
    )

# Workflow B: Update user profile (can run concurrently)
with SyncWitriumClient(api_token="your-api-token") as client:
    profile_response = client.run_workflow(
        workflow_id="profile-update-workflow-id",
        options=WorkflowRunOptionsSchema(
            args={"new_email": "newemail@example.com"},
            use_states=["authenticated-session"]  # Same state, different browser instance
        )
    )

# Both workflows are now running concurrently in separate browser instances
# but both have access to the authenticated session
```

### Pattern 2: Shared Browser Session (Recommended)

This approach uses same browser instance across workflows. The browser session is created when entering and closed when exiting the context manager.

**Advantages:**
- Simple and clean API
- Automatic resource cleanup
- Resource-efficient (reuses same browser instance)
- No manual session lifecycle management

**Use Case Example:**

```python
from witrium import SyncWitriumClient, WorkflowRunOptionsSchema

with SyncWitriumClient(api_token="your-api-token") as client:
    # Browser session automatically created
    # client.session_id contains the session UUID

    # Step 1: Run login workflow (navigates to login page and authenticates)
    login_response = client.run_workflow_and_wait(
        workflow_id="login-workflow-id",
        options=WorkflowRunOptionsSchema(
            args={"username": "user@example.com", "password": "secure123"}
        )
    )
    print("Login completed")

    # Step 2: Run subsequent workflows - they automatically use the same session
    # Use skip_goto_url_instruction=True since the browser is already on the right page
    
    # Workflow A: Extract data from dashboard
    dashboard_results = client.run_workflow_and_wait(
        workflow_id="dashboard-scraping-workflow-id",
        options=WorkflowRunOptionsSchema(
            args={"report_type": "monthly"},
            skip_goto_url_instruction=True  # Already on the dashboard after login
        )
    )
    print(f"Dashboard data: {dashboard_results.result}")

    # Workflow B: Update user profile
    profile_results = client.run_workflow_and_wait(
        workflow_id="profile-update-workflow-id",
        options=WorkflowRunOptionsSchema(
            args={"new_email": "newemail@example.com"},
            skip_goto_url_instruction=True  # Previous workflow left us on the right page
        )
    )
    print("Profile updated")

    # Browser session automatically closed on exit
```

### Choosing the Right Pattern

| Factor | State Preservation | Shared Browser Session |
|--------|-------------------|------------------------|
| **Concurrency** | ✅ Multiple workflows can run simultaneously | ❌ Must run serially |
| **Resource Usage** | Higher (multiple browser instances) | ✅ Lower (single browser instance) |
| **Isolation** | ✅ Complete isolation between workflows | ❌ Shared session state |
| **Setup Complexity** | Medium (manage state names) | ✅ Simple (just workflow run IDs) |
| **Use Case** | Bulk data processing, parallel operations | Sequential workflows, state-dependent operations |

## Complete Examples

### Example 1: E-commerce Data Extraction (State Preservation)

```python
from witrium import SyncWitriumClient, WorkflowRunStatus, WorkflowRunOptionsSchema
import concurrent.futures
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_category_data(category, state_name):
    """Extract data for a specific product category."""
    with SyncWitriumClient(api_token="your-api-token") as client:
        try:
            response = client.run_workflow(
                workflow_id="category-scraper-workflow",
                options=WorkflowRunOptionsSchema(
                    args={"category": category},
                    use_states=[state_name]
                )
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
        options=WorkflowRunOptionsSchema(
            args={"email": "seller@example.com", "password": "secure123"},
            preserve_state="ecommerce-authenticated"
        )
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
        executor.submit(extract_category_data, category,
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

### Example 2: Banking Workflow (Shared Browser Session)

```python
from witrium import SyncWitriumClient, WorkflowRunOptionsSchema
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with SyncWitriumClient(api_token="your-api-token") as client:
    # Browser session automatically created
    logger.info(f"Browser session created: {client.session_id}")
    
    # Step 1: Secure login with 2FA
    logger.info("Initiating secure banking login...")
    login_results = client.run_workflow_and_wait(
        workflow_id="bank-login-with-2fa-workflow",
        options=WorkflowRunOptionsSchema(
            args={
                "username": "customer123",
                "password": "secure456",
                "phone_number": "+1234567890"  # For 2FA
            }
        )
    )
    logger.info("Secure login completed")

    # Step 2: Check account balances
    # Skip initial navigation - login workflow already brought us to the dashboard
    logger.info("Checking account balances...")
    balance_results = client.run_workflow_and_wait(
        workflow_id="check-balances-workflow",
        options=WorkflowRunOptionsSchema(
            args={"account_types": ["checking", "savings", "credit"]},
            skip_goto_url_instruction=True
        )
    )
    logger.info(f"Account balances retrieved: {balance_results.result}")

    # Step 3: Download transaction history
    logger.info("Downloading transaction history...")
    transaction_results = client.run_workflow_and_wait(
        workflow_id="download-transactions-workflow",
        options=WorkflowRunOptionsSchema(
            args={
                "date_range": "last_30_days",
                "format": "csv",
                "accounts": ["checking", "savings"]
            },
            skip_goto_url_instruction=True
        )
    )
    logger.info("Transaction history downloaded")

    # Step 4: Generate financial report
    logger.info("Generating financial report...")
    report_results = client.run_workflow_and_wait(
        workflow_id="generate-financial-report-workflow",
        options=WorkflowRunOptionsSchema(
            args={
                "report_type": "monthly_summary",
                "include_charts": True
            },
            skip_goto_url_instruction=True
        )
    )

    logger.info("Financial report generated successfully")
    logger.info("All banking operations completed in the same secure session")
    # Browser session automatically closed on exit
```

## Running Talents

In addition to workflows, you can execute "Talents" directly. Talents are pre-defined capabilities or simpler automation units that can be executed with specific arguments. When using a context manager, talents automatically use the managed browser session.

```python
from witrium import SyncWitriumClient, TalentRunOptionsSchema

with SyncWitriumClient(api_token="your-api-token") as client:
    # Browser session automatically created and used
    
    # Run a talent by ID with options
    result = client.run_talent(
        talent_id="talent-uuid",
        options=TalentRunOptionsSchema(
            args={"key": "value"}
            # browser_session_id is automatically set to client.session_id
        )
    )

    # The result is a TalentResultSchema object
    print(f"Status: {result.status}")
    print(f"Result data: {result.result}")
    if result.error_message:
        print(f"Error: {result.error_message}")
    # Browser session automatically closed and cleaned up
```

### Combining Workflows and Talents in the Same Session

A common use case is running both workflows and talents in the same browser session context. This allows you to chain a workflow (e.g., login or navigation) with talent execution that operates on the resulting browser state.

```python
import asyncio
from witrium import AsyncWitriumClient, WorkflowRunOptionsSchema, TalentRunOptionsSchema

async def main():
    async with AsyncWitriumClient(api_token="your-api-token") as client:
        # Browser session automatically created
        print(f"Session ID: {client.session_id}")

        # Step 1: Run a workflow to set up the browser state (e.g., login, navigate)
        print("Running setup workflow...")
        workflow_result = await client.run_workflow_and_wait(
            workflow_id="setup-workflow-id",
            options=WorkflowRunOptionsSchema(
                args={"username": "user@example.com", "password": "secure123"}
            )
        )
        print(f"Workflow completed: {workflow_result.status}")

        # Step 2: Run a talent in the same browser session
        # The talent will operate on the browser state left by the workflow
        print("Running talent...")
        talent_result = await client.run_talent(
            talent_id="data-extraction-talent-id",
            options=TalentRunOptionsSchema(
                args={"product_id": "ABC123", "include_reviews": True}
            )
        )
        print(f"Talent result: {talent_result.result}")

        # Step 3: Run another workflow using the same session
        # Use skip_goto_url_instruction since we're already on the relevant page
        print("Running follow-up workflow...")
        followup_result = await client.run_workflow_and_wait(
            workflow_id="cleanup-workflow-id",
            options=WorkflowRunOptionsSchema(
                skip_goto_url_instruction=True  # Browser is already where we need it
            )
        )
        print(f"Follow-up completed: {followup_result.status}")

        # Check session details at any point
        session = await client.get_browser_session(client.session_id)
        print(f"Session status: {session.status}, Busy: {session.is_busy}")

    # Browser session automatically closed on exit
    print("Session closed")


asyncio.run(main())
```

This pattern is useful when:
- A workflow handles complex multi-step setup (login, navigation, form filling)
- A talent extracts specific data or performs a focused action
- You need to chain multiple operations that depend on shared browser state

## Basic Usage

### Synchronous Client

```python
from witrium import (
    SyncWitriumClient,
    WorkflowRunStatus,
    WorkflowRunOptionsSchema, 
    RunWorkflowAndWaitOptionsSchema,
    WaitUntilStateOptionsSchema,
    BrowserSessionCloseOptions,
)

# Using with context manager (recommended)
with SyncWitriumClient(api_token="your-api-token") as client:
    # Run a workflow and wait for results
    results = client.run_workflow_and_wait(
        workflow_id="workflow-uuid",
        options=RunWorkflowAndWaitOptionsSchema(
            args={"key1": "value1", "key2": 42},
            polling_interval=5,
            timeout=300
        )
    )
    print(f"Workflow completed with status: {results.status}")
    print(f"Results: {results.result}")

    # Or run a workflow without waiting
    response = client.run_workflow(
        workflow_id="workflow-uuid",
        options=WorkflowRunOptionsSchema(
            args={"key1": "value1"}
        )
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
from witrium import (
    AsyncWitriumClient,
    WorkflowRunStatus,
    WorkflowRunOptionsSchema,
    RunWorkflowAndWaitOptionsSchema,
)


async def run_workflow():
    # Using with async context manager (recommended)
    async with AsyncWitriumClient(api_token="your-api-token") as client:
        # Run a workflow and wait for results
        results = await client.run_workflow_and_wait(
            workflow_id="workflow-uuid",
            options=RunWorkflowAndWaitOptionsSchema(
                args={"key1": "value1", "key2": 42},
                polling_interval=5,
                timeout=300
            )
        )
        print(f"Workflow completed with status: {results.status}")
        print(f"Results: {results.result}")

        # Or start a workflow and wait for it to begin running
        response = await client.run_workflow(
            workflow_id="workflow-uuid",
            options=WorkflowRunOptionsSchema(
                args={"key1": "value1"}
            )
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
from witrium import SyncWitriumClient, WorkflowRunStatus, RunWorkflowAndWaitOptionsSchema


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
            options=RunWorkflowAndWaitOptionsSchema(
                args={"key1": "value1"},
                on_progress=progress_callback
            )
        )
        progress_bar.close()
        print("Workflow completed!")
    except Exception as e:
        progress_bar.close()
        print(f"Workflow failed: {e}")
```

### Using Callbacks for Custom Monitoring

```python
from witrium import RunWorkflowAndWaitOptionsSchema

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
        options=RunWorkflowAndWaitOptionsSchema(
            args={"key1": "value1"},
            on_progress=monitor_workflow_progress
        )
    )
```

## API Reference

### SyncWitriumClient / AsyncWitriumClient

#### Initialization

```python
SyncWitriumClient(
    api_token: str,                                        # API token for authentication
    timeout: Optional[float] = None,                       # HTTP request timeout (None = infinite)
    verify_ssl: bool = True,                               # Whether to verify SSL certificates
    session_options: Optional[BrowserSessionCreateOptions] = None  # Browser session options
)
```

**Timeout Behavior:**
- `timeout=None` (default): No timeout - requests never time out
- `timeout=30.0`: Individual HTTP requests timeout after 30 seconds
- Applies to all HTTP requests (run_workflow, get_results, etc.)

**Session Options:**
- When using context manager (`with client:`), a browser session is automatically created
- `session_options` configures the auto-created session (proxy, provider, use_states)
- `client.session_id` contains the active session UUID

#### Core Methods

##### run_workflow()

Execute a workflow in the Witrium platform.

```python
run_workflow(
    workflow_id: str,                                     # Required: ID of the workflow to run
    options: Optional[WorkflowRunOptionsSchema] = None    # Optional workflow run options
) -> WorkflowRunSubmittedSchema
```

**WorkflowRunOptionsSchema fields:**

- `args`: Optional[dict[str, str | int | float]] - Arguments to pass to the workflow
- `files`: Optional[List[FileUpload]] - Files to upload with the workflow
- `use_states`: Optional[List[str]] - List of saved state names to restore (ignored if browser_session_id is set)
- `preserve_state`: Optional[str] - Name to save the browser state as
- `no_intelligence`: bool = False - Disable AI assistance
- `record_session`: bool = False - Record the browser session
- `browser_session_id`: Optional[str] - Browser session UUID to use (auto-set by context manager)
- `skip_goto_url_instruction`: bool = False - Skip the initial URL navigation step (useful when chaining workflows in the same browser session where a previous workflow has already navigated to the relevant page)

**Session Management:**

- `browser_session_id`: UUID of the browser session to use. When using context manager, this is automatically set to `client.session_id`.
- `preserve_state`: Save the browser state with this name after workflow completion. Other workflows can restore this state using `use_states`.
- `use_states`: List of previously saved state names to restore. **Note:** This is ignored if `browser_session_id` is provided - the session's use_states takes precedence.

##### run_talent()

Run a talent by ID.

```python
run_talent(
    talent_id: str,                            # Required: ID of the talent to run
    options: Optional[TalentRunOptionsSchema] = None  # Optional talent run options
) -> TalentResultSchema
```

**TalentRunOptionsSchema fields:**

- `args`: Optional[dict[str, Any]] - Arguments to pass to the talent
- `files`: Optional[List[FileUpload]] - Files to upload with the talent
- `use_states`: Optional[List[str]] - List of saved state names to restore (ignored if browser_session_id is set)
- `preserve_state`: Optional[str] - Name to save the browser state as
- `browser_session_id`: Optional[str] - Browser session UUID to use (auto-set by context manager)

##### wait_until_state()

Wait for a workflow run to reach a specific status.

```python
wait_until_state(
    run_id: str,                                         # The workflow run ID to wait for
    target_status: str,                                  # Target status (e.g., WorkflowRunStatus.RUNNING)
    options: Optional[WaitUntilStateOptionsSchema] = None  # Optional wait options
) -> WorkflowRunResultsSchema
```

**WaitUntilStateOptionsSchema fields:**

- `all_instructions_executed`: bool = False - Also wait for all executions to complete
- `min_wait_time`: int = 0 - Minimum seconds to wait before polling starts
- `polling_interval`: int = 2 - Seconds between polling attempts
- `timeout`: Optional[float] = None - Maximum seconds to wait (None = poll forever)

**Key Parameters:**

- `target_status`: Use `WorkflowRunStatus` constants (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
- `all_instructions_executed`: When True, also waits for all individual execution steps to complete
- `min_wait_time`: Useful for long-running workflows to reduce unnecessary polling
- `timeout`: Set to `None` (default) to poll forever until workflow reaches target status, or specify a timeout in seconds

##### run_workflow_and_wait()

Run a workflow and wait for results by polling until completion.

```python
run_workflow_and_wait(
    workflow_id: str,                                           # Required: ID of the workflow to run
    options: Optional[RunWorkflowAndWaitOptionsSchema] = None   # Optional run and wait options
) -> Union[WorkflowRunResultSchema, List[WorkflowRunResultSchema]]
```

**RunWorkflowAndWaitOptionsSchema fields:**

Inherits all fields from `WorkflowRunOptionsSchema`, plus:

- `polling_interval`: int = 5 - Seconds to wait between polling attempts
- `timeout`: Optional[float] = None - Maximum seconds to poll (None = poll forever until completion)
- `return_intermediate_results`: bool = False - If True, returns list of all polled results
- `on_progress`: Optional[Callable] - Callback function called with each intermediate result

**Timeout Behavior:**
- `timeout=None` (default): Polls indefinitely until workflow completes (reaches terminal status)
- `timeout=300.0`: Raises exception if workflow doesn't complete within 300 seconds
- Different from HTTP timeout - this controls how long to wait for workflow completion

##### Browser Session Methods

- `create_browser_session(options)`: Create a standalone browser session
- `list_browser_sessions()`: List all active browser sessions
- `get_browser_session(session_uuid)`: Get details of a specific browser session
- `close_browser_session(session_uuid, options)`: Close a browser session

**BrowserSessionCreateOptions fields:**
- `provider`: str = "omega" - Browser provider
- `use_proxy`: bool = False - Enable proxy
- `proxy_country`: str = "us" - Proxy country code
- `proxy_city`: str = "New York" - Proxy city
- `use_states`: Optional[List[str]] = None - States to restore (applies to all runs using this session)
- `preserve_state`: Optional[str] = None - Name to save the browser state as when the session is closed

**BrowserSessionCloseOptions fields:**
- `force`: bool = False - Force close the session even if it's busy
- `preserve_state`: Optional[str] = None - Name to save the browser state as before closing. This will override the state name passed at creation time

##### Other Methods

- `get_workflow_results(run_id)`: Get current results of a workflow run
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

#### Option Schemas

##### WorkflowRunOptionsSchema

```python
{
    "args": Optional[dict[str, str | int | float]],
    "files": Optional[List[FileUpload]],
    "use_states": Optional[List[str]],  # Ignored if browser_session_id is set
    "preserve_state": Optional[str],
    "no_intelligence": bool = False,
    "record_session": bool = False,
    "browser_session_id": Optional[str],  # Auto-set by context manager
    "skip_goto_url_instruction": bool = False  # Skip initial navigation when chaining workflows
}
```

##### TalentRunOptionsSchema

```python
{
    "args": Optional[dict[str, Any]],
    "files": Optional[List[FileUpload]],
    "use_states": Optional[List[str]],  # Ignored if browser_session_id is set
    "preserve_state": Optional[str],
    "browser_session_id": Optional[str]  # Auto-set by context manager
}
```

##### WaitUntilStateOptionsSchema

```python
{
    "all_instructions_executed": bool = False,
    "min_wait_time": int = 0,
    "polling_interval": int = 2,
    "timeout": Optional[float] = None  # None = poll forever
}
```

##### RunWorkflowAndWaitOptionsSchema

Extends WorkflowRunOptionsSchema with additional fields:

```python
{
    # All WorkflowRunOptionsSchema fields, plus:
    "polling_interval": int = 5,
    "timeout": Optional[float] = None,  # None = poll forever
    "return_intermediate_results": bool = False,
    "on_progress": Optional[Callable]
}
```

##### BrowserSessionCreateOptions

```python
{
    "provider": str = "omega",
    "use_proxy": bool = False,
    "proxy_country": str = "us",
    "proxy_city": str = "New York",
    "use_states": Optional[List[str]] = None,  # Applies to all runs using this session
    "preserve_state": Optional[str] = None  # Save browser state with this name when session closes
}
```

##### BrowserSessionSchema

```python
{
    "uuid": str,
    "provider": str,
    "status": str,  # "active" or "closed"
    "is_busy": bool,
    "user_managed": bool,
    "current_run_type": Optional[str],  # "workflow", "talent", or None
    "current_run_id": Optional[str],
    "created_at": str,
    "started_at": Optional[str],
    "last_activity_at": Optional[str],
    "proxy_country": Optional[str],
    "proxy_city": Optional[str]
}
```

##### ListBrowserSessionSchema

```python
{
    "sessions": List[BrowserSessionSchema],
    "total_count": int
}
```

##### BrowserSessionCloseOptions

```python
{
    "force": bool = False,  # Force close even if session is busy
    "preserve_state": Optional[str] = None  # Save browser state with this name before closing
}
```

### Exception Handling

```python
from witrium import WitriumClientException, RunWorkflowAndWaitOptionsSchema

try:
    result = client.run_workflow_and_wait(
        workflow_id="my-workflow",
        options=RunWorkflowAndWaitOptionsSchema(
            args={"key": "value"}
        )
    )
except WitriumClientException as e:
    print(f"Witrium API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Cancelling Workflow Runs

You can cancel a workflow run that is in progress:

```python
from witrium import SyncWitriumClient, WorkflowRunOptionsSchema

with SyncWitriumClient(api_token="your-api-token") as client:
    # Start a workflow
    response = client.run_workflow(
        workflow_id="long-running-workflow",
        options=WorkflowRunOptionsSchema(
            args={"parameter": "value"}
        )
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
from witrium import AsyncWitriumClient, WorkflowRunOptionsSchema

# ✅ For most use cases - use a shared browser session (recommended)
async with AsyncWitriumClient(api_token="...") as client:
    # All workflows automatically share the same browser session
    result1 = await client.run_workflow_and_wait("login-workflow")
    result2 = await client.run_workflow_and_wait("scrape-workflow")
    # Session automatically cleaned up

# ✅ For concurrent operations - use state preservation
    for category in categories:
        async with AsyncWitriumClient(api_token="...") as client:
            await client.run_workflow(
                workflow_id="scraper",
                options=WorkflowRunOptionsSchema(
                    args={"category": category},
                    use_states=["logged-in-state"]  # Each runs in new browser
                )
            )
```

### 3. Implement Proper Error Handling

```python
from witrium import RunWorkflowAndWaitOptionsSchema, WitriumClientException

def run_workflow_with_retry(client, workflow_id, args, max_retries=3):
    """Run workflow with retry logic."""
    for attempt in range(max_retries):
        try:
            return client.run_workflow_and_wait(
                workflow_id=workflow_id,
                options=RunWorkflowAndWaitOptionsSchema(
                    args=args,
                    timeout=300
                )
            )
        except WitriumClientException as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff
```

### 4. Use Appropriate Timeouts

```python
from witrium import AsyncWitriumClient, RunWorkflowAndWaitOptionsSchema

# ✅ Default: No timeout (polls until completion)
async with AsyncWitriumClient(api_token="...") as client:
    result = await client.run_workflow_and_wait("workflow-id")
    # Polls forever until workflow completes

# ✅ Set timeout for workflows that should fail fast
async with AsyncWitriumClient(api_token="...", timeout=30.0) as client:
    result = await client.run_workflow_and_wait(
        "simple-workflow",
        options=RunWorkflowAndWaitOptionsSchema(timeout=60.0)  # Max 60s wait
    )

# ✅ Separate HTTP timeout from polling timeout
async with AsyncWitriumClient(api_token="...", timeout=10.0) as client:
    # Each HTTP request times out at 10s
    result = await client.run_workflow_and_wait(
        "complex-workflow",
        options=RunWorkflowAndWaitOptionsSchema(timeout=None)  # Poll forever
    )
```

### 5. Monitor Progress for Long-Running Workflows

```python
from witrium import AsyncWitriumClient, RunWorkflowAndWaitOptionsSchema

# ✅ Use callbacks for visibility into long-running processes
def log_progress(result):
    completed_steps = sum(1 for ex in result.executions if ex.status == "C")
    total_steps = len(result.executions)
    logger.info(f"Progress: {completed_steps}/{total_steps} steps completed")

async with AsyncWitriumClient(api_token="...") as client:
    result = await client.run_workflow_and_wait(
        workflow_id="long-running-workflow",
        options=RunWorkflowAndWaitOptionsSchema(
            on_progress=log_progress,
            polling_interval=10,  # Poll less frequently
            timeout=None  # Poll forever until completion
        )
    )
```
