## witrium

Python client for the Witrium cloud browser automation API.

### Installation

```bash
pip install witrium
```

### Quick start

```python
from witrium import SyncWitriumClient, WorkflowRunStatus

with SyncWitriumClient(api_token="your-api-token") as client:
    run = client.run_workflow(
        workflow_id="workflow-uuid",
        args={"key": "value"}
    )
    result = client.wait_until_state(
        run_id=run.run_id,
        target_status=WorkflowRunStatus.COMPLETED
    )
    print(result.result)
```

For full documentation and advanced patterns, see `witrium/README.md` in this repository.


