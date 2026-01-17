"""
Test script for browser session management in Witrium SDK.
Fill in the placeholders with real values and run to test against the server.
"""

import asyncio
from witrium import AsyncWitriumClient
from witrium.types import (
    BrowserSessionCreateOptions,
    WorkflowRunOptionsSchema,
)


async def test_browser_sessions():
    """Test browser session management with automatic session handling."""

    # TODO: Fill in your API token
    API_TOKEN = "your-api-token-here"

    # TODO: Fill in your workflow and talent IDs
    WORKFLOW_ID = "your-workflow-id-here"
    TALENT_ID = "your-talent-id-here"

    print("=" * 60)
    print("Testing Browser Session Management")
    print("=" * 60)

    # Test 1: Basic usage with automatic session management
    print("\n[Test 1] Basic usage with automatic session management")
    print("-" * 60)

    async with AsyncWitriumClient(api_token=API_TOKEN) as client:
        print(f"✓ Session created automatically: {client.session_id}")

        # Run workflow - session_id is automatically used
        print(f"\n→ Running workflow: {WORKFLOW_ID}")
        workflow_result = await client.run_workflow(WORKFLOW_ID)
        print(f"✓ Workflow submitted: {workflow_result.run_id}")
        print(f"  Status: {workflow_result.status}")

        # Run talent - session_id is automatically used
        print(f"\n→ Running talent: {TALENT_ID}")
        talent_result = await client.run_talent(TALENT_ID)
        print(f"✓ Talent completed: {talent_result.status}")

        print("\n✓ Session will be closed automatically on exit")

    print("\n✓ Session closed")

    # Test 2: With custom session options (proxy and use_states)
    print("\n[Test 2] Custom session options with proxy and use_states")
    print("-" * 60)

    session_options = BrowserSessionCreateOptions(
        provider="omega",
        use_proxy=True,
        proxy_country="us",
        proxy_city="New York",
        use_states=[
            "session-state-1",
            "session-state-2",
        ],  # These will be used by all runs
    )

    print("Note: use_states set in session options will be used for all runs")
    print("      Individual run options' use_states will be ignored")

    async with AsyncWitriumClient(
        api_token=API_TOKEN, session_options=session_options
    ) as client:
        print(f"✓ Session created with proxy: {client.session_id}")

        # Get session details
        session_details = await client.get_browser_session(client.session_id)
        print(f"  Provider: {session_details.provider}")
        print(f"  Proxy Country: {session_details.proxy_country}")
        print(f"  Proxy City: {session_details.proxy_city}")
        print(f"  Status: {session_details.status}")

        # Run workflow with the proxied session
        print("\n→ Running workflow with proxied session")
        workflow_result = await client.run_workflow(WORKFLOW_ID)
        print(f"✓ Workflow submitted: {workflow_result.run_id}")

    print("\n✓ Proxied session closed")

    # Test 3: Manual session management
    print("\n[Test 3] Manual session management")
    print("-" * 60)

    client = AsyncWitriumClient(api_token=API_TOKEN)

    # Create session manually
    session = await client.create_browser_session()
    print(f"✓ Session created manually: {session.uuid}")
    print(f"  Provider: {session.provider}")
    print(f"  Status: {session.status}")

    # List all sessions
    sessions_list = await client.list_browser_sessions()
    print(f"\n✓ Total active sessions: {sessions_list.total_count}")
    for idx, sess in enumerate(sessions_list.sessions, 1):
        print(f"  {idx}. {sess.uuid} - {sess.status} (busy: {sess.is_busy})")

    # Run workflow with explicit session_id
    print("\n→ Running workflow with explicit session_id")
    workflow_result = await client.run_workflow(
        WORKFLOW_ID, options=WorkflowRunOptionsSchema(browser_session_id=session.uuid)
    )
    print(f"✓ Workflow submitted: {workflow_result.run_id}")

    # Close session manually
    print("\n→ Closing session manually")
    close_result = await client.close_browser_session(session.uuid)
    print(f"✓ {close_result.message}")

    await client.close()

    # Test 4: Override automatic session with explicit session_id
    print("\n[Test 4] Override automatic session with explicit session_id")
    print("-" * 60)

    # Create a session outside the context manager
    temp_client = AsyncWitriumClient(api_token=API_TOKEN)
    external_session = await temp_client.create_browser_session()
    print(f"✓ External session created: {external_session.uuid}")

    # Use context manager but override with external session
    async with AsyncWitriumClient(api_token=API_TOKEN) as client:
        print(f"✓ Context manager session: {client.session_id}")

        # Override with external session
        print("\n→ Running workflow with external session (override)")
        workflow_result = await client.run_workflow(
            WORKFLOW_ID,
            options=WorkflowRunOptionsSchema(browser_session_id=external_session.uuid),
        )
        print(f"✓ Workflow submitted using external session: {workflow_result.run_id}")

    # Clean up external session
    await temp_client.close_browser_session(external_session.uuid)
    print(f"✓ External session closed: {external_session.uuid}")
    await temp_client.close()

    # Test 5: Demonstrating use_states behavior
    print("\n[Test 5] use_states behavior with browser sessions")
    print("-" * 60)

    # Create session with specific use_states
    session_with_states = BrowserSessionCreateOptions(use_states=["state-from-session"])

    async with AsyncWitriumClient(
        api_token=API_TOKEN, session_options=session_with_states
    ) as client:
        print("✓ Session created with use_states: ['state-from-session']")

        # Try to pass different use_states in run options - it will be ignored
        print("\n→ Running workflow with different use_states in options")
        print("   (these will be IGNORED, session's use_states will be used)")
        workflow_result = await client.run_workflow(
            WORKFLOW_ID,
            options=WorkflowRunOptionsSchema(
                use_states=["this-will-be-ignored"]  # This is ignored!
            ),
        )
        print(f"✓ Workflow submitted: {workflow_result.run_id}")
        print("   The session's use_states ['state-from-session'] was used")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


async def test_error_handling():
    """Test error handling scenarios."""

    API_TOKEN = "your-api-token-here"

    print("\n[Bonus] Error handling test")
    print("-" * 60)

    client = AsyncWitriumClient(api_token=API_TOKEN)

    try:
        # Try to get a non-existent session
        print("→ Testing error handling with invalid session UUID")
        await client.get_browser_session("invalid-uuid-12345")
    except Exception as e:
        print(f"✓ Error handled gracefully: {str(e)}")

    await client.close()


if __name__ == "__main__":
    print("\n" + "🚀 " * 20)
    print("\nWitrium Browser Session Management Test Suite")
    print("\n" + "🚀 " * 20 + "\n")

    asyncio.run(test_browser_sessions())

    # Uncomment to test error handling
    # asyncio.run(test_error_handling())
