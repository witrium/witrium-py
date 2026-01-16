"""
Quick test script for browser session management.
Simple version - just fill in the values and run.
"""

import asyncio
from witrium import (
    AsyncWitriumClient,
    # BrowserSessionCreateOptions,
    TalentRunOptionsSchema,
)


async def main():
    # TODO: Fill these in
    API_TOKEN = "<API_TOKEN>"
    WORKFLOW_ID = "<WORKFLOW_ID>"
    TALENT_ID = "<TALENT_ID>"

    print("\n🚀 Testing Browser Session Management\n")

    # Test with automatic session management
    async with AsyncWitriumClient(
        api_token=API_TOKEN,
        # session_options=BrowserSessionCreateOptions(preserve_state="my-saved-state"),
    ) as client:
        print(f"Session ID: {client.session_id}\n")

        # Run workflow - session_id is automatically used
        print(f"Running workflow: {WORKFLOW_ID}")
        result = await client.run_workflow_and_wait(WORKFLOW_ID)
        print(f"  ✓ Workflow run_id: {result.run_id}")
        print(f"  ✓ Status: {result.status}\n")

        # Run talent - session_id is automatically used
        print(f"Running talent: {TALENT_ID}")
        result2 = await client.run_talent(
            TALENT_ID, options=TalentRunOptionsSchema(args={"asin": "B08QZMJBFR"})
        )
        print(f"  ✓ Status: {result2}\n")

        # Check session details
        session = await client.get_browser_session(client.session_id)
        print("Session Details:")
        print(f"  Status: {session.status}")
        print(f"  Busy: {session.is_busy}")
        print(f"  Provider: {session.provider}\n")

    print("✓ Session automatically closed on exit\n")


if __name__ == "__main__":
    asyncio.run(main())
