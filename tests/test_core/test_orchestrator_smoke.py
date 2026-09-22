import asyncio
from core.orchestrator import RENIXAction, RENIXOrchestrator


def test_action_success_and_failure_propagate():
    async def ok(_action):
        return {"success": True, "message": "ok"}

    async def run():
        o = RENIXOrchestrator()
        o.register_action("test_ok", ok)
        good = await o.execute_action(RENIXAction(action_id="1", action_type="test_ok"))
        assert good.success is True

        o.register_action("test_fail", lambda _action: {"error": "boom", "message": "failed"})
        bad = await o.execute_action(RENIXAction(action_id="2", action_type="test_fail"))
        assert bad.success is False
        assert bad.error == "boom"

    asyncio.run(run())
