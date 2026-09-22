"""
RENIX AI - Cricket Agent

Handles cricket-related assistance.

Responsibilities:
- Batting analysis
- Bowling analysis
- Fielding analysis
- Match statistics
- Cricket training
- Shot analysis
- Performance tracking
- Match preparation
- Cricket questions and advice
- Integration with the RENIX sports/cricket subsystem
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .base_agent import (
    AgentCapability,
    AgentPriority,
    AgentResult,
    AgentTask,
    BaseAgent,
)


class CricketAgent(BaseAgent):
    """
    RENIX cricket-specialist agent.

    This agent provides cricket-specific functionality while remaining
    compatible with the larger RENIX agent architecture.
    """

    agent_name = "cricket_agent"

    agent_description = (
        "Provides cricket analysis, statistics, training, "
        "batting, bowling, fielding and match assistance."
    )

    agent_version = "1.0.0"

    def __init__(
        self,
        *,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        priority: AgentPriority = AgentPriority.NORMAL,
        logger: Optional[logging.Logger] = None,
        event_callback=None,
        confirmation_callback=None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            priority=priority,
            logger=logger,
            event_callback=event_callback,
            confirmation_callback=confirmation_callback,
            context=context,
        )

        # Cricket subsystem modules.
        self.cricket_manager = None
        self.batting_analysis = None
        self.bowling_analysis = None
        self.fielding_analysis = None
        self.match_statistics = None
        self.video_analysis = None
        self.shot_detection = None
        self.training = None

        self._modules_loaded = False

        # Fallback in-memory statistics.
        self._matches: List[Dict[str, Any]] = []
        self._batting_records: List[Dict[str, Any]] = []
        self._bowling_records: List[Dict[str, Any]] = []
        self._fielding_records: List[Dict[str, Any]] = []
        self._training_sessions: List[Dict[str, Any]] = []

    # ========================================================================
    # CAPABILITIES
    # ========================================================================

    def _register_default_capabilities(self) -> None:
        """Register cricket-specific capabilities."""

        super()._register_default_capabilities()

        capabilities = [
            AgentCapability(
                name="cricket",
                description="General cricket assistance.",
            ),
            AgentCapability(
                name="batting",
                description="Analyze and improve batting.",
            ),
            AgentCapability(
                name="bowling",
                description="Analyze and improve bowling.",
            ),
            AgentCapability(
                name="fielding",
                description="Analyze and improve fielding.",
            ),
            AgentCapability(
                name="match_statistics",
                description="Calculate cricket match statistics.",
            ),
            AgentCapability(
                name="training",
                description="Create cricket training plans.",
            ),
            AgentCapability(
                name="shot_analysis",
                description="Analyze cricket shots and technique.",
            ),
            AgentCapability(
                name="match_analysis",
                description="Analyze match performance.",
            ),
            AgentCapability(
                name="performance",
                description="Track cricket performance.",
            ),
            AgentCapability(
                name="cricket_video",
                description="Analyze cricket video when video tools are connected.",
            ),
        ]

        for capability in capabilities:
            self.register_capability(
                capability
            )

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    async def on_initialize(self) -> bool:
        """
        Initialize cricket subsystem modules.
        """

        self._load_cricket_modules()

        return True

    def _load_cricket_modules(self) -> None:
        """
        Dynamically load cricket modules.

        Missing modules are tolerated because the RENIX project is
        being built incrementally.
        """

        if self._modules_loaded:
            return

        self._modules_loaded = True

        module_map = {
            "cricket_manager": (
                "sports.cricket.cricket_manager",
                "CricketManager",
            ),
            "batting_analysis": (
                "sports.cricket.batting_analysis",
                "BattingAnalysis",
            ),
            "bowling_analysis": (
                "sports.cricket.bowling_analysis",
                "BowlingAnalysis",
            ),
            "fielding_analysis": (
                "sports.cricket.fielding_analysis",
                "FieldingAnalysis",
            ),
            "match_statistics": (
                "sports.cricket.match_statistics",
                "MatchStatistics",
            ),
            "video_analysis": (
                "sports.cricket.video_analysis",
                "VideoAnalysis",
            ),
            "shot_detection": (
                "sports.cricket.shot_detection",
                "ShotDetection",
            ),
            "training": (
                "sports.cricket.training",
                "Training",
            ),
        }

        for attribute, (
            module_name,
            class_name,
        ) in module_map.items():

            try:

                module = __import__(
                    module_name,
                    fromlist=[class_name],
                )

                cls = getattr(
                    module,
                    class_name,
                )

                try:
                    instance = cls()
                except TypeError:
                    instance = cls

                setattr(
                    self,
                    attribute,
                    instance,
                )

            except Exception as exc:

                self.logger.debug(
                    "Cricket module %s unavailable: %s",
                    class_name,
                    exc,
                )

    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================

    async def execute(
        self,
        task: AgentTask,
    ) -> Any:
        """
        Execute a cricket task.
        """

        action = self._resolve_action(
            task
        )

        handlers = {
            "cricket": self.cricket,
            "batting": self.analyze_batting,
            "batting_analysis": self.analyze_batting,
            "bowling": self.analyze_bowling,
            "bowling_analysis": self.analyze_bowling,
            "fielding": self.analyze_fielding,
            "fielding_analysis": self.analyze_fielding,
            "statistics": self.match_statistics_report,
            "match_statistics": self.match_statistics_report,
            "match": self.analyze_match,
            "match_analysis": self.analyze_match,
            "training": self.create_training_plan,
            "train": self.create_training_plan,
            "shot": self.analyze_shot,
            "shot_analysis": self.analyze_shot,
            "performance": self.performance_report,
            "video": self.analyze_video,
            "video_analysis": self.analyze_video,
        }

        handler = handlers.get(
            action
        )

        if handler is None:

            return self.failure_result(
                task,
                f"Unknown cricket action: {action}",
                message=(
                    f"I don't know how to perform "
                    f"cricket action '{action}'."
                ),
                started_at=task.created_at,
            )

        try:

            parameters = dict(
                task.parameters
            )

            parameters.pop(
                "action",
                None,
            )

            parameters.pop(
                "capability",
                None,
            )

            result = handler(
                **parameters
            )

            if asyncio.iscoroutine(
                result
            ):
                result = await result

            return result

        except Exception as exc:

            self.logger.exception(
                "Cricket action failed: %s",
                action,
            )

            return self.failure_result(
                task,
                str(exc),
                message=(
                    f"Cricket action '{action}' failed."
                ),
                started_at=task.created_at,
            )

    # ========================================================================
    # ACTION RESOLUTION
    # ========================================================================

    def _resolve_action(
        self,
        task: AgentTask,
    ) -> str:

        explicit_action = task.parameters.get(
            "action"
        )

        if explicit_action:

            return str(
                explicit_action
            ).strip().lower()

        instruction = (
            task.instruction
            .strip()
            .lower()
        )

        if (
            "batting" in instruction
            or "batting analysis" in instruction
            or "bat" in instruction
        ):
            return "batting"

        if (
            "bowling" in instruction
            or "bowling analysis" in instruction
            or "bowl" in instruction
        ):
            return "bowling"

        if (
            "fielding" in instruction
            or "fielding analysis" in instruction
            or "fielder" in instruction
        ):
            return "fielding"

        if (
            "statistics" in instruction
            or "stats" in instruction
            or "scorecard" in instruction
        ):
            return "statistics"

        if (
            "match analysis" in instruction
            or "analyze my match" in instruction
        ):
            return "match_analysis"

        if (
            "training plan" in instruction
            or "training" in instruction
            or "practice plan" in instruction
            or "practice" in instruction
        ):
            return "training"

        if (
            "shot analysis" in instruction
            or "analyze my shot" in instruction
            or "shot" in instruction
        ):
            return "shot"

        if (
            "video analysis" in instruction
            or "analyze video" in instruction
            or "cricket video" in instruction
        ):
            return "video"

        if (
            "performance" in instruction
            or "my performance" in instruction
        ):
            return "performance"

        if "cricket" in instruction:

            return "cricket"

        return "unknown"

    # ========================================================================
    # GENERAL CRICKET
    # ========================================================================

    async def cricket(
        self,
        question: Optional[str] = None,
        topic: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Handle general cricket requests.
        """

        query = (
            question
            or topic
            or "cricket"
        ).strip()

        if self.cricket_manager is not None:

            result = await self._call_module(
                self.cricket_manager,
                (
                    "answer",
                    "assist",
                    "ask",
                    "cricket",
                ),
                question=query,
                topic=topic,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "cricket",
                    "question": query,
                    "result": result,
                }

        return {
            "success": True,
            "action": "cricket",
            "question": query,
            "message": (
                "Cricket request received."
            ),
            "note": (
                "Connect the RENIX LLM/research layer "
                "for detailed natural-language cricket answers."
            ),
        }

    # ========================================================================
    # BATTING ANALYSIS
    # ========================================================================

    async def analyze_batting(
        self,
        runs: Optional[float] = None,
        balls: Optional[float] = None,
        fours: int = 0,
        sixes: int = 0,
        dismissals: int = 0,
        dot_balls: int = 0,
        boundaries: Optional[int] = None,
        strike_rotation: Optional[float] = None,
        notes: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Analyze batting performance.
        """

        if self.batting_analysis is not None:

            result = await self._call_module(
                self.batting_analysis,
                (
                    "analyze",
                    "analyze_batting",
                    "evaluate",
                ),
                runs=runs,
                balls=balls,
                fours=fours,
                sixes=sixes,
                dismissals=dismissals,
                dot_balls=dot_balls,
                boundaries=boundaries,
                strike_rotation=strike_rotation,
                notes=notes,
                **kwargs,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "batting",
                    "analysis": result,
                }

        runs_value = (
            float(runs)
            if runs is not None
            else 0.0
        )

        balls_value = (
            float(balls)
            if balls is not None
            else 0.0
        )

        fours = max(
            0,
            int(
                fours
            ),
        )

        sixes = max(
            0,
            int(
                sixes
            ),
        )

        dismissals = max(
            0,
            int(
                dismissals
            ),
        )

        dot_balls = max(
            0,
            int(
                dot_balls
            ),
        )

        if boundaries is None:

            boundaries = (
                fours
                + sixes
            )

        if balls_value > 0:

            strike_rate = (
                runs_value
                / balls_value
                * 100
            )

        else:

            strike_rate = 0.0

        boundary_runs = (
            fours * 4
            + sixes * 6
        )

        boundary_percentage = (
            boundary_runs
            / runs_value
            * 100
            if runs_value > 0
            else 0.0
        )

        dot_ball_percentage = (
            dot_balls
            / balls_value
            * 100
            if balls_value > 0
            else None
        )

        average = (
            runs_value
            / dismissals
            if dismissals > 0
            else None
        )

        strengths = []
        weaknesses = []

        if strike_rate >= 140:
            strengths.append(
                "Excellent scoring rate."
            )
        elif strike_rate >= 100:
            strengths.append(
                "Healthy scoring rate."
            )
        elif balls_value > 0:
            weaknesses.append(
                "Scoring rate can be improved."
            )

        if sixes > 0:
            strengths.append(
                "Boundary-hitting ability is present."
            )

        if (
            dot_ball_percentage is not None
            and dot_ball_percentage > 45
        ):
            weaknesses.append(
                "Too many dot balls."
            )

        if (
            boundary_percentage > 70
            and runs_value > 20
        ):
            weaknesses.append(
                "Consider improving strike rotation "
                "instead of relying heavily on boundaries."
            )

        recommendations = [
            "Practice strike rotation.",
            "Train against both pace and spin.",
            "Review dismissals and identify recurring patterns.",
            "Practice scoring options for different lengths.",
        ]

        record = {
            "timestamp": self._timestamp(),
            "runs": runs_value,
            "balls": balls_value,
            "fours": fours,
            "sixes": sixes,
            "dismissals": dismissals,
            "strike_rate": round(
                strike_rate,
                2,
            ),
        }

        self._batting_records.append(
            record
        )

        return {
            "success": True,
            "action": "batting",
            "statistics": {
                "runs": runs_value,
                "balls": balls_value,
                "fours": fours,
                "sixes": sixes,
                "dismissals": dismissals,
                "strike_rate": round(
                    strike_rate,
                    2,
                ),
                "boundary_runs": boundary_runs,
                "boundary_percentage": round(
                    boundary_percentage,
                    2,
                ),
                "dot_ball_percentage": (
                    round(
                        dot_ball_percentage,
                        2,
                    )
                    if dot_ball_percentage is not None
                    else None
                ),
                "average": (
                    round(
                        average,
                        2,
                    )
                    if average is not None
                    else None
                ),
            },
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
            "notes": notes,
            "method": "fallback",
        }

    # ========================================================================
    # BOWLING ANALYSIS
    # ========================================================================

    async def analyze_bowling(
        self,
        runs_conceded: Optional[float] = None,
        balls: Optional[int] = None,
        wickets: int = 0,
        maidens: int = 0,
        wides: int = 0,
        no_balls: int = 0,
        dot_balls: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Analyze bowling performance.
        """

        if self.bowling_analysis is not None:

            result = await self._call_module(
                self.bowling_analysis,
                (
                    "analyze",
                    "analyze_bowling",
                    "evaluate",
                ),
                runs_conceded=runs_conceded,
                balls=balls,
                wickets=wickets,
                maidens=maidens,
                wides=wides,
                no_balls=no_balls,
                dot_balls=dot_balls,
                **kwargs,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "bowling",
                    "analysis": result,
                }

        runs_value = (
            float(runs_conceded)
            if runs_conceded is not None
            else 0.0
        )

        balls_value = max(
            0,
            int(
                balls or 0
            ),
        )

        wickets = max(
            0,
            int(
                wickets
            ),
        )

        maidens = max(
            0,
            int(
                maidens
            ),
        )

        wides = max(
            0,
            int(
                wides
            ),
        )

        no_balls = max(
            0,
            int(
                no_balls
            ),
        )

        dot_balls = max(
            0,
            int(
                dot_balls
            ),
        )

        overs = balls_value / 6

        economy = (
            runs_value
            / overs
            if overs > 0
            else 0.0
        )

        strike_rate = (
            balls_value
            / wickets
            if wickets > 0
            else None
        )

        dot_percentage = (
            dot_balls
            / balls_value
            * 100
            if balls_value > 0
            else None
        )

        strengths = []
        weaknesses = []

        if economy <= 6 and balls_value > 0:
            strengths.append(
                "Good economy rate."
            )

        if wickets >= 2:
            strengths.append(
                "Good wicket-taking contribution."
            )

        if (
            dot_percentage is not None
            and dot_percentage >= 40
        ):
            strengths.append(
                "Strong dot-ball pressure."
            )

        if economy > 8 and balls_value > 0:
            weaknesses.append(
                "Economy rate needs improvement."
            )

        if wides + no_balls > 2:
            weaknesses.append(
                "Reduce extras and improve bowling discipline."
            )

        recommendations = [
            "Work on consistent line and length.",
            "Develop variations without losing control.",
            "Use field settings that match your bowling plan.",
            "Review every boundary conceded and identify the cause.",
        ]

        record = {
            "timestamp": self._timestamp(),
            "runs_conceded": runs_value,
            "balls": balls_value,
            "wickets": wickets,
            "economy": round(
                economy,
                2,
            ),
        }

        self._bowling_records.append(
            record
        )

        return {
            "success": True,
            "action": "bowling",
            "statistics": {
                "runs_conceded": runs_value,
                "balls": balls_value,
                "overs": round(
                    overs,
                    1,
                ),
                "wickets": wickets,
                "maidens": maidens,
                "wides": wides,
                "no_balls": no_balls,
                "economy": round(
                    economy,
                    2,
                ),
                "strike_rate": (
                    round(
                        strike_rate,
                        2,
                    )
                    if strike_rate is not None
                    else None
                ),
                "dot_ball_percentage": (
                    round(
                        dot_percentage,
                        2,
                    )
                    if dot_percentage is not None
                    else None
                ),
            },
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
            "method": "fallback",
        }

    # ========================================================================
    # FIELDING ANALYSIS
    # ========================================================================

    async def analyze_fielding(
        self,
        catches: int = 0,
        run_outs: int = 0,
        direct_hits: int = 0,
        misfields: int = 0,
        stops: int = 0,
        dropped_catches: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Analyze fielding performance.
        """

        if self.fielding_analysis is not None:

            result = await self._call_module(
                self.fielding_analysis,
                (
                    "analyze",
                    "analyze_fielding",
                    "evaluate",
                ),
                catches=catches,
                run_outs=run_outs,
                direct_hits=direct_hits,
                misfields=misfields,
                stops=stops,
                dropped_catches=dropped_catches,
                **kwargs,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "fielding",
                    "analysis": result,
                }

        catches = max(
            0,
            int(
                catches
            ),
        )

        run_outs = max(
            0,
            int(
                run_outs
            ),
        )

        direct_hits = max(
            0,
            int(
                direct_hits
            ),
        )

        misfields = max(
            0,
            int(
                misfields
            ),
        )

        stops = max(
            0,
            int(
                stops
            ),
        )

        dropped_catches = max(
            0,
            int(
                dropped_catches
            ),
        )

        positive_actions = (
            catches
            + run_outs
            + direct_hits
            + stops
        )

        negative_actions = (
            misfields
            + dropped_catches
        )

        strengths = []
        weaknesses = []

        if catches > 0:
            strengths.append(
                "Reliable catching contribution."
            )

        if run_outs > 0 or direct_hits > 0:
            strengths.append(
                "Good throwing and run-out potential."
            )

        if stops > 0:
            strengths.append(
                "Active ground-fielding contribution."
            )

        if dropped_catches > 0:
            weaknesses.append(
                "Catching consistency needs improvement."
            )

        if misfields > 0:
            weaknesses.append(
                "Reduce unnecessary misfields."
            )

        return {
            "success": True,
            "action": "fielding",
            "statistics": {
                "catches": catches,
                "run_outs": run_outs,
                "direct_hits": direct_hits,
                "stops": stops,
                "misfields": misfields,
                "dropped_catches": dropped_catches,
                "positive_actions": positive_actions,
                "negative_actions": negative_actions,
            },
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": [
                "Practice high catches and close catches.",
                "Improve first-step reaction speed.",
                "Practice direct-hit throwing.",
                "Train ground-ball pickup and release speed.",
            ],
            "method": "fallback",
        }

    # ========================================================================
    # MATCH STATISTICS
    # ========================================================================

    async def match_statistics_report(
        self,
        matches: Optional[
            Sequence[Dict[str, Any]]
        ] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Calculate aggregate match statistics.
        """

        if self.match_statistics is not None:

            result = await self._call_module(
                self.match_statistics,
                (
                    "calculate",
                    "analyze",
                    "statistics",
                    "report",
                ),
                matches=matches,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "match_statistics",
                    "statistics": result,
                }

        match_list = list(
            matches
            if matches is not None
            else self._matches
        )

        if not match_list:

            return {
                "success": True,
                "action": "match_statistics",
                "matches": 0,
                "statistics": {},
                "message": (
                    "No match records are available."
                ),
            }

        total_runs = sum(
            self._number(
                match.get("runs")
            )
            for match in match_list
        )

        total_balls = sum(
            self._number(
                match.get("balls")
            )
            for match in match_list
        )

        total_wickets = sum(
            self._number(
                match.get("wickets")
            )
            for match in match_list
        )

        dismissals = sum(
            self._number(
                match.get("dismissals")
            )
            for match in match_list
        )

        average = (
            total_runs / dismissals
            if dismissals > 0
            else None
        )

        strike_rate = (
            total_runs
            / total_balls
            * 100
            if total_balls > 0
            else None
        )

        return {
            "success": True,
            "action": "match_statistics",
            "matches": len(
                match_list
            ),
            "statistics": {
                "total_runs": total_runs,
                "total_balls": total_balls,
                "total_wickets": total_wickets,
                "dismissals": dismissals,
                "average": (
                    round(
                        average,
                        2,
                    )
                    if average is not None
                    else None
                ),
                "strike_rate": (
                    round(
                        strike_rate,
                        2,
                    )
                    if strike_rate is not None
                    else None
                ),
            },
            "method": "fallback",
        }

    # ========================================================================
    # MATCH ANALYSIS
    # ========================================================================

    async def analyze_match(
        self,
        match: Optional[
            Dict[str, Any]
        ] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Analyze one match.
        """

        match_data = dict(
            match
            or kwargs
        )

        if not match_data:

            return {
                "success": False,
                "error": (
                    "No match data was provided."
                ),
            }

        self._matches.append(
            {
                **match_data,
                "recorded_at": self._timestamp(),
            }
        )

        batting = await self.analyze_batting(
            runs=match_data.get(
                "runs"
            ),
            balls=match_data.get(
                "balls"
            ),
            fours=match_data.get(
                "fours",
                0,
            ),
            sixes=match_data.get(
                "sixes",
                0,
            ),
            dismissals=match_data.get(
                "dismissals",
                0,
            ),
            dot_balls=match_data.get(
                "dot_balls",
                0,
            ),
        )

        bowling = await self.analyze_bowling(
            runs_conceded=match_data.get(
                "runs_conceded"
            ),
            balls=match_data.get(
                "bowling_balls"
            ),
            wickets=match_data.get(
                "wickets",
                0,
            ),
            maidens=match_data.get(
                "maidens",
                0,
            ),
            wides=match_data.get(
                "wides",
                0,
            ),
            no_balls=match_data.get(
                "no_balls",
                0,
            ),
            dot_balls=match_data.get(
                "bowling_dot_balls",
                0,
            ),
        )

        fielding = await self.analyze_fielding(
            catches=match_data.get(
                "catches",
                0,
            ),
            run_outs=match_data.get(
                "run_outs",
                0,
            ),
            direct_hits=match_data.get(
                "direct_hits",
                0,
            ),
            misfields=match_data.get(
                "misfields",
                0,
            ),
            stops=match_data.get(
                "stops",
                0,
            ),
            dropped_catches=match_data.get(
                "dropped_catches",
                0,
            ),
        )

        return {
            "success": True,
            "action": "match_analysis",
            "match": match_data,
            "batting": batting,
            "bowling": bowling,
            "fielding": fielding,
            "summary": self._build_match_summary(
                batting,
                bowling,
                fielding,
            ),
        }

    # ========================================================================
    # TRAINING
    # ========================================================================

    async def create_training_plan(
        self,
        goals: Optional[
            Sequence[Any]
        ] = None,
        duration_minutes: int = 90,
        skill_level: str = "intermediate",
        focus: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Create a cricket training plan.
        """

        goal_list = [
            str(goal).strip()
            for goal in (
                goals or []
            )
            if str(goal).strip()
        ]

        if focus and focus.strip():
            goal_list.insert(
                0,
                focus.strip(),
            )

        if not goal_list:

            goal_list = [
                "batting",
                "bowling",
                "fielding",
                "fitness",
            ]

        duration_minutes = max(
            20,
            int(
                duration_minutes
            ),
        )

        if self.training is not None:

            result = await self._call_module(
                self.training,
                (
                    "create_training_plan",
                    "create_plan",
                    "create",
                    "generate",
                ),
                goals=goal_list,
                duration_minutes=duration_minutes,
                skill_level=skill_level,
                **kwargs,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "training",
                    "plan": result,
                }

        warmup = min(
            10,
            max(
                5,
                duration_minutes // 10,
            ),
        )

        cooldown = min(
            10,
            max(
                5,
                duration_minutes // 10,
            ),
        )

        usable_time = max(
            10,
            duration_minutes
            - warmup
            - cooldown,
        )

        block_time = max(
            5,
            usable_time
            // len(
                goal_list
            ),
        )

        drills = []

        drill_library = {
            "batting": [
                "Shadow batting",
                "Front-foot drives",
                "Back-foot shots",
                "Spin bowling practice",
                "Strike rotation",
            ],
            "bowling": [
                "Target bowling",
                "Line-and-length drill",
                "Variation practice",
                "Yorker practice",
                "Short-ball control",
            ],
            "fielding": [
                "Ground fielding",
                "High catches",
                "Direct-hit practice",
                "Reaction catches",
                "Throwing accuracy",
            ],
            "fitness": [
                "Sprint intervals",
                "Agility ladder",
                "Lateral movement",
                "Core training",
                "Mobility",
            ],
            "speed": [
                "Short sprints",
                "Acceleration drills",
                "Change-of-direction drills",
                "Reaction sprinting",
            ],
            "spin": [
                "Spin reading",
                "Footwork against spin",
                "Sweep practice",
                "Defensive rotation",
            ],
        }

        for goal in goal_list:

            normalized = goal.lower()

            selected = None

            for key, options in drill_library.items():

                if key in normalized:

                    selected = options
                    break

            if selected is None:

                selected = [
                    f"{goal} technique practice",
                    f"{goal} repetition drill",
                    f"{goal} pressure drill",
                ]

            drills.append(
                {
                    "focus": goal,
                    "minutes": block_time,
                    "drills": selected,
                }
            )

        plan = {
            "warmup": {
                "minutes": warmup,
                "activities": [
                    "Light jogging",
                    "Dynamic mobility",
                    "Shoulder and hip activation",
                ],
            },
            "skill_blocks": drills,
            "cooldown": {
                "minutes": cooldown,
                "activities": [
                    "Light walking",
                    "Mobility",
                    "Breathing recovery",
                ],
            },
        }

        session = {
            "timestamp": self._timestamp(),
            "duration_minutes": duration_minutes,
            "skill_level": skill_level,
            "goals": goal_list,
            "plan": plan,
        }

        self._training_sessions.append(
            session
        )

        return {
            "success": True,
            "action": "training",
            "duration_minutes": duration_minutes,
            "skill_level": skill_level,
            "goals": goal_list,
            "plan": plan,
            "method": "fallback",
        }

    # ========================================================================
    # SHOT ANALYSIS
    # ========================================================================

    async def analyze_shot(
        self,
        shot_type: Optional[str] = None,
        contact_point: Optional[str] = None,
        footwork: Optional[str] = None,
        ball_length: Optional[str] = None,
        ball_line: Optional[str] = None,
        outcome: Optional[str] = None,
        notes: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Analyze a cricket shot.
        """

        if self.shot_detection is not None:

            result = await self._call_module(
                self.shot_detection,
                (
                    "analyze",
                    "analyze_shot",
                    "evaluate",
                ),
                shot_type=shot_type,
                contact_point=contact_point,
                footwork=footwork,
                ball_length=ball_length,
                ball_line=ball_line,
                outcome=outcome,
                notes=notes,
                **kwargs,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "shot_analysis",
                    "analysis": result,
                }

        recommendations = []

        if not footwork:

            recommendations.append(
                "Record or describe your footwork for a more precise analysis."
            )

        if (
            ball_length
            and "short" in ball_length.lower()
            and not footwork
        ):

            recommendations.append(
                "Against short bowling, react early and choose "
                "between controlled back-foot shots or leaving the ball."
            )

        if (
            ball_line
            and "leg" in ball_line.lower()
        ):

            recommendations.append(
                "Against leg-side deliveries, maintain balance "
                "and avoid falling over toward the ball."
            )

        if (
            ball_length
            and "full" in ball_length.lower()
        ):

            recommendations.append(
                "For full deliveries, move decisively toward "
                "the line and keep the head over the ball."
            )

        if not recommendations:

            recommendations = [
                "Keep your head stable through contact.",
                "Watch the ball until contact.",
                "Maintain balance after the shot.",
                "Review whether your shot selection matched the delivery.",
            ]

        return {
            "success": True,
            "action": "shot_analysis",
            "shot": {
                "type": shot_type,
                "contact_point": contact_point,
                "footwork": footwork,
                "ball_length": ball_length,
                "ball_line": ball_line,
                "outcome": outcome,
                "notes": notes,
            },
            "recommendations": recommendations,
            "method": "fallback",
        }

    # ========================================================================
    # VIDEO ANALYSIS
    # ========================================================================

    async def analyze_video(
        self,
        video_path: Optional[str] = None,
        analysis_type: str = "all",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Analyze cricket video when the video-analysis subsystem is available.
        """

        if not video_path:

            return {
                "success": False,
                "action": "video_analysis",
                "error": (
                    "A video path is required."
                ),
            }

        if self.video_analysis is not None:

            result = await self._call_module(
                self.video_analysis,
                (
                    "analyze",
                    "analyze_video",
                    "process",
                ),
                video_path=video_path,
                analysis_type=analysis_type,
                **kwargs,
            )

            if result is not None:

                return {
                    "success": True,
                    "action": "video_analysis",
                    "video_path": video_path,
                    "analysis": result,
                }

        return {
            "success": False,
            "action": "video_analysis",
            "video_path": video_path,
            "analysis_type": analysis_type,
            "error": (
                "Video analysis module is not connected yet."
            ),
        }

    # ========================================================================
    # PERFORMANCE REPORT
    # ========================================================================

    async def performance_report(
        self,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Generate an overall cricket performance report.
        """

        batting = self._aggregate_batting()
        bowling = self._aggregate_bowling()

        return {
            "success": True,
            "action": "performance",
            "batting": batting,
            "bowling": bowling,
            "training_sessions": len(
                self._training_sessions
            ),
            "matches": len(
                self._matches
            ),
            "recommendations": self._overall_recommendations(
                batting,
                bowling,
            ),
        }

    # ========================================================================
    # AGGREGATION
    # ========================================================================

    def _aggregate_batting(
        self,
    ) -> Dict[str, Any]:

        if not self._batting_records:

            return {
                "matches_recorded": 0,
            }

        runs = sum(
            self._number(
                item.get("runs")
            )
            for item in self._batting_records
        )

        balls = sum(
            self._number(
                item.get("balls")
            )
            for item in self._batting_records
        )

        dismissals = sum(
            self._number(
                item.get("dismissals")
            )
            for item in self._batting_records
        )

        return {
            "matches_recorded": len(
                self._batting_records
            ),
            "runs": runs,
            "balls": balls,
            "dismissals": dismissals,
            "average": (
                round(
                    runs / dismissals,
                    2,
                )
                if dismissals > 0
                else None
            ),
            "strike_rate": (
                round(
                    runs / balls * 100,
                    2,
                )
                if balls > 0
                else None
            ),
        }

    def _aggregate_bowling(
        self,
    ) -> Dict[str, Any]:

        if not self._bowling_records:

            return {
                "matches_recorded": 0,
            }

        runs = sum(
            self._number(
                item.get("runs_conceded")
            )
            for item in self._bowling_records
        )

        balls = sum(
            self._number(
                item.get("balls")
            )
            for item in self._bowling_records
        )

        wickets = sum(
            self._number(
                item.get("wickets")
            )
            for item in self._bowling_records
        )

        overs = balls / 6

        return {
            "matches_recorded": len(
                self._bowling_records
            ),
            "runs_conceded": runs,
            "balls": balls,
            "overs": round(
                overs,
                1,
            ),
            "wickets": wickets,
            "economy": (
                round(
                    runs / overs,
                    2,
                )
                if overs > 0
                else None
            ),
            "bowling_average": (
                round(
                    runs / wickets,
                    2,
                )
                if wickets > 0
                else None
            ),
        }

    # ========================================================================
    # SUMMARY
    # ========================================================================

    @staticmethod
    def _build_match_summary(
        batting: Dict[str, Any],
        bowling: Dict[str, Any],
        fielding: Dict[str, Any],
    ) -> Dict[str, Any]:

        batting_strengths = batting.get(
            "strengths",
            []
        )

        bowling_strengths = bowling.get(
            "strengths",
            []
        )

        fielding_strengths = fielding.get(
            "strengths",
            []
        )

        batting_weaknesses = batting.get(
            "weaknesses",
            []
        )

        bowling_weaknesses = bowling.get(
            "weaknesses",
            []
        )

        fielding_weaknesses = fielding.get(
            "weaknesses",
            []
        )

        return {
            "strengths": (
                batting_strengths
                + bowling_strengths
                + fielding_strengths
            ),
            "areas_to_improve": (
                batting_weaknesses
                + bowling_weaknesses
                + fielding_weaknesses
            ),
        }

    @staticmethod
    def _overall_recommendations(
        batting: Dict[str, Any],
        bowling: Dict[str, Any],
    ) -> List[str]:

        recommendations = []

        batting_sr = batting.get(
            "strike_rate"
        )

        if (
            batting_sr is not None
            and batting_sr < 100
        ):

            recommendations.append(
                "Increase scoring efficiency while maintaining shot control."
            )

        bowling_economy = bowling.get(
            "economy"
        )

        if (
            bowling_economy is not None
            and bowling_economy > 8
        ):

            recommendations.append(
                "Work on bowling accuracy and reducing boundary balls."
            )

        recommendations.extend(
            [
                "Maintain regular batting, bowling and fielding practice.",
                "Track match performances instead of relying only on net-session confidence.",
                "Review mistakes after every competitive match.",
            ]
        )

        return recommendations

    # ========================================================================
    # GENERIC MODULE CALLER
    # ========================================================================

    async def _call_module(
        self,
        module: Any,
        method_names: Iterable[str],
        **kwargs: Any,
    ) -> Any:
        """
        Call the first compatible method exposed by a cricket module.
        """

        for method_name in method_names:

            method = getattr(
                module,
                method_name,
                None,
            )

            if not callable(
                method
            ):
                continue

            try:

                result = method(
                    **kwargs
                )

                if asyncio.iscoroutine(
                    result
                ):

                    result = await result

                return result

            except TypeError:

                try:

                    result = method(
                        kwargs
                    )

                    if asyncio.iscoroutine(
                        result
                    ):

                        result = await result

                    return result

                except Exception as exc:

                    self.logger.debug(
                        "Cricket method %s failed: %s",
                        method_name,
                        exc,
                    )

            except Exception as exc:

                self.logger.debug(
                    "Cricket method %s failed: %s",
                    method_name,
                    exc,
                )

        return None

    # ========================================================================
    # UTILITIES
    # ========================================================================

    @staticmethod
    def _number(
        value: Any,
    ) -> float:

        try:

            return float(
                value
                if value is not None
                else 0
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0.0

    @staticmethod
    def _timestamp() -> str:

        return (
            datetime.now(
                timezone.utc
            ).isoformat()
        )


__all__ = [
    "CricketAgent",
]


