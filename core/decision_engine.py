"""
RENIX AI
Core Decision Engine

Responsible for:
- Evaluating possible actions
- Selecting the best action
- Applying priorities
- Checking confidence
- Checking risk
- Checking confirmation requirements
- Comparing alternatives
- Making execution decisions
- Supporting autonomous decision-making
- Returning structured decisions

The Decision Engine decides WHAT should happen.
The Execution Engine is responsible for actually doing it.
"""

from __future__ import annotations

import logging
import time
import uuid

from dataclasses import dataclass, field
from typing import Any, Optional


logger = logging.getLogger(
    "RENIX.DecisionEngine"
)


# ============================================================================
# DECISION OPTION
# ============================================================================

@dataclass
class DecisionOption:
    """
    Represents one possible action RENIX can choose.
    """

    option_id: str

    name: str

    description: str

    action: Optional[str] = None

    capability: Optional[str] = None

    parameters: dict[str, Any] = field(
        default_factory=dict
    )

    score: float = 0.0

    confidence: float = 0.0

    priority: int = 5

    risk: float = 0.0

    cost: float = 0.0

    benefit: float = 0.0

    reversible: bool = True

    requires_confirmation: bool = False

    available: bool = True

    blocked: bool = False

    block_reason: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: float = field(
        default_factory=time.time
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert option into a dictionary.
        """

        return {
            "option_id": self.option_id,
            "name": self.name,
            "description": self.description,
            "action": self.action,
            "capability": self.capability,
            "parameters": dict(
                self.parameters
            ),
            "score": self.score,
            "confidence": self.confidence,
            "priority": self.priority,
            "risk": self.risk,
            "cost": self.cost,
            "benefit": self.benefit,
            "reversible": self.reversible,
            "requires_confirmation": (
                self.requires_confirmation
            ),
            "available": self.available,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "metadata": dict(
                self.metadata
            ),
            "created_at": self.created_at,
        }


# ============================================================================
# DECISION
# ============================================================================

@dataclass
class Decision:
    """
    Represents the final decision made by RENIX.
    """

    decision_id: str

    objective: str

    selected_option: Optional[
        DecisionOption
    ] = None

    alternatives: list[
        DecisionOption
    ] = field(
        default_factory=list
    )

    confidence: float = 0.0

    status: str = "pending"

    requires_confirmation: bool = False

    reason: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: float = field(
        default_factory=time.time
    )

    decided_at: Optional[float] = None

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert decision into a dictionary.
        """

        return {
            "decision_id": self.decision_id,
            "objective": self.objective,
            "selected_option": (
                self.selected_option.to_dict()
                if self.selected_option
                else None
            ),
            "alternatives": [
                option.to_dict()
                for option in self.alternatives
            ],
            "confidence": self.confidence,
            "status": self.status,
            "requires_confirmation": (
                self.requires_confirmation
            ),
            "reason": self.reason,
            "metadata": dict(
                self.metadata
            ),
            "created_at": self.created_at,
            "decided_at": self.decided_at,
        }


# ============================================================================
# DECISION ENGINE
# ============================================================================

class DecisionEngine:
    """
    Main RENIX decision-making engine.
    """

    def __init__(
        self,
        capability_manager: Any = None,
        context_engine: Any = None,
        security_manager: Any = None,
    ) -> None:

        self.logger = logger

        self.capability_manager = (
            capability_manager
        )

        self.context_engine = (
            context_engine
        )

        self.security_manager = (
            security_manager
        )

        self.initialized = False

        self.created_at = time.time()

        self.decisions: dict[
            str,
            Decision
        ] = {}

        self.active_decision: Optional[
            Decision
        ] = None

        self.decision_count = 0

        self.successful_decisions = 0

        self.rejected_decisions = 0

        self.confirmation_count = 0

        self.minimum_confidence = 0.55

        self.maximum_risk = 0.75

        self.autonomous_mode = True

        self.max_history = 500

    # ========================================================================
    # INITIALIZE
    # ========================================================================

    async def initialize(
        self,
    ) -> None:
        """
        Initialize the decision engine.
        """

        self.initialized = True

        self.logger.info(
            "RENIX Decision Engine initialized."
        )

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    async def shutdown(
        self,
    ) -> None:
        """
        Shut down the decision engine.
        """

        self.initialized = False

        self.logger.info(
            "RENIX Decision Engine shutdown."
        )

    # ========================================================================
    # ID GENERATION
    # ========================================================================

    @staticmethod
    def _generate_id(
        prefix: str = "decision",
    ) -> str:
        """
        Generate a unique decision ID.
        """

        return (
            f"{prefix}_"
            f"{uuid.uuid4().hex}"
        )

    # ========================================================================
    # CLAMP
    # ========================================================================

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:
        """
        Keep a numeric value inside a range.
        """

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            value = minimum

        return max(
            minimum,
            min(
                maximum,
                value,
            ),
        )

    # ========================================================================
    # NORMALIZE OBJECTIVE
    # ========================================================================

    @staticmethod
    def _normalize_objective(
        objective: Any,
    ) -> str:
        """
        Normalize an objective.
        """

        if objective is None:

            return ""

        return str(
            objective
        ).strip()

    # ========================================================================
    # CREATE OPTION
    # ========================================================================

    def create_option(
        self,
        name: str,
        description: str,
        *,
        action: Optional[str] = None,
        capability: Optional[str] = None,
        parameters: Optional[
            dict[str, Any]
        ] = None,
        confidence: float = 0.0,
        priority: int = 5,
        risk: float = 0.0,
        cost: float = 0.0,
        benefit: float = 0.0,
        reversible: bool = True,
        requires_confirmation: bool = False,
        available: bool = True,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> DecisionOption:
        """
        Create a decision option.
        """

        return DecisionOption(
            option_id=self._generate_id(
                "option"
            ),
            name=name,
            description=description,
            action=action,
            capability=capability,
            parameters=(
                dict(parameters)
                if parameters
                else {}
            ),
            confidence=self._clamp(
                confidence
            ),
            priority=priority,
            risk=self._clamp(
                risk
            ),
            cost=max(
                0.0,
                float(cost),
            ),
            benefit=max(
                0.0,
                float(benefit),
            ),
            reversible=reversible,
            requires_confirmation=(
                requires_confirmation
            ),
            available=available,
            metadata=(
                dict(metadata)
                if metadata
                else {}
            ),
        )

    # ========================================================================
    # BUILD CONTEXT
    # ========================================================================

    def build_context(
        self,
        objective: str,
        context: Optional[
            dict[str, Any]
        ] = None,
    ) -> dict[str, Any]:
        """
        Build decision context.
        """

        result: dict[
            str,
            Any
        ] = {}

        if self.context_engine is not None:

            try:

                if hasattr(
                    self.context_engine,
                    "build_context_window",
                ):

                    context_result = (
                        self.context_engine
                        .build_context_window()
                    )

                    if isinstance(
                        context_result,
                        dict,
                    ):

                        result.update(
                            context_result
                        )

                elif hasattr(
                    self.context_engine,
                    "get_all",
                ):

                    context_result = (
                        self.context_engine
                        .get_all()
                    )

                    if isinstance(
                        context_result,
                        dict,
                    ):

                        result.update(
                            context_result
                        )

            except Exception as exc:

                self.logger.warning(
                    "Unable to load decision context: %s",
                    exc,
                )

        if context:

            result.update(
                context
            )

        result[
            "objective"
        ] = objective

        return result

    # ========================================================================
    # CHECK CAPABILITY
    # ========================================================================

    def check_capability(
        self,
        option: DecisionOption,
    ) -> bool:
        """
        Check whether the requested capability is available.
        """

        if option.capability is None:

            return True

        if self.capability_manager is None:

            return True

        try:

            if hasattr(
                self.capability_manager,
                "has_capability",
            ):

                return bool(
                    self.capability_manager
                    .has_capability(
                        option.capability
                    )
                )

            if hasattr(
                self.capability_manager,
                "is_available",
            ):

                return bool(
                    self.capability_manager
                    .is_available(
                        option.capability
                    )
                )

            if hasattr(
                self.capability_manager,
                "get_capability",
            ):

                return (
                    self.capability_manager
                    .get_capability(
                        option.capability
                    )
                    is not None
                )

        except Exception as exc:

            self.logger.warning(
                "Capability check failed: %s",
                exc,
            )

            return False

        return True

    # ========================================================================
    # CHECK SECURITY
    # ========================================================================

    def check_security(
        self,
        option: DecisionOption,
    ) -> tuple[
        bool,
        Optional[str],
    ]:
        """
        Ask the security layer whether an option is allowed.
        """

        if self.security_manager is None:

            return True, None

        try:

            if hasattr(
                self.security_manager,
                "check_action",
            ):

                result = (
                    self.security_manager
                    .check_action(
                        option.action,
                        option.parameters,
                    )
                )

                if isinstance(
                    result,
                    tuple,
                ):

                    if len(result) >= 2:

                        return (
                            bool(result[0]),
                            str(result[1])
                            if result[1]
                            else None,
                        )

                    if len(result) == 1:

                        return (
                            bool(result[0]),
                            None,
                        )

                if isinstance(
                    result,
                    bool,
                ):

                    return (
                        result,
                        None
                        if result
                        else "Security manager rejected action.",
                    )

            if hasattr(
                self.security_manager,
                "is_allowed",
            ):

                result = (
                    self.security_manager
                    .is_allowed(
                        option.action
                    )
                )

                return (
                    bool(result),
                    None
                    if result
                    else "Security policy rejected action.",
                )

        except Exception as exc:

            self.logger.warning(
                "Security check failed: %s",
                exc,
            )

            return (
                False,
                "Security check failed.",
            )

        return True, None

    # ========================================================================
    # EVALUATE OPTION
    # ========================================================================

    def evaluate_option(
        self,
        option: DecisionOption,
        *,
        context: Optional[
            dict[str, Any]
        ] = None,
    ) -> DecisionOption:
        """
        Evaluate and score one option.
        """

        if not option.available:

            option.blocked = True

            option.block_reason = (
                "Option is unavailable."
            )

            option.score = 0.0

            return option

        if not self.check_capability(
            option
        ):

            option.available = False

            option.blocked = True

            option.block_reason = (
                "Required capability is unavailable."
            )

            option.score = 0.0

            return option

        security_allowed, security_reason = (
            self.check_security(
                option
            )
        )

        if not security_allowed:

            option.available = False

            option.blocked = True

            option.block_reason = (
                security_reason
                or "Security policy rejected option."
            )

            option.score = 0.0

            return option

        # --------------------------------------------------------------------
        # Base score
        # --------------------------------------------------------------------

        confidence_score = (
            self._clamp(
                option.confidence
            )
        )

        benefit_score = (
            self._clamp(
                option.benefit
            )
        )

        risk_score = (
            self._clamp(
                option.risk
            )
        )

        cost_score = (
            self._clamp(
                option.cost
            )
        )

        priority_score = (
            self._priority_score(
                option.priority
            )
        )

        reversibility_score = (
            1.0
            if option.reversible
            else 0.5
        )

        # --------------------------------------------------------------------
        # Overall decision score
        # --------------------------------------------------------------------

        score = (
            confidence_score * 0.30
            + benefit_score * 0.25
            + priority_score * 0.15
            + reversibility_score * 0.10
            + (1.0 - risk_score) * 0.15
            + (1.0 - cost_score) * 0.05
        )

        # --------------------------------------------------------------------
        # High-risk penalty
        # --------------------------------------------------------------------

        if (
            option.risk
            > self.maximum_risk
        ):

            score *= 0.40

        # --------------------------------------------------------------------
        # Low-confidence penalty
        # --------------------------------------------------------------------

        if (
            option.confidence
            < self.minimum_confidence
        ):

            score *= 0.70

        option.score = self._clamp(
            score
        )

        return option

    # ========================================================================
    # PRIORITY SCORE
    # ========================================================================

    @staticmethod
    def _priority_score(
        priority: int,
    ) -> float:
        """
        Convert priority into a normalized score.

        Priority:
        1 = highest
        10 = lowest
        """

        try:

            priority = int(
                priority
            )

        except (
            TypeError,
            ValueError,
        ):

            priority = 5

        priority = max(
            1,
            min(
                10,
                priority,
            ),
        )

        return (
            1.0
            - (
                (priority - 1)
                / 9.0
            )
        )

    # ========================================================================
    # EVALUATE OPTIONS
    # ========================================================================

    def evaluate_options(
        self,
        options: list[
            DecisionOption
        ],
        *,
        context: Optional[
            dict[str, Any]
        ] = None,
    ) -> list[DecisionOption]:
        """
        Evaluate all available decision options.
        """

        evaluated = []

        for option in options:

            evaluated.append(
                self.evaluate_option(
                    option,
                    context=context,
                )
            )

        evaluated.sort(
            key=lambda option: (
                option.blocked,
                -option.score,
                option.priority,
            )
        )

        return evaluated

    # ========================================================================
    # SELECT BEST OPTION
    # ========================================================================

    def select_best_option(
        self,
        options: list[
            DecisionOption
        ],
    ) -> Optional[
        DecisionOption
    ]:
        """
        Select the highest-scoring valid option.
        """

        valid_options = [
            option
            for option in options
            if (
                option.available
                and not option.blocked
            )
        ]

        if not valid_options:

            return None

        valid_options.sort(
            key=lambda option: (
                option.score,
                option.benefit,
                option.confidence,
                -option.risk,
                -option.priority,
            ),
            reverse=True,
        )

        return valid_options[0]

    # ========================================================================
    # SHOULD CONFIRM
    # ========================================================================

    def should_require_confirmation(
        self,
        option: Optional[
            DecisionOption
        ],
    ) -> bool:
        """
        Determine whether user confirmation is required.
        """

        if option is None:

            return True

        if option.requires_confirmation:

            return True

        if option.blocked:

            return True

        if option.risk > self.maximum_risk:

            return True

        if option.confidence < self.minimum_confidence:

            return True

        if not option.reversible:

            return True

        return False

    # ========================================================================
    # MAKE DECISION
    # ========================================================================

    def make_decision(
        self,
        objective: str,
        options: list[
            DecisionOption
        ],
        *,
        context: Optional[
            dict[str, Any]
        ] = None,
    ) -> Decision:
        """
        Evaluate alternatives and make a final decision.
        """

        normalized = (
            self._normalize_objective(
                objective
            )
        )

        if not normalized:

            raise ValueError(
                "Decision objective cannot be empty."
            )

        if not options:

            decision = Decision(
                decision_id=self._generate_id(),
                objective=normalized,
                status="rejected",
                confidence=0.0,
                requires_confirmation=True,
                reason=(
                    "No decision options were provided."
                ),
            )

            self.decisions[
                decision.decision_id
            ] = decision

            self.rejected_decisions += 1

            return decision

        built_context = (
            self.build_context(
                normalized,
                context,
            )
        )

        evaluated = (
            self.evaluate_options(
                options,
                context=built_context,
            )
        )

        selected = (
            self.select_best_option(
                evaluated
            )
        )

        if selected is None:

            decision = Decision(
                decision_id=self._generate_id(),
                objective=normalized,
                alternatives=evaluated,
                status="rejected",
                confidence=0.0,
                requires_confirmation=True,
                reason=(
                    "No valid option is available."
                ),
                metadata={
                    "context": built_context,
                },
            )

            self.decisions[
                decision.decision_id
            ] = decision

            self.rejected_decisions += 1

            return decision

        confirmation_required = (
            self.should_require_confirmation(
                selected
            )
        )

        confidence = (
            selected.confidence
            * 0.60
            + selected.score
            * 0.40
        )

        confidence = self._clamp(
            confidence
        )

        if confirmation_required:

            status = (
                "awaiting_confirmation"
            )

            self.confirmation_count += 1

        else:

            status = "approved"

        reason = (
            f"Selected '{selected.name}' "
            f"with score "
            f"{selected.score:.3f}."
        )

        if selected.risk > 0:

            reason += (
                f" Risk={selected.risk:.2f}."
            )

        decision = Decision(
            decision_id=self._generate_id(),
            objective=normalized,
            selected_option=selected,
            alternatives=[
                option
                for option in evaluated
                if option.option_id
                != selected.option_id
            ],
            confidence=confidence,
            status=status,
            requires_confirmation=(
                confirmation_required
            ),
            reason=reason,
            metadata={
                "context": built_context,
                "option_count": len(
                    evaluated
                ),
            },
            decided_at=time.time(),
        )

        self.decisions[
            decision.decision_id
        ] = decision

        self.active_decision = decision

        self.decision_count += 1

        return decision

    # ========================================================================
    # APPROVE DECISION
    # ========================================================================

    def approve_decision(
        self,
        decision: Decision,
    ) -> Decision:
        """
        Approve a decision that was waiting for confirmation.
        """

        if (
            decision.status
            not in {
                "awaiting_confirmation",
                "pending",
            }
        ):

            return decision

        if decision.selected_option is None:

            decision.status = "rejected"

            decision.reason = (
                "Cannot approve a decision "
                "without a selected option."
            )

            self.rejected_decisions += 1

            return decision

        decision.status = "approved"

        decision.requires_confirmation = False

        decision.decided_at = time.time()

        self.active_decision = decision

        return decision

    # ========================================================================
    # REJECT DECISION
    # ========================================================================

    def reject_decision(
        self,
        decision: Decision,
        reason: str = "Rejected by user.",
    ) -> Decision:
        """
        Reject a decision.
        """

        decision.status = "rejected"

        decision.reason = reason

        decision.requires_confirmation = False

        decision.decided_at = time.time()

        self.rejected_decisions += 1

        if (
            self.active_decision
            and self.active_decision.decision_id
            == decision.decision_id
        ):

            self.active_decision = None

        return decision

    # ========================================================================
    # MARK EXECUTING
    # ========================================================================

    def mark_executing(
        self,
        decision: Decision,
    ) -> Decision:
        """
        Mark an approved decision as executing.
        """

        if decision.status != "approved":

            raise RuntimeError(
                (
                    "Only approved decisions "
                    "can begin execution."
                )
            )

        decision.status = "executing"

        return decision

    # ========================================================================
    # MARK COMPLETED
    # ========================================================================

    def mark_completed(
        self,
        decision: Decision,
        result: Any = None,
    ) -> Decision:
        """
        Mark a decision as completed.
        """

        decision.status = "completed"

        decision.metadata[
            "result"
        ] = result

        decision.decided_at = time.time()

        self.successful_decisions += 1

        if (
            self.active_decision
            and self.active_decision.decision_id
            == decision.decision_id
        ):

            self.active_decision = None

        return decision

    # ========================================================================
    # MARK FAILED
    # ========================================================================

    def mark_failed(
        self,
        decision: Decision,
        error: Any,
    ) -> Decision:
        """
        Mark a decision as failed.
        """

        decision.status = "failed"

        decision.metadata[
            "error"
        ] = str(error)

        decision.decided_at = time.time()

        if (
            self.active_decision
            and self.active_decision.decision_id
            == decision.decision_id
        ):

            self.active_decision = None

        return decision

    # ========================================================================
    # GET DECISION
    # ========================================================================

    def get_decision(
        self,
        decision_id: str,
    ) -> Optional[
        Decision
    ]:
        """
        Retrieve a decision by ID.
        """

        return self.decisions.get(
            decision_id
        )

    # ========================================================================
    # GET ACTIVE DECISION
    # ========================================================================

    def get_active_decision(
        self,
    ) -> Optional[
        Decision
    ]:
        """
        Return the active decision.
        """

        return self.active_decision

    # ========================================================================
    # LIST DECISIONS
    # ========================================================================

    def list_decisions(
        self,
        status: Optional[str] = None,
    ) -> list[Decision]:
        """
        List stored decisions.
        """

        decisions = list(
            self.decisions.values()
        )

        if status is not None:

            decisions = [
                decision
                for decision in decisions
                if decision.status
                == status
            ]

        decisions.sort(
            key=lambda decision: (
                decision.created_at
            ),
            reverse=True,
        )

        return decisions[
            :self.max_history
        ]

    # ========================================================================
    # DELETE DECISION
    # ========================================================================

    def delete_decision(
        self,
        decision_id: str,
    ) -> bool:
        """
        Delete a stored decision.
        """

        if (
            decision_id
            not in self.decisions
        ):

            return False

        if (
            self.active_decision
            and self.active_decision.decision_id
            == decision_id
        ):

            self.active_decision = None

        del self.decisions[
            decision_id
        ]

        return True

    # ========================================================================
    # ENABLE AUTONOMOUS MODE
    # ========================================================================

    def enable_autonomous_mode(
        self,
    ) -> None:
        """
        Enable autonomous decision-making.
        """

        self.autonomous_mode = True

    # ========================================================================
    # DISABLE AUTONOMOUS MODE
    # ========================================================================

    def disable_autonomous_mode(
        self,
    ) -> None:
        """
        Disable autonomous decision-making.
        """

        self.autonomous_mode = False

    # ========================================================================
    # AUTONOMOUS DECISION CHECK
    # ========================================================================

    def can_execute_autonomously(
        self,
        option: Optional[
            DecisionOption
        ],
    ) -> bool:
        """
        Determine whether an option can be executed
        without asking the user.
        """

        if not self.autonomous_mode:

            return False

        if option is None:

            return False

        if option.blocked:

            return False

        if not option.available:

            return False

        if option.risk > self.maximum_risk:

            return False

        if option.confidence < self.minimum_confidence:

            return False

        if option.requires_confirmation:

            return False

        if not option.reversible:

            return False

        return True

    # ========================================================================
    # DECISION SUMMARY
    # ========================================================================

    def summarize_decision(
        self,
        decision: Decision,
    ) -> dict[str, Any]:
        """
        Return a compact decision summary.
        """

        selected = (
            decision.selected_option
        )

        return {
            "decision_id": decision.decision_id,
            "objective": decision.objective,
            "status": decision.status,
            "confidence": decision.confidence,
            "requires_confirmation": (
                decision.requires_confirmation
            ),
            "selected_option": (
                selected.name
                if selected
                else None
            ),
            "selected_action": (
                selected.action
                if selected
                else None
            ),
            "selected_score": (
                selected.score
                if selected
                else None
            ),
            "reason": decision.reason,
        }

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(
        self,
    ) -> dict[str, Any]:
        """
        Return decision engine statistics.
        """

        return {
            "initialized": (
                self.initialized
            ),
            "decision_count": (
                self.decision_count
            ),
            "stored_decisions": len(
                self.decisions
            ),
            "successful_decisions": (
                self.successful_decisions
            ),
            "rejected_decisions": (
                self.rejected_decisions
            ),
            "confirmation_count": (
                self.confirmation_count
            ),
            "autonomous_mode": (
                self.autonomous_mode
            ),
            "minimum_confidence": (
                self.minimum_confidence
            ),
            "maximum_risk": (
                self.maximum_risk
            ),
            "active_decision": (
                self.active_decision.decision_id
                if self.active_decision
                else None
            ),
            "created_at": (
                self.created_at
            ),
        }


# ============================================================================
# GLOBAL DECISION ENGINE
# ============================================================================

decision_engine = DecisionEngine()


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def make_decision(
    objective: str,
    options: list[
        DecisionOption
    ],
    *,
    context: Optional[
        dict[str, Any]
    ] = None,
) -> Decision:
    """
    Convenience wrapper around the global decision engine.
    """

    return decision_engine.make_decision(
        objective,
        options,
        context=context,
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "DecisionOption",
    "Decision",
    "DecisionEngine",
    "decision_engine",
    "make_decision",
]


