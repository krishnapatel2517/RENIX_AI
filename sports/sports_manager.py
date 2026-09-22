"""
RENIX Sports Manager
====================

Central sports management system for RENIX.

Responsibilities:
- Register and manage sports modules
- Manage athletes and player profiles
- Track training sessions
- Record performance data
- Manage matches and activities
- Provide statistics and summaries
- Connect sport-specific modules
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


logger = logging.getLogger("RENIX.Sports.Manager")


class SportsManager:
    """
    Central manager for RENIX sports features.

    The manager acts as a bridge between different
    sports modules such as cricket and performance analysis.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.config = config or {}

        self.enabled = True

        # Registered sports modules
        self.sports: Dict[str, Any] = {}

        # Athlete profiles
        self.athletes: Dict[str, Dict[str, Any]] = {}

        # Training sessions
        self.training_sessions: List[Dict[str, Any]] = []

        # Matches and events
        self.matches: List[Dict[str, Any]] = []

        # Performance records
        self.performance_records: Dict[
            str,
            List[Dict[str, Any]]
        ] = {}

        self.max_records = int(
            self.config.get(
                "max_records",
                10000,
            )
        )

        logger.info(
            "SportsManager initialized."
        )

    # ============================================================
    # SPORTS REGISTRATION
    # ============================================================

    def register_sport(
        self,
        name: str,
        module: Any,
    ) -> bool:
        """
        Register a sport module.

        Example:
            manager.register_sport(
                "cricket",
                cricket_manager
            )
        """

        if not name:
            return False

        key = name.lower().strip()

        self.sports[key] = module

        logger.info(
            "Sport registered: %s",
            key,
        )

        return True

    def unregister_sport(
        self,
        name: str,
    ) -> bool:
        """
        Remove a registered sport.
        """

        key = name.lower().strip()

        if key not in self.sports:
            return False

        del self.sports[key]

        logger.info(
            "Sport unregistered: %s",
            key,
        )

        return True

    def get_sport(
        self,
        name: str,
    ) -> Optional[Any]:
        """
        Get a registered sport module.
        """

        return self.sports.get(
            name.lower().strip()
        )

    def list_sports(self) -> List[str]:
        """
        Return all registered sports.
        """

        return list(
            self.sports.keys()
        )

    # ============================================================
    # ATHLETE MANAGEMENT
    # ============================================================

    def create_athlete(
        self,
        name: str,
        sport: str,
        age: Optional[int] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> str:
        """
        Create an athlete profile.

        Returns:
            Athlete ID
        """

        athlete_id = str(
            uuid.uuid4()
        )

        athlete = {
            "id": athlete_id,
            "name": name,
            "sport": sport.lower(),
            "age": age,
            "created_at": (
                datetime.now().isoformat()
            ),
            "updated_at": (
                datetime.now().isoformat()
            ),
            "metadata": metadata or {},
        }

        self.athletes[
            athlete_id
        ] = athlete

        self.performance_records[
            athlete_id
        ] = []

        logger.info(
            "Athlete created: %s",
            name,
        )

        return athlete_id

    def get_athlete(
        self,
        athlete_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get athlete profile.
        """

        athlete = self.athletes.get(
            athlete_id
        )

        if athlete is None:
            return None

        return athlete.copy()

    def update_athlete(
        self,
        athlete_id: str,
        **updates: Any,
    ) -> bool:
        """
        Update athlete information.
        """

        athlete = self.athletes.get(
            athlete_id
        )

        if athlete is None:
            return False

        protected_fields = {
            "id",
            "created_at",
        }

        for key, value in updates.items():

            if key not in protected_fields:

                athlete[key] = value

        athlete["updated_at"] = (
            datetime.now().isoformat()
        )

        return True

    def delete_athlete(
        self,
        athlete_id: str,
    ) -> bool:
        """
        Delete athlete profile.
        """

        if athlete_id not in self.athletes:
            return False

        del self.athletes[
            athlete_id
        ]

        self.performance_records.pop(
            athlete_id,
            None,
        )

        logger.info(
            "Athlete deleted: %s",
            athlete_id,
        )

        return True

    def list_athletes(
        self,
        sport: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return registered athletes.

        Optionally filter by sport.
        """

        athletes = list(
            self.athletes.values()
        )

        if sport:

            sport = sport.lower()

            athletes = [
                athlete
                for athlete in athletes
                if athlete.get("sport")
                == sport
            ]

        return [
            athlete.copy()
            for athlete in athletes
        ]

    # ============================================================
    # TRAINING MANAGEMENT
    # ============================================================

    def record_training(
        self,
        athlete_id: str,
        training_type: str,
        duration_minutes: float,
        notes: str = "",
        metrics: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Optional[str]:
        """
        Record a training session.

        Returns:
            Training session ID
        """

        if athlete_id not in self.athletes:

            logger.warning(
                "Unknown athlete: %s",
                athlete_id,
            )

            return None

        session_id = str(
            uuid.uuid4()
        )

        session = {
            "id": session_id,
            "athlete_id": athlete_id,
            "training_type": training_type,
            "duration_minutes": float(
                duration_minutes
            ),
            "notes": notes,
            "metrics": metrics or {},
            "timestamp": (
                datetime.now().isoformat()
            ),
        }

        self.training_sessions.append(
            session
        )

        self._limit_records(
            self.training_sessions
        )

        logger.info(
            "Training recorded for athlete %s",
            athlete_id,
        )

        return session_id

    def get_training_history(
        self,
        athlete_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get training history for an athlete.
        """

        sessions = [
            session
            for session
            in self.training_sessions
            if session["athlete_id"]
            == athlete_id
        ]

        return sessions[-limit:]

    def get_total_training_time(
        self,
        athlete_id: str,
    ) -> float:
        """
        Calculate total training time
        in minutes.
        """

        sessions = (
            self.get_training_history(
                athlete_id,
                limit=self.max_records,
            )
        )

        return sum(
            session.get(
                "duration_minutes",
                0,
            )
            for session in sessions
        )

    # ============================================================
    # PERFORMANCE RECORDING
    # ============================================================

    def record_performance(
        self,
        athlete_id: str,
        category: str,
        metrics: Dict[str, Any],
        notes: str = "",
    ) -> Optional[str]:
        """
        Record athlete performance data.

        Example:
            record_performance(
                athlete_id,
                "batting",
                {
                    "runs": 54,
                    "balls": 42,
                    "strike_rate": 128.57
                }
            )
        """

        if athlete_id not in self.athletes:

            logger.warning(
                "Unknown athlete: %s",
                athlete_id,
            )

            return None

        record_id = str(
            uuid.uuid4()
        )

        record = {
            "id": record_id,
            "athlete_id": athlete_id,
            "category": category.lower(),
            "metrics": metrics.copy(),
            "notes": notes,
            "timestamp": (
                datetime.now().isoformat()
            ),
        }

        self.performance_records[
            athlete_id
        ].append(record)

        self._limit_records(
            self.performance_records[
                athlete_id
            ]
        )

        return record_id

    def get_performance_history(
        self,
        athlete_id: str,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get athlete performance records.
        """

        records = (
            self.performance_records.get(
                athlete_id,
                [],
            )
        )

        if category:

            category = category.lower()

            records = [
                record
                for record in records
                if record["category"]
                == category
            ]

        return records[-limit:]

    # ============================================================
    # MATCH MANAGEMENT
    # ============================================================

    def record_match(
        self,
        sport: str,
        athlete_ids: List[str],
        match_data: Dict[str, Any],
    ) -> str:
        """
        Record a sports match or event.

        Returns:
            Match ID
        """

        match_id = str(
            uuid.uuid4()
        )

        match = {
            "id": match_id,
            "sport": sport.lower(),
            "athlete_ids": athlete_ids,
            "data": match_data.copy(),
            "timestamp": (
                datetime.now().isoformat()
            ),
        }

        self.matches.append(
            match
        )

        self._limit_records(
            self.matches
        )

        logger.info(
            "Match recorded: %s",
            match_id,
        )

        return match_id

    def get_matches(
        self,
        sport: Optional[str] = None,
        athlete_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get recorded matches.
        """

        matches = self.matches

        if sport:

            matches = [
                match
                for match in matches
                if match["sport"]
                == sport.lower()
            ]

        if athlete_id:

            matches = [
                match
                for match in matches
                if athlete_id
                in match.get(
                    "athlete_ids",
                    [],
                )
            ]

        return matches[-limit:]

    # ============================================================
    # ANALYSIS
    # ============================================================

    def get_athlete_summary(
        self,
        athlete_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate overall athlete summary.
        """

        athlete = self.get_athlete(
            athlete_id
        )

        if athlete is None:
            return None

        training = (
            self.get_training_history(
                athlete_id,
                limit=self.max_records,
            )
        )

        performances = (
            self.get_performance_history(
                athlete_id,
                limit=self.max_records,
            )
        )

        matches = self.get_matches(
            athlete_id=athlete_id,
            limit=self.max_records,
        )

        return {
            "athlete": athlete,
            "total_training_sessions": len(
                training
            ),
            "total_training_minutes": (
                self.get_total_training_time(
                    athlete_id
                )
            ),
            "performance_records": len(
                performances
            ),
            "matches": len(matches),
            "last_training": (
                training[-1]
                if training
                else None
            ),
            "last_performance": (
                performances[-1]
                if performances
                else None
            ),
        }

    # ============================================================
    # UTILITIES
    # ============================================================

    def _limit_records(
        self,
        records: List[Any],
    ) -> None:
        """
        Prevent unlimited memory usage.
        """

        if (
            len(records)
            > self.max_records
        ):

            excess = (
                len(records)
                - self.max_records
            )

            del records[:excess]

    def clear_data(
        self,
        athlete_id: Optional[str] = None,
    ) -> bool:
        """
        Clear sports data.

        If athlete_id is provided,
        clears only that athlete's data.
        """

        if athlete_id:

            if athlete_id not in self.athletes:
                return False

            self.performance_records[
                athlete_id
            ] = []

            self.training_sessions = [
                session
                for session
                in self.training_sessions
                if session["athlete_id"]
                != athlete_id
            ]

            self.matches = [
                match
                for match in self.matches
                if athlete_id
                not in match.get(
                    "athlete_ids",
                    [],
                )
            ]

            return True

        self.training_sessions.clear()
        self.matches.clear()

        for athlete in self.performance_records:

            self.performance_records[
                athlete
            ] = []

        return True

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return SportsManager status.
        """

        return {
            "enabled": self.enabled,
            "registered_sports": (
                self.list_sports()
            ),
            "athletes": len(
                self.athletes
            ),
            "training_sessions": len(
                self.training_sessions
            ),
            "matches": len(
                self.matches
            ),
            "performance_records": sum(
                len(records)
                for records
                in self.performance_records.values()
            ),
        }


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    manager = SportsManager()

    athlete_id = manager.create_athlete(
        name="Krishna",
        sport="cricket",
        age=15,
    )

    manager.record_training(
        athlete_id=athlete_id,
        training_type="Batting Practice",
        duration_minutes=90,
        metrics={
            "balls_faced": 200,
            "shots_practiced": 8,
        },
    )

    manager.record_performance(
        athlete_id=athlete_id,
        category="batting",
        metrics={
            "runs": 54,
            "balls": 42,
            "fours": 6,
            "sixes": 2,
        },
    )

    print(
        manager.get_athlete_summary(
            athlete_id
        )
    )

    print(
        manager.get_status()
    )


