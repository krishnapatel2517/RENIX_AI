"""
RENIX Cricket Manager
=====================

Central cricket management system for RENIX.

Responsibilities:
- Manage player cricket profiles
- Record batting, bowling and fielding performances
- Manage matches and innings
- Calculate basic cricket statistics
- Connect cricket analysis modules
- Track player progress
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


logger = logging.getLogger("RENIX.Sports.Cricket.Manager")


class CricketManager:
    """
    Central manager for all cricket-related features.

    This class can work with:
    - BattingAnalysis
    - BowlingAnalysis
    - FieldingAnalysis
    - MatchStatistics
    - VideoAnalysis
    - ShotDetection
    - Training
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.config = config or {}

        self.players: Dict[
            str,
            Dict[str, Any]
        ] = {}

        self.matches: Dict[
            str,
            Dict[str, Any]
        ] = {}

        self.performances: Dict[
            str,
            List[Dict[str, Any]]
        ] = {}

        self.training_sessions: Dict[
            str,
            List[Dict[str, Any]]
        ] = {}

        self.modules: Dict[
            str,
            Any
        ] = {}

        self.max_records = int(
            self.config.get(
                "max_records",
                5000,
            )
        )

        logger.info(
            "CricketManager initialized."
        )

    # ============================================================
    # MODULE MANAGEMENT
    # ============================================================

    def register_module(
        self,
        name: str,
        module: Any,
    ) -> bool:
        """
        Register a cricket analysis module.
        """

        if not name:
            return False

        key = name.lower().strip()

        self.modules[key] = module

        logger.info(
            "Cricket module registered: %s",
            key,
        )

        return True

    def get_module(
        self,
        name: str,
    ) -> Optional[Any]:
        """
        Get a registered cricket module.
        """

        return self.modules.get(
            name.lower().strip()
        )

    def list_modules(
        self,
    ) -> List[str]:
        """
        Return all registered modules.
        """

        return list(
            self.modules.keys()
        )

    # ============================================================
    # PLAYER MANAGEMENT
    # ============================================================

    def create_player(
        self,
        name: str,
        age: Optional[int] = None,
        role: str = "all_rounder",
        batting_style: str = "right_hand",
        bowling_style: Optional[
            str
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> str:
        """
        Create a cricket player profile.

        Returns:
            Player ID
        """

        player_id = str(
            uuid.uuid4()
        )

        player = {
            "id": player_id,
            "name": name,
            "age": age,
            "role": role,
            "batting_style": batting_style,
            "bowling_style": bowling_style,
            "metadata": metadata or {},
            "created_at": (
                datetime.now().isoformat()
            ),
            "updated_at": (
                datetime.now().isoformat()
            ),
        }

        self.players[
            player_id
        ] = player

        self.performances[
            player_id
        ] = []

        self.training_sessions[
            player_id
        ] = []

        logger.info(
            "Cricket player created: %s",
            name,
        )

        return player_id

    def get_player(
        self,
        player_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get player profile.
        """

        player = self.players.get(
            player_id
        )

        if player is None:
            return None

        return player.copy()

    def update_player(
        self,
        player_id: str,
        **updates: Any,
    ) -> bool:
        """
        Update player information.
        """

        player = self.players.get(
            player_id
        )

        if player is None:
            return False

        protected = {
            "id",
            "created_at",
        }

        for key, value in updates.items():

            if key not in protected:

                player[key] = value

        player["updated_at"] = (
            datetime.now().isoformat()
        )

        return True

    def delete_player(
        self,
        player_id: str,
    ) -> bool:
        """
        Delete player and associated data.
        """

        if player_id not in self.players:
            return False

        del self.players[
            player_id
        ]

        self.performances.pop(
            player_id,
            None,
        )

        self.training_sessions.pop(
            player_id,
            None,
        )

        logger.info(
            "Player deleted: %s",
            player_id,
        )

        return True

    def list_players(
        self,
        role: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return all registered players.

        Optionally filter by role.
        """

        players = list(
            self.players.values()
        )

        if role:

            role = role.lower()

            players = [
                player
                for player in players
                if player.get("role", "").lower()
                == role
            ]

        return [
            player.copy()
            for player in players
        ]

    # ============================================================
    # MATCH MANAGEMENT
    # ============================================================

    def create_match(
        self,
        team_a: str,
        team_b: str,
        match_type: str = "limited_overs",
        overs: Optional[int] = None,
        location: Optional[str] = None,
    ) -> str:
        """
        Create a cricket match.

        Returns:
            Match ID
        """

        match_id = str(
            uuid.uuid4()
        )

        match = {
            "id": match_id,
            "team_a": team_a,
            "team_b": team_b,
            "match_type": match_type,
            "overs": overs,
            "location": location,
            "status": "created",
            "innings": [],
            "created_at": (
                datetime.now().isoformat()
            ),
            "updated_at": (
                datetime.now().isoformat()
            ),
        }

        self.matches[
            match_id
        ] = match

        logger.info(
            "Match created: %s vs %s",
            team_a,
            team_b,
        )

        return match_id

    def get_match(
        self,
        match_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get match data.
        """

        match = self.matches.get(
            match_id
        )

        if match is None:
            return None

        return match

    def update_match_status(
        self,
        match_id: str,
        status: str,
    ) -> bool:
        """
        Update match status.
        """

        match = self.matches.get(
            match_id
        )

        if match is None:
            return False

        match["status"] = status

        match["updated_at"] = (
            datetime.now().isoformat()
        )

        return True

    # ============================================================
    # INNINGS MANAGEMENT
    # ============================================================

    def add_innings(
        self,
        match_id: str,
        batting_team: str,
        bowling_team: str,
    ) -> Optional[str]:
        """
        Add innings to a match.

        Returns:
            Innings ID
        """

        match = self.matches.get(
            match_id
        )

        if match is None:
            return None

        innings_id = str(
            uuid.uuid4()
        )

        innings = {
            "id": innings_id,
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "runs": 0,
            "wickets": 0,
            "overs": 0.0,
            "balls": 0,
            "batting": [],
            "bowling": [],
            "extras": {
                "wides": 0,
                "no_balls": 0,
                "byes": 0,
                "leg_byes": 0,
            },
            "created_at": (
                datetime.now().isoformat()
            ),
        }

        match["innings"].append(
            innings
        )

        return innings_id

    def get_innings(
        self,
        match_id: str,
        innings_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve innings data.
        """

        match = self.matches.get(
            match_id
        )

        if match is None:
            return None

        for innings in match[
            "innings"
        ]:

            if innings["id"] == innings_id:
                return innings

        return None

    # ============================================================
    # BATTING PERFORMANCE
    # ============================================================

    def record_batting(
        self,
        player_id: str,
        runs: int,
        balls: int,
        fours: int = 0,
        sixes: int = 0,
        dismissal: Optional[str] = None,
        match_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Record batting performance.
        """

        if player_id not in self.players:
            return None

        performance_id = str(
            uuid.uuid4()
        )

        strike_rate = 0.0

        if balls > 0:

            strike_rate = round(
                (runs / balls) * 100,
                2,
            )

        performance = {
            "id": performance_id,
            "type": "batting",
            "player_id": player_id,
            "match_id": match_id,
            "runs": int(runs),
            "balls": int(balls),
            "fours": int(fours),
            "sixes": int(sixes),
            "strike_rate": strike_rate,
            "dismissal": dismissal,
            "timestamp": (
                datetime.now().isoformat()
            ),
        }

        self.performances[
            player_id
        ].append(
            performance
        )

        self._limit_records(
            self.performances[
                player_id
            ]
        )

        return performance_id

    # ============================================================
    # BOWLING PERFORMANCE
    # ============================================================

    def record_bowling(
        self,
        player_id: str,
        balls: int,
        runs_conceded: int,
        wickets: int = 0,
        maidens: int = 0,
        wides: int = 0,
        no_balls: int = 0,
        match_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Record bowling performance.
        """

        if player_id not in self.players:
            return None

        performance_id = str(
            uuid.uuid4()
        )

        overs = round(
            balls / 6,
            2,
        )

        economy = 0.0

        if balls > 0:

            economy = round(
                runs_conceded
                / (balls / 6),
                2,
            )

        performance = {
            "id": performance_id,
            "type": "bowling",
            "player_id": player_id,
            "match_id": match_id,
            "balls": int(balls),
            "overs": overs,
            "runs_conceded": int(
                runs_conceded
            ),
            "wickets": int(wickets),
            "maidens": int(maidens),
            "wides": int(wides),
            "no_balls": int(no_balls),
            "economy": economy,
            "timestamp": (
                datetime.now().isoformat()
            ),
        }

        self.performances[
            player_id
        ].append(
            performance
        )

        self._limit_records(
            self.performances[
                player_id
            ]
        )

        return performance_id

    # ============================================================
    # FIELDING PERFORMANCE
    # ============================================================

    def record_fielding(
        self,
        player_id: str,
        catches: int = 0,
        run_outs: int = 0,
        stumpings: int = 0,
        dropped_catches: int = 0,
        match_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Record fielding performance.
        """

        if player_id not in self.players:
            return None

        performance_id = str(
            uuid.uuid4()
        )

        performance = {
            "id": performance_id,
            "type": "fielding",
            "player_id": player_id,
            "match_id": match_id,
            "catches": int(catches),
            "run_outs": int(run_outs),
            "stumpings": int(stumpings),
            "dropped_catches": int(
                dropped_catches
            ),
            "timestamp": (
                datetime.now().isoformat()
            ),
        }

        self.performances[
            player_id
        ].append(
            performance
        )

        self._limit_records(
            self.performances[
                player_id
            ]
        )

        return performance_id

    # ============================================================
    # PERFORMANCE RETRIEVAL
    # ============================================================

    def get_player_performances(
        self,
        player_id: str,
        performance_type: Optional[
            str
        ] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get player performance history.
        """

        records = (
            self.performances.get(
                player_id,
                [],
            )
        )

        if performance_type:

            performance_type = (
                performance_type.lower()
            )

            records = [
                record
                for record in records
                if record["type"]
                == performance_type
            ]

        return records[-limit:]

    # ============================================================
    # PLAYER STATISTICS
    # ============================================================

    def get_batting_statistics(
        self,
        player_id: str,
    ) -> Dict[str, Any]:
        """
        Calculate batting statistics.
        """

        records = (
            self.get_player_performances(
                player_id,
                performance_type="batting",
                limit=self.max_records,
            )
        )

        if not records:

            return {
                "innings": 0,
                "runs": 0,
                "balls": 0,
                "average": 0.0,
                "strike_rate": 0.0,
                "fours": 0,
                "sixes": 0,
                "highest_score": 0,
            }

        total_runs = sum(
            record["runs"]
            for record in records
        )

        total_balls = sum(
            record["balls"]
            for record in records
        )

        total_fours = sum(
            record["fours"]
            for record in records
        )

        total_sixes = sum(
            record["sixes"]
            for record in records
        )

        highest_score = max(
            record["runs"]
            for record in records
        )

        dismissals = sum(
            1
            for record in records
            if record.get("dismissal")
            not in (
                None,
                "",
                "not_out",
            )
        )

        average = 0.0

        if dismissals > 0:

            average = round(
                total_runs / dismissals,
                2,
            )

        strike_rate = 0.0

        if total_balls > 0:

            strike_rate = round(
                (total_runs / total_balls)
                * 100,
                2,
            )

        return {
            "innings": len(records),
            "runs": total_runs,
            "balls": total_balls,
            "average": average,
            "strike_rate": strike_rate,
            "fours": total_fours,
            "sixes": total_sixes,
            "highest_score": highest_score,
            "dismissals": dismissals,
        }

    def get_bowling_statistics(
        self,
        player_id: str,
    ) -> Dict[str, Any]:
        """
        Calculate bowling statistics.
        """

        records = (
            self.get_player_performances(
                player_id,
                performance_type="bowling",
                limit=self.max_records,
            )
        )

        if not records:

            return {
                "innings": 0,
                "balls": 0,
                "runs_conceded": 0,
                "wickets": 0,
                "economy": 0.0,
                "average": 0.0,
                "best_wickets": 0,
            }

        total_balls = sum(
            record["balls"]
            for record in records
        )

        total_runs = sum(
            record["runs_conceded"]
            for record in records
        )

        total_wickets = sum(
            record["wickets"]
            for record in records
        )

        best_wickets = max(
            record["wickets"]
            for record in records
        )

        economy = 0.0

        if total_balls > 0:

            economy = round(
                total_runs
                / (total_balls / 6),
                2,
            )

        average = 0.0

        if total_wickets > 0:

            average = round(
                total_runs
                / total_wickets,
                2,
            )

        return {
            "innings": len(records),
            "balls": total_balls,
            "runs_conceded": total_runs,
            "wickets": total_wickets,
            "economy": economy,
            "average": average,
            "best_wickets": best_wickets,
        }

    def get_fielding_statistics(
        self,
        player_id: str,
    ) -> Dict[str, Any]:
        """
        Calculate fielding statistics.
        """

        records = (
            self.get_player_performances(
                player_id,
                performance_type="fielding",
                limit=self.max_records,
            )
        )

        return {
            "matches": len(records),
            "catches": sum(
                record["catches"]
                for record in records
            ),
            "run_outs": sum(
                record["run_outs"]
                for record in records
            ),
            "stumpings": sum(
                record["stumpings"]
                for record in records
            ),
            "dropped_catches": sum(
                record["dropped_catches"]
                for record in records
            ),
        }

    # ============================================================
    # TRAINING
    # ============================================================

    def record_training(
        self,
        player_id: str,
        training_type: str,
        duration_minutes: float,
        drills: Optional[
            List[str]
        ] = None,
        notes: str = "",
        metrics: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Optional[str]:
        """
        Record cricket training session.
        """

        if player_id not in self.players:
            return None

        training_id = str(
            uuid.uuid4()
        )

        session = {
            "id": training_id,
            "player_id": player_id,
            "training_type": training_type,
            "duration_minutes": float(
                duration_minutes
            ),
            "drills": drills or [],
            "notes": notes,
            "metrics": metrics or {},
            "timestamp": (
                datetime.now().isoformat()
            ),
        }

        self.training_sessions[
            player_id
        ].append(
            session
        )

        self._limit_records(
            self.training_sessions[
                player_id
            ]
        )

        return training_id

    def get_training_history(
        self,
        player_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get player training history.
        """

        records = (
            self.training_sessions.get(
                player_id,
                [],
            )
        )

        return records[-limit:]

    # ============================================================
    # PLAYER SUMMARY
    # ============================================================

    def get_player_summary(
        self,
        player_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate complete player summary.
        """

        player = self.get_player(
            player_id
        )

        if player is None:
            return None

        batting = (
            self.get_batting_statistics(
                player_id
            )
        )

        bowling = (
            self.get_bowling_statistics(
                player_id
            )
        )

        fielding = (
            self.get_fielding_statistics(
                player_id
            )
        )

        training = (
            self.get_training_history(
                player_id,
                limit=self.max_records,
            )
        )

        return {
            "player": player,
            "batting": batting,
            "bowling": bowling,
            "fielding": fielding,
            "training_sessions": len(
                training
            ),
            "total_training_minutes": sum(
                session.get(
                    "duration_minutes",
                    0,
                )
                for session in training
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
        Limit stored records.
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

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return CricketManager status.
        """

        return {
            "players": len(
                self.players
            ),
            "matches": len(
                self.matches
            ),
            "registered_modules": (
                self.list_modules()
            ),
            "total_performance_records": sum(
                len(records)
                for records
                in self.performances.values()
            ),
            "total_training_sessions": sum(
                len(records)
                for records
                in self.training_sessions.values()
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

    cricket = CricketManager()

    player_id = cricket.create_player(
        name="Krishna",
        age=15,
        role="batsman",
        batting_style="right_hand",
    )

    cricket.record_batting(
        player_id=player_id,
        runs=54,
        balls=42,
        fours=6,
        sixes=2,
        dismissal="caught",
    )

    cricket.record_bowling(
        player_id=player_id,
        balls=24,
        runs_conceded=28,
        wickets=2,
    )

    cricket.record_fielding(
        player_id=player_id,
        catches=1,
        run_outs=1,
    )

    cricket.record_training(
        player_id=player_id,
        training_type="Batting",
        duration_minutes=90,
        drills=[
            "Throwdowns",
            "Front foot defense",
            "Cover drive",
        ],
    )

    print("\nPLAYER SUMMARY\n")

    summary = cricket.get_player_summary(
        player_id
    )

    for key, value in summary.items():

        print(
            f"{key}: {value}"
        )

    print("\nSTATUS\n")

    print(
        cricket.get_status()
    )


