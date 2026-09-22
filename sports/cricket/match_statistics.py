"""
RENIX Match Statistics
======================

Cricket match statistics and score analysis.

Features:
- Team scorecards
- Batting statistics
- Bowling statistics
- Run rate calculation
- Required run rate
- Partnership tracking
- Extras analysis
- Match summary
- Player of the match scoring
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional


logger = logging.getLogger(
    "RENIX.Sports.Cricket.MatchStatistics"
)


class MatchStatistics:
    """
    Cricket match statistics engine.

    Stores and analyzes innings, batting,
    bowling, extras, and overall match data.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.config = config or {}

        self.matches: Dict[
            str,
            Dict[str, Any]
        ] = {}

        logger.info(
            "MatchStatistics initialized."
        )

    # ============================================================
    # MATCH MANAGEMENT
    # ============================================================

    def create_match(
        self,
        match_id: str,
        team_a: str,
        team_b: str,
        match_type: str = "limited_overs",
        total_overs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Create a new match statistics record.
        """

        match = {
            "match_id": match_id,
            "team_a": team_a,
            "team_b": team_b,
            "match_type": match_type,
            "total_overs": total_overs,
            "innings": [],
            "result": None,
            "status": "created",
        }

        self.matches[match_id] = match

        return match

    def get_match(
        self,
        match_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve match information.
        """

        return self.matches.get(
            match_id
        )

    # ============================================================
    # INNINGS MANAGEMENT
    # ============================================================

    def create_innings(
        self,
        match_id: str,
        batting_team: str,
        bowling_team: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new innings.
        """

        match = self.matches.get(
            match_id
        )

        if match is None:
            return None

        innings = {
            "innings_number": (
                len(match["innings"])
                + 1
            ),
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "runs": 0,
            "wickets": 0,
            "balls": 0,
            "batting": [],
            "bowling": [],
            "extras": {
                "wides": 0,
                "no_balls": 0,
                "byes": 0,
                "leg_byes": 0,
                "penalty": 0,
            },
            "fall_of_wickets": [],
            "partnerships": [],
        }

        match["innings"].append(
            innings
        )

        return innings

    def get_innings(
        self,
        match_id: str,
        innings_number: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve innings by number.
        """

        match = self.matches.get(
            match_id
        )

        if match is None:
            return None

        for innings in match[
            "innings"
        ]:

            if (
                innings[
                    "innings_number"
                ]
                == innings_number
            ):
                return innings

        return None

    # ============================================================
    # BALL / SCORE UPDATES
    # ============================================================

    def add_runs(
        self,
        innings: Dict[str, Any],
        runs: int,
        legal_delivery: bool = True,
    ) -> None:
        """
        Add runs and optionally count a legal ball.
        """

        innings["runs"] += int(runs)

        if legal_delivery:

            innings["balls"] += 1

    def add_wicket(
        self,
        innings: Dict[str, Any],
        batter: str,
        score: Optional[int] = None,
        dismissal: str = "unknown",
    ) -> None:
        """
        Add a wicket to innings.
        """

        innings["wickets"] += 1

        wicket_score = (
            score
            if score is not None
            else innings["runs"]
        )

        innings[
            "fall_of_wickets"
        ].append(
            {
                "wicket_number": (
                    innings["wickets"]
                ),
                "batter": batter,
                "score": wicket_score,
                "dismissal": dismissal,
                "balls": innings["balls"],
            }
        )

    def add_extra(
        self,
        innings: Dict[str, Any],
        extra_type: str,
        runs: int,
    ) -> bool:
        """
        Add extras to innings.
        """

        extra_type = (
            extra_type.lower()
            .strip()
            .replace(
                " ",
                "_",
            )
        )

        if (
            extra_type
            not in innings["extras"]
        ):
            return False

        innings["extras"][
            extra_type
        ] += int(runs)

        innings["runs"] += int(runs)

        return True

    # ============================================================
    # BATTING SCORECARD
    # ============================================================

    def add_batting_entry(
        self,
        innings: Dict[str, Any],
        player: str,
        runs: int,
        balls: int,
        fours: int = 0,
        sixes: int = 0,
        dismissal: str = "not_out",
    ) -> Dict[str, Any]:
        """
        Add batting scorecard entry.
        """

        strike_rate = (
            self.calculate_strike_rate(
                runs,
                balls,
            )
        )

        entry = {
            "player": player,
            "runs": int(runs),
            "balls": int(balls),
            "fours": int(fours),
            "sixes": int(sixes),
            "dismissal": dismissal,
            "strike_rate": strike_rate,
        }

        innings["batting"].append(
            entry
        )

        return entry

    # ============================================================
    # BOWLING SCORECARD
    # ============================================================

    def add_bowling_entry(
        self,
        innings: Dict[str, Any],
        player: str,
        balls: int,
        runs_conceded: int,
        wickets: int = 0,
        maidens: int = 0,
        wides: int = 0,
        no_balls: int = 0,
    ) -> Dict[str, Any]:
        """
        Add bowling scorecard entry.
        """

        overs = (
            self.convert_balls_to_overs(
                balls
            )
        )

        economy = (
            self.calculate_economy(
                balls,
                runs_conceded,
            )
        )

        entry = {
            "player": player,
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
        }

        innings["bowling"].append(
            entry
        )

        return entry

    # ============================================================
    # BASIC CALCULATIONS
    # ============================================================

    @staticmethod
    def convert_balls_to_overs(
        balls: int,
    ) -> str:
        """
        Convert total legal balls into
        cricket overs notation.

        Example:
        17 balls -> 2.5 overs
        """

        if balls <= 0:
            return "0.0"

        complete_overs = balls // 6

        remaining_balls = balls % 6

        return (
            f"{complete_overs}."
            f"{remaining_balls}"
        )

    @staticmethod
    def calculate_run_rate(
        runs: int,
        balls: int,
    ) -> float:
        """
        Calculate current run rate.
        """

        if balls <= 0:
            return 0.0

        overs = balls / 6

        return round(
            runs / overs,
            2,
        )

    @staticmethod
    def calculate_strike_rate(
        runs: int,
        balls: int,
    ) -> float:
        """
        Calculate batting strike rate.
        """

        if balls <= 0:
            return 0.0

        return round(
            (runs / balls) * 100,
            2,
        )

    @staticmethod
    def calculate_economy(
        balls: int,
        runs: int,
    ) -> float:
        """
        Calculate bowling economy.
        """

        if balls <= 0:
            return 0.0

        overs = balls / 6

        return round(
            runs / overs,
            2,
        )

    @staticmethod
    def calculate_required_run_rate(
        runs_required: int,
        balls_remaining: int,
    ) -> float:
        """
        Calculate required run rate.
        """

        if balls_remaining <= 0:
            return 0.0

        overs_remaining = (
            balls_remaining / 6
        )

        return round(
            runs_required
            / overs_remaining,
            2,
        )

    # ============================================================
    # PARTNERSHIPS
    # ============================================================

    def add_partnership(
        self,
        innings: Dict[str, Any],
        batter_one: str,
        batter_two: str,
        runs: int,
        balls: int,
    ) -> Dict[str, Any]:
        """
        Add batting partnership.
        """

        partnership = {
            "batter_one": batter_one,
            "batter_two": batter_two,
            "runs": int(runs),
            "balls": int(balls),
            "run_rate": (
                self.calculate_run_rate(
                    runs,
                    balls,
                )
            ),
        }

        innings[
            "partnerships"
        ].append(
            partnership
        )

        return partnership

    def get_best_partnership(
        self,
        innings: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Return highest run partnership.
        """

        partnerships = innings.get(
            "partnerships",
            [],
        )

        if not partnerships:
            return None

        return max(
            partnerships,
            key=lambda item: item["runs"],
        )

    # ============================================================
    # EXTRAS ANALYSIS
    # ============================================================

    @staticmethod
    def get_total_extras(
        innings: Dict[str, Any],
    ) -> int:
        """
        Calculate total extras.
        """

        extras = innings.get(
            "extras",
            {},
        )

        return sum(
            int(value)
            for value in extras.values()
        )

    def get_extras_analysis(
        self,
        innings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze extras.
        """

        extras = innings.get(
            "extras",
            {},
        )

        total = self.get_total_extras(
            innings
        )

        return {
            "total": total,
            "wides": extras.get(
                "wides",
                0,
            ),
            "no_balls": extras.get(
                "no_balls",
                0,
            ),
            "byes": extras.get(
                "byes",
                0,
            ),
            "leg_byes": extras.get(
                "leg_byes",
                0,
            ),
            "penalty": extras.get(
                "penalty",
                0,
            ),
        }

    # ============================================================
    # BATTING LEADERBOARD
    # ============================================================

    @staticmethod
    def get_top_batters(
        innings: Dict[str, Any],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Return highest run scorers.
        """

        batting = innings.get(
            "batting",
            [],
        )

        sorted_batting = sorted(
            batting,
            key=lambda item: (
                item["runs"],
                item["strike_rate"],
            ),
            reverse=True,
        )

        return sorted_batting[:limit]

    # ============================================================
    # BOWLING LEADERBOARD
    # ============================================================

    @staticmethod
    def get_top_bowlers(
        innings: Dict[str, Any],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Return best bowlers.

        Priority:
        1. Wickets
        2. Economy
        """

        bowling = innings.get(
            "bowling",
            [],
        )

        sorted_bowling = sorted(
            bowling,
            key=lambda item: (
                item["wickets"],
                -item["economy"],
            ),
            reverse=True,
        )

        return sorted_bowling[:limit]

    # ============================================================
    # INNINGS SUMMARY
    # ============================================================

    def generate_innings_summary(
        self,
        innings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate complete innings summary.
        """

        runs = innings.get(
            "runs",
            0,
        )

        wickets = innings.get(
            "wickets",
            0,
        )

        balls = innings.get(
            "balls",
            0,
        )

        return {
            "innings_number": innings.get(
                "innings_number"
            ),
            "batting_team": innings.get(
                "batting_team"
            ),
            "bowling_team": innings.get(
                "bowling_team"
            ),
            "score": (
                f"{runs}/{wickets}"
            ),
            "runs": runs,
            "wickets": wickets,
            "balls": balls,
            "overs": (
                self.convert_balls_to_overs(
                    balls
                )
            ),
            "run_rate": (
                self.calculate_run_rate(
                    runs,
                    balls,
                )
            ),
            "extras": (
                self.get_extras_analysis(
                    innings
                )
            ),
            "top_batters": (
                self.get_top_batters(
                    innings
                )
            ),
            "top_bowlers": (
                self.get_top_bowlers(
                    innings
                )
            ),
            "best_partnership": (
                self.get_best_partnership(
                    innings
                )
            ),
        }

    # ============================================================
    # MATCH RESULT
    # ============================================================

    def set_match_result(
        self,
        match_id: str,
        winner: Optional[str],
        result_text: str,
    ) -> bool:
        """
        Set match result.
        """

        match = self.matches.get(
            match_id
        )

        if match is None:
            return False

        match["result"] = {
            "winner": winner,
            "description": result_text,
        }

        match["status"] = "completed"

        return True

    # ============================================================
    # MATCH SUMMARY
    # ============================================================

    def generate_match_summary(
        self,
        match_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate complete match summary.
        """

        match = self.matches.get(
            match_id
        )

        if match is None:
            return None

        innings_summaries = [
            self.generate_innings_summary(
                innings
            )
            for innings
            in match.get(
                "innings",
                [],
            )
        ]

        return {
            "match_id": match_id,
            "teams": {
                "team_a": match.get(
                    "team_a"
                ),
                "team_b": match.get(
                    "team_b"
                ),
            },
            "match_type": match.get(
                "match_type"
            ),
            "total_overs": match.get(
                "total_overs"
            ),
            "status": match.get(
                "status"
            ),
            "innings": innings_summaries,
            "result": match.get(
                "result"
            ),
        }

    # ============================================================
    # PLAYER OF THE MATCH
    # ============================================================

    def calculate_player_points(
        self,
        batting: Optional[
            Dict[str, Any]
        ] = None,
        bowling: Optional[
            Dict[str, Any]
        ] = None,
        fielding: Optional[
            Dict[str, Any]
        ] = None,
    ) -> float:
        """
        Calculate player match impact score.

        This can be used as one factor
        for Player of the Match selection.
        """

        points = 0.0

        if batting:

            runs = batting.get(
                "runs",
                0,
            )

            strike_rate = batting.get(
                "strike_rate",
                0,
            )

            points += runs

            if strike_rate >= 150:
                points += 15
            elif strike_rate >= 120:
                points += 10
            elif strike_rate >= 100:
                points += 5

            points += (
                batting.get(
                    "fours",
                    0,
                )
                * 1
            )

            points += (
                batting.get(
                    "sixes",
                    0,
                )
                * 2
            )

        if bowling:

            wickets = bowling.get(
                "wickets",
                0,
            )

            economy = bowling.get(
                "economy",
                0,
            )

            points += wickets * 20

            if (
                wickets > 0
                and economy <= 5
            ):
                points += 10

            points += (
                bowling.get(
                    "maidens",
                    0,
                )
                * 8
            )

        if fielding:

            points += (
                fielding.get(
                    "catches",
                    0,
                )
                * 8
            )

            points += (
                fielding.get(
                    "run_outs",
                    0,
                )
                * 10
            )

            points += (
                fielding.get(
                    "stumpings",
                    0,
                )
                * 8
            )

        return round(
            points,
            2,
        )

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return MatchStatistics status.
        """

        return {
            "total_matches": len(
                self.matches
            ),
            "completed_matches": sum(
                1
                for match
                in self.matches.values()
                if match.get("status")
                == "completed"
            ),
            "active_matches": sum(
                1
                for match
                in self.matches.values()
                if match.get("status")
                != "completed"
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

    stats = MatchStatistics()

    stats.create_match(
        match_id="match_001",
        team_a="Mumbai Warriors",
        team_b="Pune Titans",
        match_type="T20",
        total_overs=20,
    )

    innings = stats.create_innings(
        match_id="match_001",
        batting_team="Mumbai Warriors",
        bowling_team="Pune Titans",
    )

    if innings:

        stats.add_runs(
            innings,
            runs=150,
            legal_delivery=False,
        )

        innings["balls"] = 120
        innings["wickets"] = 6

        stats.add_batting_entry(
            innings,
            player="Krishna",
            runs=54,
            balls=42,
            fours=6,
            sixes=2,
            dismissal="caught",
        )

        stats.add_batting_entry(
            innings,
            player="Player 2",
            runs=45,
            balls=31,
            fours=4,
            sixes=3,
            dismissal="bowled",
        )

        stats.add_bowling_entry(
            innings,
            player="Bowler 1",
            balls=24,
            runs_conceded=28,
            wickets=2,
        )

        stats.add_partnership(
            innings,
            batter_one="Krishna",
            batter_two="Player 2",
            runs=75,
            balls=48,
        )

    stats.set_match_result(
        match_id="match_001",
        winner="Mumbai Warriors",
        result_text=(
            "Mumbai Warriors won by 15 runs"
        ),
    )

    summary = (
        stats.generate_match_summary(
            "match_001"
        )
    )

    print(
        "\nRENIX MATCH SUMMARY\n"
    )

    print(summary)


