"""
RENIX Batting Analysis
======================

Advanced batting performance analysis for cricket players.

Features:
- Strike rate analysis
- Batting average calculation
- Boundary percentage
- Dot ball percentage
- Scoring consistency
- Form analysis
- Strength and weakness detection
- Performance rating
- Recent trend analysis
"""

from __future__ import annotations

import logging
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional


logger = logging.getLogger(
    "RENIX.Sports.Cricket.BattingAnalysis"
)


class BattingAnalysis:
    """
    Analyze cricket batting performances.

    Expected innings format:

    {
        "runs": 54,
        "balls": 42,
        "fours": 6,
        "sixes": 2,
        "dismissal": "caught",
        "dot_balls": 12
    }
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.config = config or {}

        self.minimum_innings = int(
            self.config.get(
                "minimum_innings",
                3,
            )
        )

        logger.info(
            "BattingAnalysis initialized."
        )

    # ============================================================
    # BASIC CALCULATIONS
    # ============================================================

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
    def calculate_average(
        total_runs: int,
        dismissals: int,
    ) -> float:
        """
        Calculate batting average.
        """

        if dismissals <= 0:
            return float(total_runs)

        return round(
            total_runs / dismissals,
            2,
        )

    @staticmethod
    def calculate_boundary_runs(
        fours: int,
        sixes: int,
    ) -> int:
        """
        Calculate runs scored from boundaries.
        """

        return (
            fours * 4
            + sixes * 6
        )

    @staticmethod
    def calculate_boundary_percentage(
        runs: int,
        fours: int,
        sixes: int,
    ) -> float:
        """
        Calculate percentage of runs
        scored through boundaries.
        """

        if runs <= 0:
            return 0.0

        boundary_runs = (
            fours * 4
            + sixes * 6
        )

        return round(
            (boundary_runs / runs) * 100,
            2,
        )

    @staticmethod
    def calculate_dot_ball_percentage(
        balls: int,
        dot_balls: int,
    ) -> float:
        """
        Calculate percentage of dot balls.
        """

        if balls <= 0:
            return 0.0

        return round(
            (dot_balls / balls) * 100,
            2,
        )

    # ============================================================
    # SINGLE INNINGS ANALYSIS
    # ============================================================

    def analyze_innings(
        self,
        innings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze a single batting innings.
        """

        runs = int(
            innings.get("runs", 0)
        )

        balls = int(
            innings.get("balls", 0)
        )

        fours = int(
            innings.get("fours", 0)
        )

        sixes = int(
            innings.get("sixes", 0)
        )

        dot_balls = int(
            innings.get(
                "dot_balls",
                0,
            )
        )

        strike_rate = (
            self.calculate_strike_rate(
                runs,
                balls,
            )
        )

        boundary_runs = (
            self.calculate_boundary_runs(
                fours,
                sixes,
            )
        )

        boundary_percentage = (
            self.calculate_boundary_percentage(
                runs,
                fours,
                sixes,
            )
        )

        dot_ball_percentage = (
            self.calculate_dot_ball_percentage(
                balls,
                dot_balls,
            )
        )

        singles_and_other_runs = max(
            0,
            runs - boundary_runs,
        )

        analysis = {
            "runs": runs,
            "balls": balls,
            "fours": fours,
            "sixes": sixes,
            "strike_rate": strike_rate,
            "boundary_runs": boundary_runs,
            "boundary_percentage": (
                boundary_percentage
            ),
            "dot_balls": dot_balls,
            "dot_ball_percentage": (
                dot_ball_percentage
            ),
            "non_boundary_runs": (
                singles_and_other_runs
            ),
            "rating": (
                self.calculate_innings_rating(
                    runs=runs,
                    balls=balls,
                    fours=fours,
                    sixes=sixes,
                    dot_balls=dot_balls,
                )
            ),
        }

        analysis["assessment"] = (
            self.assess_innings(
                analysis
            )
        )

        return analysis

    # ============================================================
    # INNINGS RATING
    # ============================================================

    def calculate_innings_rating(
        self,
        runs: int,
        balls: int,
        fours: int,
        sixes: int,
        dot_balls: int = 0,
    ) -> float:
        """
        Calculate batting innings rating.

        Rating range:
        0 - 100
        """

        if balls <= 0:
            return 0.0

        score = 0.0

        strike_rate = (
            self.calculate_strike_rate(
                runs,
                balls,
            )
        )

        boundary_percentage = (
            self.calculate_boundary_percentage(
                runs,
                fours,
                sixes,
            )
        )

        dot_percentage = (
            self.calculate_dot_ball_percentage(
                balls,
                dot_balls,
            )
        )

        # Runs contribution
        score += min(
            runs * 0.8,
            40,
        )

        # Strike rate contribution
        if strike_rate >= 150:
            score += 25
        elif strike_rate >= 120:
            score += 20
        elif strike_rate >= 100:
            score += 15
        elif strike_rate >= 80:
            score += 10
        else:
            score += 5

        # Boundary contribution
        if boundary_percentage >= 50:
            score += 15
        elif boundary_percentage >= 35:
            score += 10
        elif boundary_percentage >= 20:
            score += 5

        # Rotation / dot-ball contribution
        if dot_percentage <= 20:
            score += 20
        elif dot_percentage <= 35:
            score += 15
        elif dot_percentage <= 50:
            score += 10
        else:
            score += 5

        return round(
            min(score, 100),
            2,
        )

    # ============================================================
    # INNINGS ASSESSMENT
    # ============================================================

    def assess_innings(
        self,
        analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate human-readable
        assessment of innings.
        """

        strengths = []
        improvements = []

        runs = analysis.get(
            "runs",
            0,
        )

        strike_rate = analysis.get(
            "strike_rate",
            0,
        )

        boundary_percentage = analysis.get(
            "boundary_percentage",
            0,
        )

        dot_percentage = analysis.get(
            "dot_ball_percentage",
            0,
        )

        rating = analysis.get(
            "rating",
            0,
        )

        if runs >= 50:
            strengths.append(
                "Strong run contribution"
            )
        elif runs >= 30:
            strengths.append(
                "Useful batting contribution"
            )

        if strike_rate >= 120:
            strengths.append(
                "Excellent scoring tempo"
            )
        elif strike_rate < 70:
            improvements.append(
                "Increase scoring tempo"
            )

        if boundary_percentage >= 40:
            strengths.append(
                "Good boundary scoring ability"
            )

        if dot_percentage <= 25:
            strengths.append(
                "Good strike rotation"
            )
        elif dot_percentage >= 50:
            improvements.append(
                "Reduce dot ball percentage"
            )

        if rating >= 80:
            overall = "excellent"
        elif rating >= 65:
            overall = "very_good"
        elif rating >= 50:
            overall = "good"
        elif rating >= 35:
            overall = "average"
        else:
            overall = "needs_improvement"

        return {
            "overall": overall,
            "strengths": strengths,
            "improvements": improvements,
        }

    # ============================================================
    # MULTIPLE INNINGS ANALYSIS
    # ============================================================

    def analyze_performances(
        self,
        innings_list: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Analyze multiple batting innings.
        """

        if not innings_list:

            return self._empty_analysis()

        total_runs = sum(
            int(
                innings.get(
                    "runs",
                    0,
                )
            )
            for innings in innings_list
        )

        total_balls = sum(
            int(
                innings.get(
                    "balls",
                    0,
                )
            )
            for innings in innings_list
        )

        total_fours = sum(
            int(
                innings.get(
                    "fours",
                    0,
                )
            )
            for innings in innings_list
        )

        total_sixes = sum(
            int(
                innings.get(
                    "sixes",
                    0,
                )
            )
            for innings in innings_list
        )

        total_dot_balls = sum(
            int(
                innings.get(
                    "dot_balls",
                    0,
                )
            )
            for innings in innings_list
        )

        dismissals = sum(
            1
            for innings in innings_list
            if innings.get(
                "dismissal"
            )
            not in (
                None,
                "",
                "not_out",
            )
        )

        highest_score = max(
            int(
                innings.get(
                    "runs",
                    0,
                )
            )
            for innings in innings_list
        )

        scores = [
            int(
                innings.get(
                    "runs",
                    0,
                )
            )
            for innings in innings_list
        ]

        average = (
            self.calculate_average(
                total_runs,
                dismissals,
            )
        )

        strike_rate = (
            self.calculate_strike_rate(
                total_runs,
                total_balls,
            )
        )

        boundary_percentage = (
            self.calculate_boundary_percentage(
                total_runs,
                total_fours,
                total_sixes,
            )
        )

        dot_ball_percentage = (
            self.calculate_dot_ball_percentage(
                total_balls,
                total_dot_balls,
            )
        )

        individual_analysis = [
            self.analyze_innings(
                innings
            )
            for innings in innings_list
        ]

        ratings = [
            analysis["rating"]
            for analysis
            in individual_analysis
        ]

        overall_rating = (
            round(
                mean(ratings),
                2,
            )
            if ratings
            else 0.0
        )

        consistency = (
            self.calculate_consistency(
                scores
            )
        )

        form = self.analyze_form(
            innings_list
        )

        return {
            "innings": len(
                innings_list
            ),
            "total_runs": total_runs,
            "total_balls": total_balls,
            "average": average,
            "strike_rate": strike_rate,
            "highest_score": highest_score,
            "fours": total_fours,
            "sixes": total_sixes,
            "boundary_percentage": (
                boundary_percentage
            ),
            "dot_ball_percentage": (
                dot_ball_percentage
            ),
            "consistency": consistency,
            "form": form,
            "overall_rating": overall_rating,
            "strengths": (
                self.identify_strengths(
                    innings_list
                )
            ),
            "improvements": (
                self.identify_improvements(
                    innings_list
                )
            ),
        }

    # ============================================================
    # CONSISTENCY ANALYSIS
    # ============================================================

    def calculate_consistency(
        self,
        scores: List[int],
    ) -> Dict[str, Any]:
        """
        Calculate batting consistency.

        Lower variation generally means
        greater consistency.
        """

        if not scores:

            return {
                "score": 0.0,
                "standard_deviation": 0.0,
                "classification": (
                    "insufficient_data"
                ),
            }

        if len(scores) == 1:

            return {
                "score": 50.0,
                "standard_deviation": 0.0,
                "classification": (
                    "insufficient_data"
                ),
            }

        avg_score = mean(
            scores
        )

        deviation = pstdev(
            scores
        )

        if avg_score <= 0:

            consistency_score = 0.0
        else:

            variation = (
                deviation
                / avg_score
            )

            consistency_score = max(
                0.0,
                min(
                    100.0,
                    100
                    - variation * 100,
                ),
            )

        if consistency_score >= 80:
            classification = "excellent"
        elif consistency_score >= 65:
            classification = "good"
        elif consistency_score >= 50:
            classification = "average"
        else:
            classification = (
                "inconsistent"
            )

        return {
            "score": round(
                consistency_score,
                2,
            ),
            "standard_deviation": round(
                deviation,
                2,
            ),
            "classification": classification,
        }

    # ============================================================
    # FORM ANALYSIS
    # ============================================================

    def analyze_form(
        self,
        innings_list: List[
            Dict[str, Any]
        ],
        recent_matches: int = 5,
    ) -> Dict[str, Any]:
        """
        Analyze recent batting form.
        """

        if not innings_list:

            return {
                "status": "no_data",
                "recent_average": 0.0,
                "previous_average": 0.0,
                "trend": "unknown",
            }

        recent = innings_list[
            -recent_matches:
        ]

        previous = innings_list[
            :-recent_matches
        ]

        recent_scores = [
            int(
                innings.get(
                    "runs",
                    0,
                )
            )
            for innings in recent
        ]

        recent_average = round(
            mean(recent_scores),
            2,
        )

        if previous:

            previous_scores = [
                int(
                    innings.get(
                        "runs",
                        0,
                    )
                )
                for innings in previous
            ]

            previous_average = round(
                mean(
                    previous_scores
                ),
                2,
            )

        else:

            previous_average = (
                recent_average
            )

        difference = (
            recent_average
            - previous_average
        )

        if difference >= 5:
            trend = "improving"
        elif difference <= -5:
            trend = "declining"
        else:
            trend = "stable"

        if recent_average >= 40:
            status = "excellent"
        elif recent_average >= 25:
            status = "good"
        elif recent_average >= 15:
            status = "average"
        else:
            status = "poor"

        return {
            "status": status,
            "recent_average": (
                recent_average
            ),
            "previous_average": (
                previous_average
            ),
            "difference": round(
                difference,
                2,
            ),
            "trend": trend,
            "recent_scores": (
                recent_scores
            ),
        }

    # ============================================================
    # STRENGTH ANALYSIS
    # ============================================================

    def identify_strengths(
        self,
        innings_list: List[
            Dict[str, Any]
        ],
    ) -> List[str]:
        """
        Identify batting strengths.
        """

        if not innings_list:
            return []

        strengths = []

        analysis = self.analyze_performance_metrics(
            innings_list
        )

        if analysis[
            "strike_rate"
        ] >= 120:

            strengths.append(
                "High scoring rate"
            )

        if analysis[
            "average"
        ] >= 35:

            strengths.append(
                "Strong batting average"
            )

        if analysis[
            "boundary_percentage"
        ] >= 40:

            strengths.append(
                "Boundary hitting ability"
            )

        if analysis[
            "dot_ball_percentage"
        ] <= 25:

            strengths.append(
                "Excellent strike rotation"
            )

        if analysis[
            "consistency_score"
        ] >= 70:

            strengths.append(
                "Consistent performances"
            )

        if not strengths:

            strengths.append(
                "Developing batting foundation"
            )

        return strengths

    # ============================================================
    # IMPROVEMENT ANALYSIS
    # ============================================================

    def identify_improvements(
        self,
        innings_list: List[
            Dict[str, Any]
        ],
    ) -> List[str]:
        """
        Identify areas requiring improvement.
        """

        if not innings_list:
            return [
                "More match data required"
            ]

        improvements = []

        analysis = self.analyze_performance_metrics(
            innings_list
        )

        if analysis[
            "strike_rate"
        ] < 80:

            improvements.append(
                "Improve scoring rate"
            )

        if analysis[
            "average"
        ] < 20:

            improvements.append(
                "Improve wicket protection"
            )

        if analysis[
            "dot_ball_percentage"
        ] > 45:

            improvements.append(
                "Reduce dot balls and rotate strike"
            )

        if analysis[
            "consistency_score"
        ] < 50:

            improvements.append(
                "Improve batting consistency"
            )

        if not improvements:

            improvements.append(
                "Maintain current batting performance"
            )

        return improvements

    # ============================================================
    # PERFORMANCE METRICS
    # ============================================================

    def analyze_performance_metrics(
        self,
        innings_list: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Calculate metrics used internally
        for strength and improvement analysis.
        """

        if not innings_list:

            return {
                "average": 0.0,
                "strike_rate": 0.0,
                "boundary_percentage": 0.0,
                "dot_ball_percentage": 0.0,
                "consistency_score": 0.0,
            }

        total_runs = sum(
            int(
                item.get(
                    "runs",
                    0,
                )
            )
            for item in innings_list
        )

        total_balls = sum(
            int(
                item.get(
                    "balls",
                    0,
                )
            )
            for item in innings_list
        )

        total_fours = sum(
            int(
                item.get(
                    "fours",
                    0,
                )
            )
            for item in innings_list
        )

        total_sixes = sum(
            int(
                item.get(
                    "sixes",
                    0,
                )
            )
            for item in innings_list
        )

        total_dot_balls = sum(
            int(
                item.get(
                    "dot_balls",
                    0,
                )
            )
            for item in innings_list
        )

        dismissals = sum(
            1
            for item in innings_list
            if item.get(
                "dismissal"
            )
            not in (
                None,
                "",
                "not_out",
            )
        )

        scores = [
            int(
                item.get(
                    "runs",
                    0,
                )
            )
            for item in innings_list
        ]

        consistency = (
            self.calculate_consistency(
                scores
            )
        )

        return {
            "average": (
                self.calculate_average(
                    total_runs,
                    dismissals,
                )
            ),
            "strike_rate": (
                self.calculate_strike_rate(
                    total_runs,
                    total_balls,
                )
            ),
            "boundary_percentage": (
                self.calculate_boundary_percentage(
                    total_runs,
                    total_fours,
                    total_sixes,
                )
            ),
            "dot_ball_percentage": (
                self.calculate_dot_ball_percentage(
                    total_balls,
                    total_dot_balls,
                )
            ),
            "consistency_score": (
                consistency["score"]
            ),
        }

    # ============================================================
    # PLAYER REPORT
    # ============================================================

    def generate_report(
        self,
        innings_list: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Generate complete batting report.
        """

        analysis = (
            self.analyze_performances(
                innings_list
            )
        )

        if (
            len(innings_list)
            < self.minimum_innings
        ):

            data_quality = (
                "limited_data"
            )
        else:

            data_quality = (
                "sufficient_data"
            )

        return {
            "data_quality": data_quality,
            "summary": analysis,
            "recommendations": (
                self.generate_recommendations(
                    analysis
                )
            ),
        }

    # ============================================================
    # RECOMMENDATIONS
    # ============================================================

    def generate_recommendations(
        self,
        analysis: Dict[str, Any],
    ) -> List[str]:
        """
        Generate batting recommendations.
        """

        recommendations = []

        strike_rate = analysis.get(
            "strike_rate",
            0,
        )

        dot_percentage = analysis.get(
            "dot_ball_percentage",
            0,
        )

        consistency = analysis.get(
            "consistency",
            {}
        ).get(
            "score",
            0,
        )

        form = analysis.get(
            "form",
            {}
        )

        if strike_rate < 80:

            recommendations.append(
                "Practice scoring singles and rotating strike."
            )

        if dot_percentage > 40:

            recommendations.append(
                "Work on reducing dot balls through better shot selection."
            )

        if consistency < 60:

            recommendations.append(
                "Focus on building consistent innings instead of relying only on big scores."
            )

        if form.get(
            "trend"
        ) == "declining":

            recommendations.append(
                "Review recent dismissals and identify recurring mistakes."
            )

        if not recommendations:

            recommendations.append(
                "Maintain current training and continue improving shot selection."
            )

        return recommendations

    # ============================================================
    # EMPTY ANALYSIS
    # ============================================================

    @staticmethod
    def _empty_analysis() -> Dict[str, Any]:
        """
        Return empty analysis structure.
        """

        return {
            "innings": 0,
            "total_runs": 0,
            "total_balls": 0,
            "average": 0.0,
            "strike_rate": 0.0,
            "highest_score": 0,
            "fours": 0,
            "sixes": 0,
            "boundary_percentage": 0.0,
            "dot_ball_percentage": 0.0,
            "consistency": {
                "score": 0.0,
                "standard_deviation": 0.0,
                "classification": (
                    "no_data"
                ),
            },
            "form": {
                "status": "no_data",
                "trend": "unknown",
            },
            "overall_rating": 0.0,
            "strengths": [],
            "improvements": [
                "More batting data required"
            ],
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

    analyzer = BattingAnalysis()

    innings = [
        {
            "runs": 35,
            "balls": 30,
            "fours": 4,
            "sixes": 1,
            "dot_balls": 8,
            "dismissal": "caught",
        },
        {
            "runs": 54,
            "balls": 42,
            "fours": 6,
            "sixes": 2,
            "dot_balls": 10,
            "dismissal": "bowled",
        },
        {
            "runs": 18,
            "balls": 22,
            "fours": 2,
            "sixes": 0,
            "dot_balls": 9,
            "dismissal": "caught",
        },
        {
            "runs": 47,
            "balls": 35,
            "fours": 5,
            "sixes": 2,
            "dot_balls": 7,
            "dismissal": "not_out",
        },
    ]

    report = analyzer.generate_report(
        innings
    )

    print("\nRENIX BATTING REPORT\n")

    for key, value in report.items():
        print(
            f"{key}: {value}"
        )


