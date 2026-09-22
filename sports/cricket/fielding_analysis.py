"""
RENIX Fielding Analysis
=======================

Fielding performance analysis for cricket players.

Features:
- Catch statistics
- Catch success rate
- Dropped catch analysis
- Run-out statistics
- Stumping statistics
- Fielding efficiency rating
- Consistency analysis
- Strength identification
- Improvement recommendations
- Overall fielding reports
"""

from __future__ import annotations

import logging
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional


logger = logging.getLogger(
    "RENIX.Sports.Cricket.FieldingAnalysis"
)


class FieldingAnalysis:
    """
    Analyze cricket fielding performances.

    Expected match format:

    {
        "catches": 2,
        "catch_chances": 3,
        "dropped_catches": 1,
        "run_outs": 1,
        "run_out_attempts": 2,
        "stumpings": 0,
        "direct_hits": 2,
        "misfields": 1,
        "stops": 8
    }
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.config = config or {}

        self.minimum_matches = int(
            self.config.get(
                "minimum_matches",
                3,
            )
        )

        logger.info(
            "FieldingAnalysis initialized."
        )

    # ============================================================
    # BASIC CALCULATIONS
    # ============================================================

    @staticmethod
    def calculate_percentage(
        successful: int,
        attempts: int,
    ) -> float:
        """
        Calculate success percentage.
        """

        if attempts <= 0:
            return 0.0

        return round(
            (successful / attempts) * 100,
            2,
        )

    @staticmethod
    def calculate_catch_success_rate(
        catches: int,
        chances: int,
    ) -> float:
        """
        Calculate catch success rate.
        """

        return FieldingAnalysis.calculate_percentage(
            catches,
            chances,
        )

    @staticmethod
    def calculate_run_out_success_rate(
        run_outs: int,
        attempts: int,
    ) -> float:
        """
        Calculate run-out success rate.
        """

        return FieldingAnalysis.calculate_percentage(
            run_outs,
            attempts,
        )

    @staticmethod
    def calculate_error_rate(
        errors: int,
        opportunities: int,
    ) -> float:
        """
        Calculate fielding error rate.
        """

        if opportunities <= 0:
            return 0.0

        return round(
            (errors / opportunities) * 100,
            2,
        )

    # ============================================================
    # SINGLE MATCH ANALYSIS
    # ============================================================

    def analyze_match(
        self,
        performance: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze a single fielding performance.
        """

        catches = int(
            performance.get(
                "catches",
                0,
            )
        )

        catch_chances = int(
            performance.get(
                "catch_chances",
                catches,
            )
        )

        dropped_catches = int(
            performance.get(
                "dropped_catches",
                0,
            )
        )

        run_outs = int(
            performance.get(
                "run_outs",
                0,
            )
        )

        run_out_attempts = int(
            performance.get(
                "run_out_attempts",
                run_outs,
            )
        )

        stumpings = int(
            performance.get(
                "stumpings",
                0,
            )
        )

        direct_hits = int(
            performance.get(
                "direct_hits",
                0,
            )
        )

        misfields = int(
            performance.get(
                "misfields",
                0,
            )
        )

        stops = int(
            performance.get(
                "stops",
                0,
            )
        )

        catch_success_rate = (
            self.calculate_catch_success_rate(
                catches,
                catch_chances,
            )
        )

        run_out_success_rate = (
            self.calculate_run_out_success_rate(
                run_outs,
                run_out_attempts,
            )
        )

        opportunities = (
            catch_chances
            + run_out_attempts
            + stops
        )

        errors = (
            dropped_catches
            + misfields
        )

        error_rate = (
            self.calculate_error_rate(
                errors,
                opportunities,
            )
        )

        rating = (
            self.calculate_match_rating(
                catches=catches,
                catch_chances=catch_chances,
                run_outs=run_outs,
                run_out_attempts=(
                    run_out_attempts
                ),
                stumpings=stumpings,
                direct_hits=direct_hits,
                stops=stops,
                errors=errors,
            )
        )

        analysis = {
            "catches": catches,
            "catch_chances": catch_chances,
            "dropped_catches": dropped_catches,
            "catch_success_rate": (
                catch_success_rate
            ),
            "run_outs": run_outs,
            "run_out_attempts": (
                run_out_attempts
            ),
            "run_out_success_rate": (
                run_out_success_rate
            ),
            "stumpings": stumpings,
            "direct_hits": direct_hits,
            "stops": stops,
            "misfields": misfields,
            "error_rate": error_rate,
            "rating": rating,
        }

        analysis["assessment"] = (
            self.assess_match(
                analysis
            )
        )

        return analysis

    # ============================================================
    # FIELDING RATING
    # ============================================================

    def calculate_match_rating(
        self,
        catches: int,
        catch_chances: int,
        run_outs: int,
        run_out_attempts: int,
        stumpings: int,
        direct_hits: int,
        stops: int,
        errors: int,
    ) -> float:
        """
        Calculate overall fielding rating.

        Rating range:
        0 - 100
        """

        score = 0.0

        catch_success = (
            self.calculate_catch_success_rate(
                catches,
                catch_chances,
            )
        )

        run_out_success = (
            self.calculate_run_out_success_rate(
                run_outs,
                run_out_attempts,
            )
        )

        # Catching contribution
        if catch_chances > 0:

            score += (
                catch_success * 0.25
            )

        # Run-out contribution
        if run_out_attempts > 0:

            score += (
                run_out_success * 0.15
            )

        # Successful catches
        score += min(
            catches * 8,
            20,
        )

        # Run-outs
        score += min(
            run_outs * 10,
            20,
        )

        # Stumpings
        score += min(
            stumpings * 8,
            15,
        )

        # Direct hits
        score += min(
            direct_hits * 4,
            10,
        )

        # Stops
        score += min(
            stops * 1.5,
            10,
        )

        # Error penalty
        score -= errors * 8

        return round(
            max(
                0.0,
                min(
                    score,
                    100.0,
                ),
            ),
            2,
        )

    # ============================================================
    # MATCH ASSESSMENT
    # ============================================================

    def assess_match(
        self,
        analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate fielding assessment.
        """

        strengths = []
        improvements = []

        rating = analysis.get(
            "rating",
            0,
        )

        catch_success = analysis.get(
            "catch_success_rate",
            0,
        )

        run_out_success = analysis.get(
            "run_out_success_rate",
            0,
        )

        errors = (
            analysis.get(
                "dropped_catches",
                0,
            )
            + analysis.get(
                "misfields",
                0,
            )
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

        if (
            catch_success >= 90
            and analysis.get(
                "catch_chances",
                0,
            ) > 0
        ):

            strengths.append(
                "Reliable catching"
            )

        if (
            run_out_success >= 70
            and analysis.get(
                "run_out_attempts",
                0,
            ) > 0
        ):

            strengths.append(
                "Strong throwing accuracy"
            )

        if analysis.get(
            "direct_hits",
            0,
        ) >= 2:

            strengths.append(
                "Excellent direct-hit ability"
            )

        if analysis.get(
            "stops",
            0,
        ) >= 5:

            strengths.append(
                "Good ground fielding"
            )

        if errors == 0:

            strengths.append(
                "Clean fielding performance"
            )

        if (
            catch_success < 70
            and analysis.get(
                "catch_chances",
                0,
            ) > 0
        ):

            improvements.append(
                "Improve catching consistency"
            )

        if (
            run_out_success < 50
            and analysis.get(
                "run_out_attempts",
                0,
            ) > 0
        ):

            improvements.append(
                "Improve throwing accuracy"
            )

        if analysis.get(
            "misfields",
            0,
        ) >= 2:

            improvements.append(
                "Improve ground fielding technique"
            )

        if analysis.get(
            "dropped_catches",
            0,
        ) > 0:

            improvements.append(
                "Practice high-pressure catching drills"
            )

        return {
            "overall": overall,
            "strengths": strengths,
            "improvements": improvements,
        }

    # ============================================================
    # MULTIPLE MATCH ANALYSIS
    # ============================================================

    def analyze_performances(
        self,
        performances: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Analyze multiple fielding performances.
        """

        if not performances:
            return self._empty_analysis()

        total_catches = sum(
            int(
                item.get(
                    "catches",
                    0,
                )
            )
            for item in performances
        )

        total_chances = sum(
            int(
                item.get(
                    "catch_chances",
                    item.get(
                        "catches",
                        0,
                    ),
                )
            )
            for item in performances
        )

        total_drops = sum(
            int(
                item.get(
                    "dropped_catches",
                    0,
                )
            )
            for item in performances
        )

        total_run_outs = sum(
            int(
                item.get(
                    "run_outs",
                    0,
                )
            )
            for item in performances
        )

        total_run_out_attempts = sum(
            int(
                item.get(
                    "run_out_attempts",
                    item.get(
                        "run_outs",
                        0,
                    ),
                )
            )
            for item in performances
        )

        total_stumpings = sum(
            int(
                item.get(
                    "stumpings",
                    0,
                )
            )
            for item in performances
        )

        total_direct_hits = sum(
            int(
                item.get(
                    "direct_hits",
                    0,
                )
            )
            for item in performances
        )

        total_misfields = sum(
            int(
                item.get(
                    "misfields",
                    0,
                )
            )
            for item in performances
        )

        total_stops = sum(
            int(
                item.get(
                    "stops",
                    0,
                )
            )
            for item in performances
        )

        catch_success_rate = (
            self.calculate_catch_success_rate(
                total_catches,
                total_chances,
            )
        )

        run_out_success_rate = (
            self.calculate_run_out_success_rate(
                total_run_outs,
                total_run_out_attempts,
            )
        )

        individual_analyses = [
            self.analyze_match(
                item
            )
            for item in performances
        ]

        ratings = [
            item["rating"]
            for item
            in individual_analyses
        ]

        average_rating = (
            round(
                mean(ratings),
                2,
            )
            if ratings
            else 0.0
        )

        consistency = (
            self.calculate_consistency(
                ratings
            )
        )

        form = self.analyze_form(
            performances
        )

        return {
            "matches": len(
                performances
            ),
            "catches": total_catches,
            "catch_chances": total_chances,
            "dropped_catches": total_drops,
            "catch_success_rate": (
                catch_success_rate
            ),
            "run_outs": total_run_outs,
            "run_out_attempts": (
                total_run_out_attempts
            ),
            "run_out_success_rate": (
                run_out_success_rate
            ),
            "stumpings": total_stumpings,
            "direct_hits": total_direct_hits,
            "misfields": total_misfields,
            "stops": total_stops,
            "average_rating": average_rating,
            "consistency": consistency,
            "form": form,
            "strengths": (
                self.identify_strengths(
                    performances
                )
            ),
            "improvements": (
                self.identify_improvements(
                    performances
                )
            ),
        }

    # ============================================================
    # CONSISTENCY ANALYSIS
    # ============================================================

    def calculate_consistency(
        self,
        ratings: List[float],
    ) -> Dict[str, Any]:
        """
        Calculate fielding consistency.
        """

        if not ratings:

            return {
                "score": 0.0,
                "standard_deviation": 0.0,
                "classification": "no_data",
            }

        if len(ratings) == 1:

            return {
                "score": 50.0,
                "standard_deviation": 0.0,
                "classification": (
                    "insufficient_data"
                ),
            }

        average_rating = mean(
            ratings
        )

        deviation = pstdev(
            ratings
        )

        if average_rating <= 0:

            consistency_score = 0.0

        else:

            variation = (
                deviation
                / average_rating
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
            classification = "inconsistent"

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
        performances: List[
            Dict[str, Any]
        ],
        recent_matches: int = 5,
    ) -> Dict[str, Any]:
        """
        Analyze recent fielding form.
        """

        if not performances:

            return {
                "status": "no_data",
                "trend": "unknown",
            }

        analyses = [
            self.analyze_match(
                performance
            )
            for performance
            in performances
        ]

        ratings = [
            analysis["rating"]
            for analysis
            in analyses
        ]

        recent = ratings[
            -recent_matches:
        ]

        previous = ratings[
            :-recent_matches
        ]

        recent_average = round(
            mean(recent),
            2,
        )

        previous_average = (
            round(
                mean(previous),
                2,
            )
            if previous
            else recent_average
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

        if recent_average >= 80:
            status = "excellent"
        elif recent_average >= 65:
            status = "good"
        elif recent_average >= 45:
            status = "average"
        else:
            status = "needs_improvement"

        return {
            "status": status,
            "trend": trend,
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
        }

    # ============================================================
    # STRENGTH IDENTIFICATION
    # ============================================================

    def identify_strengths(
        self,
        performances: List[
            Dict[str, Any]
        ],
    ) -> List[str]:
        """
        Identify fielding strengths.
        """

        if not performances:
            return []

        analysis = (
            self._calculate_metrics(
                performances
            )
        )

        strengths = []

        if (
            analysis[
                "catch_success_rate"
            ] >= 85
        ):

            strengths.append(
                "Reliable catching ability"
            )

        if (
            analysis[
                "run_out_success_rate"
            ] >= 65
        ):

            strengths.append(
                "Accurate throwing"
            )

        if (
            analysis[
                "average_direct_hits"
            ] >= 1
        ):

            strengths.append(
                "Strong direct-hit capability"
            )

        if (
            analysis[
                "average_stops"
            ] >= 5
        ):

            strengths.append(
                "Good ground coverage"
            )

        if (
            analysis[
                "error_rate"
            ] <= 10
        ):

            strengths.append(
                "Low fielding error rate"
            )

        if not strengths:

            strengths.append(
                "Developing fielding skills"
            )

        return strengths

    # ============================================================
    # IMPROVEMENT IDENTIFICATION
    # ============================================================

    def identify_improvements(
        self,
        performances: List[
            Dict[str, Any]
        ],
    ) -> List[str]:
        """
        Identify fielding improvement areas.
        """

        if not performances:

            return [
                "More fielding data required"
            ]

        analysis = (
            self._calculate_metrics(
                performances
            )
        )

        improvements = []

        if (
            analysis[
                "catch_success_rate"
            ] < 70
        ):

            improvements.append(
                "Improve catching technique and consistency"
            )

        if (
            analysis[
                "run_out_success_rate"
            ] < 50
        ):

            improvements.append(
                "Improve throwing accuracy"
            )

        if (
            analysis[
                "error_rate"
            ] > 20
        ):

            improvements.append(
                "Reduce misfields and handling errors"
            )

        if (
            analysis[
                "consistency_score"
            ] < 55
        ):

            improvements.append(
                "Improve consistency under match pressure"
            )

        if not improvements:

            improvements.append(
                "Maintain current fielding standards"
            )

        return improvements

    # ============================================================
    # INTERNAL METRICS
    # ============================================================

    def _calculate_metrics(
        self,
        performances: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Calculate internal performance metrics.
        """

        if not performances:

            return {
                "catch_success_rate": 0.0,
                "run_out_success_rate": 0.0,
                "error_rate": 0.0,
                "average_direct_hits": 0.0,
                "average_stops": 0.0,
                "consistency_score": 0.0,
            }

        analyses = [
            self.analyze_match(
                performance
            )
            for performance
            in performances
        ]

        total_catches = sum(
            item["catches"]
            for item in analyses
        )

        total_chances = sum(
            item["catch_chances"]
            for item in analyses
        )

        total_run_outs = sum(
            item["run_outs"]
            for item in analyses
        )

        total_attempts = sum(
            item["run_out_attempts"]
            for item in analyses
        )

        total_errors = sum(
            item["dropped_catches"]
            + item["misfields"]
            for item in analyses
        )

        total_opportunities = sum(
            item["catch_chances"]
            + item["run_out_attempts"]
            + item["stops"]
            for item in analyses
        )

        ratings = [
            item["rating"]
            for item in analyses
        ]

        consistency = (
            self.calculate_consistency(
                ratings
            )
        )

        return {
            "catch_success_rate": (
                self.calculate_catch_success_rate(
                    total_catches,
                    total_chances,
                )
            ),
            "run_out_success_rate": (
                self.calculate_run_out_success_rate(
                    total_run_outs,
                    total_attempts,
                )
            ),
            "error_rate": (
                self.calculate_error_rate(
                    total_errors,
                    total_opportunities,
                )
            ),
            "average_direct_hits": round(
                mean(
                    item["direct_hits"]
                    for item in analyses
                ),
                2,
            ),
            "average_stops": round(
                mean(
                    item["stops"]
                    for item in analyses
                ),
                2,
            ),
            "consistency_score": (
                consistency["score"]
            ),
        }

    # ============================================================
    # COMPLETE REPORT
    # ============================================================

    def generate_report(
        self,
        performances: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Generate complete fielding report.
        """

        analysis = (
            self.analyze_performances(
                performances
            )
        )

        data_quality = (
            "sufficient_data"
            if len(performances)
            >= self.minimum_matches
            else "limited_data"
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
        Generate fielding training recommendations.
        """

        recommendations = []

        if (
            analysis.get(
                "catch_success_rate",
                0,
            )
            < 75
        ):

            recommendations.append(
                "Practice high catches, low catches, and reaction catching drills."
            )

        if (
            analysis.get(
                "run_out_success_rate",
                0,
            )
            < 55
        ):

            recommendations.append(
                "Practice quick pickup and direct-hit throwing drills."
            )

        if (
            analysis.get(
                "misfields",
                0,
            )
            > 2
        ):

            recommendations.append(
                "Work on body position and keeping the head over the ball."
            )

        consistency = analysis.get(
            "consistency",
            {},
        )

        if (
            consistency.get(
                "score",
                0,
            )
            < 60
        ):

            recommendations.append(
                "Practice fielding under fatigue and match-pressure conditions."
            )

        if not recommendations:

            recommendations.append(
                "Maintain current fielding training and continue improving reaction speed."
            )

        return recommendations

    # ============================================================
    # EMPTY ANALYSIS
    # ============================================================

    @staticmethod
    def _empty_analysis() -> Dict[str, Any]:
        """
        Return default empty analysis.
        """

        return {
            "matches": 0,
            "catches": 0,
            "catch_chances": 0,
            "dropped_catches": 0,
            "catch_success_rate": 0.0,
            "run_outs": 0,
            "run_out_attempts": 0,
            "run_out_success_rate": 0.0,
            "stumpings": 0,
            "direct_hits": 0,
            "misfields": 0,
            "stops": 0,
            "average_rating": 0.0,
            "consistency": {
                "score": 0.0,
                "classification": "no_data",
            },
            "form": {
                "status": "no_data",
                "trend": "unknown",
            },
            "strengths": [],
            "improvements": [
                "More fielding data required"
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

    analyzer = FieldingAnalysis()

    performances = [
        {
            "catches": 2,
            "catch_chances": 2,
            "dropped_catches": 0,
            "run_outs": 1,
            "run_out_attempts": 2,
            "direct_hits": 1,
            "misfields": 0,
            "stops": 6,
        },
        {
            "catches": 1,
            "catch_chances": 2,
            "dropped_catches": 1,
            "run_outs": 0,
            "run_out_attempts": 1,
            "direct_hits": 0,
            "misfields": 1,
            "stops": 5,
        },
        {
            "catches": 2,
            "catch_chances": 2,
            "dropped_catches": 0,
            "run_outs": 1,
            "run_out_attempts": 1,
            "direct_hits": 2,
            "misfields": 0,
            "stops": 8,
        },
    ]

    report = analyzer.generate_report(
        performances
    )

    print("\nRENIX FIELDING REPORT\n")

    for key, value in report.items():
        print(
            f"{key}: {value}"
        )


