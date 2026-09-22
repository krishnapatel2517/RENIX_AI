"""
RENIX Analytics Package
=======================

Provides data loading, analysis, statistics, visualization,
table generation, and report generation capabilities.
"""

from .data_engine import DataEngine
from .csv_reader import CSVReader
from .excel_reader import ExcelReader
from .json_reader import JSONReader
from .statistics import StatisticsEngine
from .charts import ChartEngine
from .tables import TableEngine
from .reports import ReportEngine

__all__ = [
    "DataEngine",
    "CSVReader",
    "ExcelReader",
    "JSONReader",
    "StatisticsEngine",
    "ChartEngine",
    "TableEngine",
    "ReportEngine",
]

__version__ = "1.0.0"


