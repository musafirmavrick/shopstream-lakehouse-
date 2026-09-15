"""
Pure transformation functions for cleaning ShopStream catalog and review data.

These functions take a DataFrame and return a DataFrame -- nothing else.
No Spark session creation, no reads, no writes, no Databricks-specific
APIs. That's deliberate: every function here can be unit tested with
local PySpark in a plain pytest file (Phase 8), with no dependency on a
Databricks pipeline or workspace. The pipeline module
(bronze_silver_catalog.py) imports these functions and wires them into
the actual Auto Loader flow -- it should contain almost no cleaning
logic of its own. If you find yourself writing a .withColumn(...) in the
pipeline file, it probably belongs here instead.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def clean_products(df: DataFrame) -> DataFrame:
    """Cleans the raw products catalog.

    - Trims whitespace from string columns (a common CSV export artifact).
    - Casts numeric-looking columns to proper numeric types using
      try_cast, which returns null on unparseable values (like the "N/A"
      file_dropper's batch 3 injects) instead of raising and killing the
      whole micro-batch.
    - Lowercases product_condition so "New" and "new" don't count as two
      different values downstream.
    - Leaves _rescued_data untouched -- if Auto Loader routed anything
      there, that's a signal worth keeping, not cleaning away.
    """
    string_cols = ["product_id", "product_category_name", "product_condition"]
    numeric_casts = {
        "product_weight_g": "double",
        "product_length_cm": "double",
        "product_height_cm": "double",
        "product_width_cm": "double",
    }

    for col_name in string_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.trim(F.col(col_name)))

    for col_name, target_type in numeric_casts.items():
        if col_name in df.columns:
            df = df.withColumn(
                col_name, F.expr(f"try_cast({col_name} as {target_type})")
            )

    if "product_condition" in df.columns:
        df = df.withColumn(
            "product_condition",
            F.when(
                F.col("product_condition").isNotNull(),
                F.lower(F.col("product_condition")),
            ),
        )

    return df


def clean_reviews(df: DataFrame) -> DataFrame:
    """Cleans the raw order reviews file.

    - Trims free-text review fields (Olist's export carries stray
      whitespace in the title/comment columns).
    - Casts review_score to an integer via try_cast, coercing anything
      unparseable to null rather than failing the batch.
    - Parses the two timestamp columns Olist ships as plain strings.
    """
    if "review_comment_title" in df.columns:
        df = df.withColumn(
            "review_comment_title", F.trim(F.col("review_comment_title"))
        )
    if "review_comment_message" in df.columns:
        df = df.withColumn(
            "review_comment_message", F.trim(F.col("review_comment_message"))
        )

    if "review_score" in df.columns:
        df = df.withColumn("review_score", F.expr("try_cast(review_score as int)"))

    for ts_col in ("review_creation_date", "review_answer_timestamp"):
        if ts_col in df.columns:
            df = df.withColumn(ts_col, F.expr(f"try_cast({ts_col} as timestamp)"))

    return df
