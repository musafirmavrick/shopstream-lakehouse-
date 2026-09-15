"""
Spark Declarative Pipeline: bronze + silver for the ShopStream file-based
sources (product catalog + order reviews), ingested via Auto Loader from
the Unity Catalog landing volume that simulators/file_dropper.py
populates.

This file only WIRES things together -- Auto Loader reads, expectations,
and calls into the pure transform functions in
shopstream.transforms.catalog_cleaning. It should never contain cleaning
logic of its own.

Landing paths (populated by file_dropper.py):
    /Volumes/shopstream_dev/raw/landing/catalog/   <- products + category translation
    /Volumes/shopstream_dev/raw/landing/reviews/   <- order reviews

Auto Loader schema/checkpoint state lives under the same volume, since
Free Edition doesn't allow custom storage locations outside Unity
Catalog Volumes.
"""

import os
import sys

from pyspark import pipelines as dp
from pyspark.sql import functions as F

# The pipeline auto-adds only its OWN root folder to sys.path, not sibling
# folders. transforms/ lives one level up and over from this file
# (.../src/shopstream/pipelines/ -> .../src/shopstream/transforms/), so we
# compute that path relative to this file's own location and add it
# explicitly. This works in dev and prod alike since it never hardcodes a
# username or repo name.
_shopstream_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_shopstream_root, "transforms"))

from catalog_cleaning import clean_products, clean_reviews

CATALOG = "shopstream_dev"
LANDING_ROOT = f"/Volumes/{CATALOG}/raw/landing"

# ---------------------------------------------------------------------------
# Bronze: raw ingestion via Auto Loader, no cleaning.
#
# cloudFiles.inferColumnTypes=true is what makes the rescued-data column
# meaningful here: without it, Auto Loader treats every CSV column as a
# plain string and nothing ever looks "wrong" at the type level. With it
# on, Auto Loader infers product_weight_g etc. as numeric from the first
# clean batches -- so when batch 3's drifted file puts the literal string
# "N/A" in that column, it no longer matches the inferred type and gets
# captured in _rescued_data instead of silently becoming a bad numeric
# value.
#
# schemaEvolutionMode="addNewColumns" handles the OTHER half of drift: a
# genuinely new column (batch 3's product_condition) isn't a type
# mismatch, it's a column bronze has never seen. addNewColumns stops the
# stream once, updates the schema location, and the pipeline's automatic
# restart picks the new column up going forward.
# ---------------------------------------------------------------------------


@dp.table(
    name=f"{CATALOG}.bronze.products",
    comment="Raw product catalog, ingested via Auto Loader from the landing volume.",
)
def bronze_products():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("cloudFiles.schemaLocation", f"{LANDING_ROOT}/_schemas/products")
        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
        .option("rescuedDataColumn", "_rescued_data")
        .load(f"{LANDING_ROOT}/catalog/")
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_file", F.col("_metadata.file_path"))
    )


@dp.table(
    name=f"{CATALOG}.bronze.reviews",
    comment="Raw order reviews, ingested via Auto Loader from the landing volume.",
)
def bronze_reviews():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("cloudFiles.schemaLocation", f"{LANDING_ROOT}/_schemas/reviews")
        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
        .option("rescuedDataColumn", "_rescued_data")
        .load(f"{LANDING_ROOT}/reviews/")
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_file", F.col("_metadata.file_path"))
    )


# ---------------------------------------------------------------------------
# Silver: cleaned + quality-checked.
#
# Three expectations, three different violation actions -- exactly the
# spread the phase brief asks for:
#   @dp.expect            "warn"  -> row is kept, violation only logged
#   @dp.expect_or_drop     "drop"  -> row silently removed before writing
#   @dp.expect_or_fail     "fail"  -> the entire pipeline update stops
#
# Note on non_negative_weight: none of file_dropper's current malformed
# rows actually trip this one (a rescued "N/A" becomes null, which the
# rule allows). That's intentional -- fail is reserved for a genuine
# business-rule violation, not a benign missing value. To actually SEE
# the fail path in action, extend file_dropper's batch 3 to inject one
# row with a negative weight -- a good "Break it" exercise in its own
# right.
# ---------------------------------------------------------------------------


@dp.table(
    name=f"{CATALOG}.silver.dim_product",
    comment="Cleaned product catalog with data-quality expectations applied.",
)
@dp.expect(
    "plausible_condition",
    "product_condition IS NULL OR product_condition IN ('new', 'refurbished', 'open_box')",
)
@dp.expect_or_drop("valid_product_id", "product_id IS NOT NULL AND product_id != ''")
@dp.expect_or_fail("non_negative_weight", "product_weight_g IS NULL OR product_weight_g >= 0")
def silver_dim_product():
    bronze_df = spark.readStream.table(f"{CATALOG}.bronze.products")
    return clean_products(bronze_df)


@dp.table(
    name=f"{CATALOG}.silver.reviews",
    comment="Cleaned order reviews.",
)
@dp.expect_or_drop("valid_review_id", "review_id IS NOT NULL")
def silver_reviews():
    bronze_df = spark.readStream.table(f"{CATALOG}.bronze.reviews")
    return clean_reviews(bronze_df)
