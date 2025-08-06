"""
Comprehensive test suite for Arkouda parquet file reading capabilities.

This module tests Arkouda's parquet reading functionality against the Apache
parquet-testing repository dataset. It categorizes files by expected behavior
and provides detailed reporting on support status.

IMPORTANT:
Set ARKOUDA_PARQUET_TEST_DATA_DIR environment variable to enable these tests.

Test data from: https://github.com/apache/parquet-testing/

Test Categories:
- Crash files: Known to crash the server (skipped entirely)
- Expected failures: Files that currently fail to read
- Pandas incompatible: Files pandas cannot read (Arkouda-only testing)
- Correctness failures: Files that read but produce incorrect data
- Passing files: Files that should read correctly

Environment Variables:
- ARKOUDA_PARQUET_TEST_DATA_DIR: Path to parquet test files directory (REQUIRED)
"""

import os
from pathlib import Path
from typing import List, Optional, Union
import logging

import pytest
import pandas as pd
import arkouda as ak

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration and constants
# Get test data directory from environment - no default, must be explicitly set
TEST_DATA_DIR = os.getenv("ARKOUDA_PARQUET_TEST_DATA_DIR")

# Skip entire module if environment not configured
if TEST_DATA_DIR is None:
    pytest.skip(
        "ARKOUDA_PARQUET_TEST_DATA_DIR environment variable must be set to run parquet tests",
        allow_module_level=True
    )

# Files that crash the Arkouda server - these are skipped entirely
CRASH_FILES = frozenset([
    'int32_decimal.parquet',
    'int64_decimal.parquet',
    'byte_array_decimal.parquet'
])

# Files that pandas cannot read correctly
PANDAS_INCOMPATIBLE_FILES = frozenset([
    'large_string_map.brotli.parquet',
    'nested_structs.rust.parquet',
    'fixed_length_byte_array.parquet',
    'incorrect_map_schema.parquet',
])

# Files that currently fail to read with Arkouda
EXPECTED_READ_FAILURES = frozenset([
    'nulls.snappy.parquet',
    'byte_stream_split_extended.gzip.parquet',
    'nested_maps.snappy.parquet',
    'nullable.impala.parquet',
    'float16_nonzeros_and_nans.parquet',
    'alltypes_plain.snappy.parquet',
    'old_list_structure.parquet',
    'delta_encoding_optional_column.parquet',
    'int32_with_null_pages.parquet',
    'large_string_map.brotli.parquet',
    'map_no_value.parquet',
    'nested_structs.rust.parquet',
    'rle_boolean_encoding.parquet',
    'alltypes_dictionary.parquet',
    'repeated_no_annotation.parquet',
    'nonnullable.impala.parquet',
    'repeated_primitive_no_list.parquet',
    'datapage_v2.snappy.parquet',
    'page_v2_empty_compressed.parquet',
    'float16_zeros_and_nans.parquet',
    'list_columns.parquet',
    'fixed_length_byte_array.parquet',
    'null_list.parquet',
    'alltypes_tiny_pages.parquet',
    'alltypes_tiny_pages_plain.parquet',
    'incorrect_map_schema.parquet',
])

# Files that read successfully but produce incorrect data
EXPECTED_CORRECTNESS_FAILURES = frozenset([
    'binary.parquet',
    'unknown-logical-type.parquet',
    'non_hadoop_lz4_compressed.parquet',
    'fixed_length_decimal_legacy.parquet',
    'fixed_length_decimal.parquet',
    'nation.dict-malformed.parquet',
    'nested_lists.snappy.parquet',
    'int96_from_spark.parquet',
    'delta_byte_array.parquet',
    'sort_columns.parquet',
    'alltypes_plain.parquet',
    'rle-dict-snappy-checksum.parquet',
    'hadoop_lz4_compressed_larger.parquet',
    'hadoop_lz4_compressed.parquet',
    'lz4_raw_compressed_larger.parquet',
    'lz4_raw_compressed.parquet',
    'delta_encoding_required_column.parquet',
    'plain-dict-uncompressed-checksum.parquet',
    'rle-dict-uncompressed-corrupt-checksum.parquet',
])

# Known Arkouda string conversion issues
ARKOUDA_STRING_ERRORS = (
    "Bad index type or format",
    "data type '<U-1' not understood"
)


def get_test_parquet_files() -> List[Path]:
    """
    Discover parquet test files, excluding those known to crash the server.

    Returns:
        List of Path objects for parquet files to test (empty list if env var not set)

    Raises:
        FileNotFoundError: If test data directory doesn't exist
        ValueError: If no parquet files found in valid directory
    """
    if TEST_DATA_DIR is None:
        # Return empty list for parametrize at import time - tests will be skipped
        return []

    data_dir = Path(TEST_DATA_DIR)

    if not data_dir.exists():
        # Return empty list to avoid import-time errors - tests will be skipped at runtime
        return []

    parquet_files = [
        f for f in data_dir.glob("*.parquet")
        if f.name not in CRASH_FILES
    ]

    if not parquet_files:
        # Return empty list to avoid import-time errors - tests will be skipped at runtime
        return []

    logger.info(f"Found {len(parquet_files)} parquet test files in {data_dir}")
    return parquet_files


class ParquetTestResult:
    """Data class to hold test results for a single file."""

    def __init__(
        self,
        filename: str,
        read_success: bool,
        correctness_match: Optional[bool] = None,
        error_message: Optional[str] = None,
        pandas_skipped: bool = False
    ):
        self.filename = filename
        self.read_success = read_success
        self.correctness_match = correctness_match
        self.error_message = error_message
        self.pandas_skipped = pandas_skipped


def read_parquet_with_arkouda(
    filepath: Union[str, Path],
    skip_pandas_check: bool = False
) -> ParquetTestResult:
    """
    Read a parquet file with Arkouda and optionally compare with pandas.

    Returns:
        ParquetTestResult object with test outcomes
    """
    filename = Path(filepath).name

    # Step 1: Try to read with Arkouda
    try:
        ak_data = ak.read_parquet(str(filepath))
        logger.debug(f"✓ Arkouda successfully read {filename}")
    except Exception as e:
        error_msg = f"Arkouda failed to read: {str(e)}"
        logger.debug(f"✗ {error_msg}")
        return ParquetTestResult(filename, False, error_message=error_msg)

    # Step 2: Skip pandas comparison if requested
    if skip_pandas_check:
        return ParquetTestResult(filename, True, pandas_skipped=True)

    # Step 3: Try to read with pandas for comparison
    try:
        df_pandas = pd.read_parquet(str(filepath))
        logger.debug(f"✓ Pandas successfully read {filename}")
    except Exception as e:
        error_msg = f"Pandas failed to read: {str(e)}"
        logger.debug(f"✗ {error_msg}")
        return ParquetTestResult(filename, True, error_message=error_msg)

    # Step 4: Convert Arkouda data to DataFrame and then to pandas
    try:
        df_arkouda = ak.DataFrame(ak_data)
        df_ak_as_pandas = df_arkouda.to_pandas()
        logger.debug(f"✓ Arkouda DataFrame conversion successful for {filename}")
    except Exception as e:
        error_msg = f"Arkouda DataFrame conversion failed: {str(e)}"

        # Check for known string conversion issues
        if any(err_pattern in str(e) for err_pattern in ARKOUDA_STRING_ERRORS):
            error_msg = f"Known Arkouda string conversion issue: {str(e)}"

        logger.debug(f"✗ {error_msg}")
        return ParquetTestResult(filename, True, False, error_msg)

    # Step 5: Compare DataFrames
    try:
        pd.testing.assert_frame_equal(
            df_ak_as_pandas,
            df_pandas,
            check_dtype=False,
            check_index_type=False
        )
        correctness_match = True
        logger.debug(f"✓ Data correctness check passed for {filename}")
    except AssertionError:
        correctness_match = False
        logger.debug(f"✗ Data correctness check failed for {filename}")

    return ParquetTestResult(filename, True, correctness_match)


class TestParquetReading:
    """Test reading edge case parquet files. Requires explicit opt-in via environment variable."""

    @classmethod
    def setup_class(cls) -> None:
        # Environment variable check now handled at module level
        
        # Validate test data directory exists
        data_dir = Path(TEST_DATA_DIR)
        if not data_dir.exists():
            pytest.skip(
                f"Test data directory not found: {data_dir}. "
                f"Ensure ARKOUDA_PARQUET_TEST_DATA_DIR points to a valid directory"
            )

        # Check for parquet files
        parquet_files = [
            f for f in data_dir.glob("*.parquet")
            if f.name not in CRASH_FILES
        ]

        if not parquet_files:
            pytest.skip(f"No parquet files found in {data_dir}")

        cls.test_files = parquet_files
        logger.info(f"Test setup complete: {len(cls.test_files)} files to test")

    @pytest.mark.parametrize("parquet_file", get_test_parquet_files(), ids=lambda p: p.name)
    def test_parquet_file_reading(self, parquet_file: Path) -> None:
        """
        Test reading individual parquet files with appropriate expectations.

        Args:
            parquet_file: Path to parquet file to test
        """
        filename = parquet_file.name

        # Determine test expectations
        is_expected_failure = filename in EXPECTED_READ_FAILURES
        is_pandas_incompatible = filename in PANDAS_INCOMPATIBLE_FILES
        is_expected_correctness_failure = filename in EXPECTED_CORRECTNESS_FAILURES

        # Apply xfail marker for expected failures
        if is_expected_failure:
            pytest.xfail(f"File {filename} is expected to fail reading with Arkouda")

        # Run the test
        result = read_parquet_with_arkouda(parquet_file, skip_pandas_check=is_pandas_incompatible)

        # Assert read success
        assert result.read_success, f"Failed to read {filename}: {result.error_message}"

        # Handle correctness checking
        if not result.pandas_skipped and result.correctness_match is not None:
            if is_expected_correctness_failure:
                if result.correctness_match:
                    pytest.fail(
                        f"File {filename} unexpectedly has correct data! "
                        f"Correctness may have improved - remove from EXPECTED_CORRECTNESS_FAILURES"
                    )
                else:
                    logger.info(f"File {filename} has expected correctness issues")
            else:
                assert result.correctness_match, (
                    f"Data correctness check failed for {filename}. "
                    f"Add to EXPECTED_CORRECTNESS_FAILURES if this is expected."
                )
        elif result.error_message and "Pandas failed to read" in result.error_message:
            # Only fail if we expected pandas to work but it didn't
            if not is_pandas_incompatible:
                pytest.fail(
                    f"Pandas failed to read {filename} but this wasn't expected. "
                    f"Add to PANDAS_INCOMPATIBLE_FILES if needed: {result.error_message}"
                )

    def test_crash_files_existence(self) -> None:
        """Verify that crash files exist but are properly skipped."""
        data_dir = Path(TEST_DATA_DIR)
        existing_crash_files = [
            crash_file for crash_file in CRASH_FILES
            if (data_dir / crash_file).exists()
        ]

        if existing_crash_files:
            logger.info(f"Crash files found (properly skipped): {existing_crash_files}")
        else:
            logger.warning("No crash files found - they may have been fixed or removed")

    def test_configuration_consistency(self) -> None:
        """Verify that file categorization lists don't overlap inappropriately."""
        # Check for overlaps between categories
        read_failures_and_correctness = EXPECTED_READ_FAILURES & EXPECTED_CORRECTNESS_FAILURES
        if read_failures_and_correctness:
            pytest.fail(
                f"Files cannot be both read failures and correctness failures: "
                f"{read_failures_and_correctness}"
            )

        crash_and_others = CRASH_FILES & (EXPECTED_READ_FAILURES | EXPECTED_CORRECTNESS_FAILURES)
        if crash_and_others:
            pytest.fail(
                f"Crash files should not be in other categories: {crash_and_others}"
            )

    def test_summary_statistics(self) -> None:
        """Generate comprehensive summary statistics of test results. Always passes."""
        stats = {
            'total_files': 0,
            'successful_reads': 0,
            'failed_reads': 0,
            'correctness_passed': 0,
            'correctness_failed_unexpected': 0,
            'correctness_failed_expected': 0,
            'pandas_skipped': 0,
            'conversion_errors': 0
        }

        results = []
        for parquet_file in self.test_files:
            filename = parquet_file.name
            stats['total_files'] += 1

            is_pandas_incompatible = filename in PANDAS_INCOMPATIBLE_FILES
            is_expected_correctness_failure = filename in EXPECTED_CORRECTNESS_FAILURES

            try:
                result = read_parquet_with_arkouda(parquet_file, skip_pandas_check=is_pandas_incompatible)
                results.append(result)

                if result.read_success:
                    stats['successful_reads'] += 1

                    if result.pandas_skipped:
                        stats['pandas_skipped'] += 1
                    elif result.correctness_match is True:
                        stats['correctness_passed'] += 1
                    elif result.correctness_match is False:
                        if is_expected_correctness_failure:
                            stats['correctness_failed_expected'] += 1
                        else:
                            stats['correctness_failed_unexpected'] += 1

                    if result.error_message and any(err in result.error_message for err in ARKOUDA_STRING_ERRORS):
                        stats['conversion_errors'] += 1
                else:
                    stats['failed_reads'] += 1

            except Exception as e:
                stats['failed_reads'] += 1
                logger.error(f"Unexpected error testing {filename}: {e}")

        # Print comprehensive summary
        print(f"\n{'='*50}")
        print("ARKOUDA PARQUET READING SUMMARY")
        print(f"{'='*50}")
        print(f"Total files tested: {stats['total_files']}")
        print(f"Successful reads: {stats['successful_reads']}")
        print(f"Failed reads: {stats['failed_reads']}")
        print(f"Success rate: {stats['successful_reads']/stats['total_files']*100:.1f}%")
        print()
        print("CORRECTNESS ANALYSIS:")
        print(f"  Correctness passed: {stats['correctness_passed']}")
        print(f"  Correctness failed (unexpected): {stats['correctness_failed_unexpected']}")
        print(f"  Correctness failed (expected): {stats['correctness_failed_expected']}")
        print(f"  Pandas incompatible (skipped): {stats['pandas_skipped']}")
        print(f"  String conversion errors: {stats['conversion_errors']}")

        total_correctness_checks = (
            stats['correctness_passed'] +
            stats['correctness_failed_unexpected'] +
            stats['correctness_failed_expected']
        )

        if total_correctness_checks > 0:
            correctness_rate = stats['correctness_passed'] / total_correctness_checks * 100
            print(f"Overall correctness rate: {correctness_rate:.1f}%")

            unexpected_checks = stats['correctness_passed'] + stats['correctness_failed_unexpected']
            if unexpected_checks > 0:
                unexpected_rate = stats['correctness_passed'] / unexpected_checks * 100
                print(f"Correctness rate (excluding expected failures): {unexpected_rate:.1f}%")

        print(f"{'='*50}")

        # Always pass - this is just for reporting
        assert True
