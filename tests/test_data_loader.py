# Copyright (c) 2026 Seizh7
# Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0)


from capitolwatch.analysis.data_loader import (
    parse_value_range,
    load_politicians,
    load_assets_with_products,
    get_dataset_summary
)


class TestParseValueRange:
    """Tests for parse_value_range function"""

    def test_parse_value_range_standard(self):
        """Test parsing a standard range"""
        result = parse_value_range("$1,001 - $15,000")
        expected = (1001 + 15000) / 2
        assert result == expected, f"Expected {expected}, got {result}"

    def test_parse_value_range_large(self):
        """Test parsing a large range"""
        result = parse_value_range("$1,000,001 - $5,000,000")
        expected = (1000001 + 5000000) / 2
        assert result == expected

    def test_parse_value_range_plus_sign(self):
        """Test parsing format '$50,000,001+'"""
        result = parse_value_range("$50,000,001+")
        expected = 50000001.0
        assert result == expected

    def test_parse_value_range_none(self):
        """Test parsing None value"""
        result = parse_value_range(None)
        assert result == 0.0

    def test_parse_value_range_empty(self):
        """Test parsing empty string"""
        result = parse_value_range("")
        assert result == 0.0

    def test_parse_value_range_none_text(self):
        """Test parsing 'None (or less than $201)'"""
        result = parse_value_range("None (or less than $201)")
        assert result == 201.0

    def test_parse_value_range_less_than_with_comma(self):
        """Test parsing 'None (or less than $1,000)'"""
        result = parse_value_range("None (or less than $1,000)")
        assert result == 1000.0

    def test_parse_value_range_just_none(self):
        """Test parsing just 'None' without threshold"""
        result = parse_value_range("None")
        assert result == 0.0

    def test_parse_value_range_without_commas(self):
        """Test parsing without commas"""
        result = parse_value_range("$1001 - $15000")
        expected = (1001 + 15000) / 2
        assert result == expected


class TestLoadPoliticians:
    """Tests for load_politicians function"""

    def test_load_politicians_not_empty(self):
        """Test that politicians are loaded (not empty)"""
        df = load_politicians()
        assert len(df) > 0, "No politicians loaded"

    def test_load_politicians_columns(self):
        """Test that expected columns are present"""
        df = load_politicians()
        expected_columns = {'id', 'first_name', 'last_name', 'party'}
        assert expected_columns.issubset(set(df.columns)), \
            f"Missing columns: {expected_columns - set(df.columns)}"

    def test_load_politicians_types(self):
        """Test that column types are correct"""
        df = load_politicians()
        assert df['id'].dtype == 'object', "ID should be string (VARCHAR)"
        assert df['first_name'].dtype == 'object', \
            "first_name should be string"
        assert df['party'].dtype == 'object', "party should be string"

    def test_load_politicians_no_nulls(self):
        """Test that critical columns have no NULL values"""
        df = load_politicians()
        assert df['id'].notnull().all(), "Some IDs are NULL"
        assert df['party'].notnull().all(), "Some parties are NULL"

    def test_load_politicians_valid_parties(self):
        """Test that parties are valid (Republican, Democratic, \
Independent)"""
        df = load_politicians()
        valid_parties = {
            'Republican', 'Democratic', 'Democrat', 'Independent'
        }
        invalid = df[~df['party'].isin(valid_parties)]
        assert len(invalid) == 0, \
            f"Invalid parties found: {invalid['party'].unique()}"


class TestLoadAssetsWithProducts:
    """Tests for load_assets_with_products function"""

    def test_load_assets_not_empty(self):
        """Test that assets are loaded (not empty)"""
        df = load_assets_with_products()
        assert len(df) > 0, "No assets loaded"

    def test_load_assets_columns(self):
        """Test that expected columns are present"""
        df = load_assets_with_products()
        expected_columns = {
            'asset_id', 'politician_id', 'product_id', 'value',
            'value_numeric', 'subtype', 'product_name'
        }
        assert expected_columns.issubset(set(df.columns)), \
            f"Missing columns: {expected_columns - set(df.columns)}"

    def test_load_assets_value_numeric_type(self):
        """Test that value_numeric is float type"""
        df = load_assets_with_products()
        assert df['value_numeric'].dtype == 'float64', \
            "value_numeric should be float64"

    def test_load_assets_value_numeric_non_negative(self):
        """Test that all numeric values are >= 0"""
        df = load_assets_with_products()
        assert (df['value_numeric'] >= 0).all(), \
            "Negative values found"

    def test_load_assets_subtype_exists(self):
        """Test that subtype column exists and has no empty values"""
        df = load_assets_with_products()
        # All assets should have a subtype (empty ones normalized to
        # 'Uncategorized')
        assert df['subtype'].notnull().all(), "Some subtypes are NULL"
        # Check that no empty strings exist
        assert (df['subtype'] != '').all(), \
            "Some subtypes are empty strings"
        # Check that 'Uncategorized' category exists
        # (for previously empty subtypes)
        assert 'Uncategorized' in df['subtype'].values, \
            "No 'Uncategorized' subtype found (empty subtypes \
should be normalized)"

    def test_load_assets_politician_ids_valid(self):
        """Test that all politician_ids correspond to active politicians"""
        politicians = load_politicians()
        assets = load_assets_with_products()

        valid_ids = set(politicians['id'])
        asset_politician_ids = set(assets['politician_id'].unique())

        invalid = asset_politician_ids - valid_ids
        assert len(invalid) == 0, \
            f"Assets with invalid politician_id: {invalid}"


class TestGetDatasetSummary:
    """Tests for get_dataset_summary function"""

    def test_summary_structure(self):
        """Test that summary contains all expected keys"""
        summary = get_dataset_summary()
        expected_keys = {
            'n_politicians', 'n_assets', 'n_unique_subtypes',
            'avg_assets_per_politician', 'party_distribution',
            'top_subtypes', 'total_value', 'mean_value', 'median_value'
        }
        assert expected_keys.issubset(set(summary.keys())), \
            f"Missing keys: {expected_keys - set(summary.keys())}"

    def test_summary_values(self):
        """Test that summary values are consistent"""
        summary = get_dataset_summary()

        assert summary['n_politicians'] > 0, "No politicians in summary"
        assert summary['n_assets'] > 0, "No assets in summary"
        assert summary['avg_assets_per_politician'] > 0, \
            "Average should be > 0"
        assert summary['n_unique_subtypes'] > 0, \
            "Should have at least some subtypes"

    def test_summary_party_distribution(self):
        """Test that party distribution is consistent"""
        summary = get_dataset_summary()
        party_dist = summary['party_distribution']

        # Total should match number of politicians
        total = sum(party_dist.values())
        assert total == summary['n_politicians'], \
            f"Party total = {total}, expected {summary['n_politicians']}"

        # Should contain at least one major party
        has_party = any(
            party in party_dist
            for party in ['Republican', 'Democratic', 'Democrat']
        )
        assert has_party, "No major parties found in distribution"


class TestIntegration:
    """Integration tests to verify overall consistency"""

    def test_all_politicians_have_assets(self):
        """Test that all loaded politicians have at least 1 asset"""
        politicians = load_politicians()
        assets = load_assets_with_products()

        # Count assets per politician
        assets_per_pol = assets.groupby('politician_id').size()

        for pol_id in politicians['id']:
            assert pol_id in assets_per_pol.index, \
                f"Politician {pol_id} has no assets"
            assert assets_per_pol[pol_id] > 0, \
                f"Politician {pol_id} has 0 assets"

    def test_value_parsing_applied(self):
        """Test that value parsing has been applied"""
        assets = load_assets_with_products()

        # Verify that value_numeric has some non-zero values
        non_zero = (assets['value_numeric'] > 0).sum()
        total = len(assets)

        # At least 10% of assets should have parsed values > 0
        assert non_zero > total * 0.1, \
            f"Only {non_zero}/{total} non-zero values, possible parsing error"

    def test_data_quality_metrics(self):
        """Test data quality metrics"""
        summary = get_dataset_summary()

        # Verify metrics are reasonable
        assert summary['mean_value'] > 0, "Mean value should be > 0"
        assert summary['median_value'] >= 0, "Median value should be >= 0"
        assert summary['total_value'] >= 0, "Total value should be >= 0"


if __name__ == "__main__":
    try:
        print("\n1. Test parse_value_range...")
        assert parse_value_range("$1,001 - $15,000") == 8000.5
        assert parse_value_range("$50,000,001+") == 50000001.0
        assert parse_value_range(None) == 0.0
        print("   parse_value_range OK")

        print("\n2. Test load_politicians...")
        politicians = load_politicians()
        assert len(politicians) > 0, "No politicians loaded"
        print(f"   load_politicians OK ({len(politicians)} loaded)")

        print("\n3. Test load_assets_with_products...")
        assets = load_assets_with_products()
        assert len(assets) > 0, "No assets loaded"
        assert 'value_numeric' in assets.columns
        print(f"   load_assets_with_products OK ({len(assets)} loaded)")

        print("\n4. Test get_dataset_summary...")
        summary = get_dataset_summary()
        assert summary['n_politicians'] > 0
        assert summary['n_assets'] > 0
        n_pols = summary['n_politicians']
        n_assets = summary['n_assets']
        print(f"   get_dataset_summary OK ({n_pols} politicians, "
              f"{n_assets} assets)")

    except AssertionError as e:
        print(f"\nTest failed: {e}")
    except Exception as e:
        print(f"\nError: {e}")
