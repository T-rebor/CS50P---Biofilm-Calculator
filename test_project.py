"""
Test Suite for Biofilm Calculator - Public Educational Version
CS50P Final Project Testing Module

Tests all main functions from project.py using pytest.
Based on published literature values only.
"""

import pytest
import json
import os
from dataclasses import asdict, fields

from project import (
    Formulation,
    CalculatedProperties,
    TargetProperties,
    calculate_properties,
    optimize_formulation,
    compare_materials,
    generate_report,
    VERSION,
)

from fiber_database import (
    FiberType,
    MatrixType,
    PlasticizerType,
    get_fiber,
    get_matrix,
    get_plasticizer,
    list_available_fibers,
    list_available_matrices,
    list_available_plasticizers,
    FIBER_DATABASE,
    MATRIX_DATABASE,
    PLASTICIZER_DATABASE,
    REFERENCE_MATERIALS,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def valid_formulation():
    """Standard valid formulation for testing."""
    return Formulation(
        name="test_formulation",
        fiber_type=FiberType.SISAL,
        matrix_type=MatrixType.STARCH_CORN,
        plasticizer_type=PlasticizerType.GLYCEROL,
        fiber_content=15.0,
        matrix_content=60.0,
        plasticizer_content=25.0
    )


@pytest.fixture
def high_fiber_formulation():
    """High fiber content formulation."""
    return Formulation(
        name="high_fiber",
        fiber_type=FiberType.JUTE,
        matrix_type=MatrixType.STARCH_POTATO,
        plasticizer_type=PlasticizerType.SORBITOL,
        fiber_content=35.0,
        matrix_content=45.0,
        plasticizer_content=20.0
    )


@pytest.fixture
def low_plasticizer_formulation():
    """Low plasticizer formulation for rigid films."""
    return Formulation(
        name="rigid_film",
        fiber_type=FiberType.FLAX,
        matrix_type=MatrixType.PLA,
        plasticizer_type=PlasticizerType.PEG_400,
        fiber_content=20.0,
        matrix_content=70.0,
        plasticizer_content=10.0
    )


@pytest.fixture
def calculated_properties(valid_formulation):
    """Pre-calculated properties for comparison tests."""
    return calculate_properties(valid_formulation)


# =============================================================================
# TEST DATABASE INTEGRITY
# =============================================================================

class TestDatabaseIntegrity:
    """Tests for fiber_database.py data integrity."""

    def test_all_fibers_have_required_attributes(self):
        """Verify all fibers have complete data."""
        for fiber_type in FiberType:
            fiber = get_fiber(fiber_type)
            assert fiber is not None, f"Fiber {fiber_type} not found"
            assert fiber.mechanical.tensile_strength > 0
            assert fiber.mechanical.youngs_modulus > 0
            assert fiber.density > 0

    def test_all_matrices_have_required_attributes(self):
        """Verify all matrices have complete data."""
        for matrix_type in MatrixType:
            matrix = get_matrix(matrix_type)
            assert matrix is not None, f"Matrix {matrix_type} not found"
            assert matrix.mechanical.tensile_strength > 0

    def test_all_plasticizers_have_flexibility_factor(self):
        """Verify all plasticizers have valid flexibility factor."""
        for plas_type in PlasticizerType:
            plas = get_plasticizer(plas_type)
            assert plas is not None, f"Plasticizer {plas_type} not found"
            assert 0 < plas.flexibility_factor <= 2.0

    def test_reference_materials_exist(self):
        """Verify reference materials are available."""
        assert len(REFERENCE_MATERIALS) >= 4, "Need at least 4 reference materials"

        for ref_id, ref in REFERENCE_MATERIALS.items():
            assert ref.tensile_strength > 0

    def test_fiber_source_citations(self):
        """Verify all fibers have source references."""
        for fiber_type in FiberType:
            fiber = get_fiber(fiber_type)
            assert fiber.source, f"Fiber {fiber_type} missing source"
            assert len(fiber.source) > 5


# =============================================================================
# TEST FORMULATION CLASS
# =============================================================================

class TestFormulation:
    """Tests for Formulation dataclass."""

    def test_valid_formulation_creation(self, valid_formulation):
        """Test creating a valid formulation."""
        assert valid_formulation.fiber_type == FiberType.SISAL
        assert valid_formulation.fiber_content == 15.0
        assert valid_formulation.plasticizer_content == 25.0

    def test_formulation_validation_passes(self, valid_formulation):
        """Test validation passes for valid formulation."""
        is_valid, msg = valid_formulation.validate()
        assert is_valid is True

    def test_fiber_content_validation(self):
        """Test fiber content bounds."""
        # Valid range should be checked by validate()
        formulation = Formulation(
            name="test",
            fiber_type=FiberType.SISAL,
            matrix_type=MatrixType.STARCH_CORN,
            plasticizer_type=PlasticizerType.GLYCEROL,
            fiber_content=15.0,
            matrix_content=60.0,
            plasticizer_content=25.0
        )
        is_valid, _ = formulation.validate()
        assert is_valid

    def test_formulation_serialization(self, valid_formulation):
        """Test formulation can be serialized to dict."""
        formulation_dict = asdict(valid_formulation)
        assert isinstance(formulation_dict, dict)
        assert formulation_dict["fiber_content"] == 15.0


# =============================================================================
# TEST CALCULATE_PROPERTIES FUNCTION
# =============================================================================

class TestCalculateProperties:
    """Tests for calculate_properties() main function."""

    def test_returns_calculated_properties(self, valid_formulation):
        """Test function returns CalculatedProperties object."""
        result = calculate_properties(valid_formulation)
        assert isinstance(result, CalculatedProperties)

    def test_tensile_strength_positive(self, valid_formulation):
        """Test tensile strength is positive."""
        result = calculate_properties(valid_formulation)
        assert result.tensile_strength > 0

    def test_tensile_strength_realistic_range(self, valid_formulation):
        """Test tensile strength is in realistic range (1-100 MPa)."""
        result = calculate_properties(valid_formulation)
        assert 1 <= result.tensile_strength <= 100

    def test_elongation_positive(self, valid_formulation):
        """Test elongation is positive."""
        result = calculate_properties(valid_formulation)
        assert result.elongation_at_break > 0

    def test_elongation_realistic_range(self, valid_formulation):
        """Test elongation is in realistic range (1-200%)."""
        result = calculate_properties(valid_formulation)
        assert 1 <= result.elongation_at_break <= 200

    def test_modulus_positive(self, valid_formulation):
        """Test Young's modulus is positive."""
        result = calculate_properties(valid_formulation)
        assert result.youngs_modulus > 0

    def test_wvp_positive(self, valid_formulation):
        """Test WVP is positive."""
        result = calculate_properties(valid_formulation)
        assert result.wvp > 0

    def test_biodegradation_days_realistic(self, valid_formulation):
        """Test biodegradation time is realistic (30-365 days)."""
        result = calculate_properties(valid_formulation)
        assert 30 <= result.biodegradation_days <= 365

    def test_bio_content_range(self, valid_formulation):
        """Test bio content is 0-100."""
        result = calculate_properties(valid_formulation)
        assert 0 <= result.bio_content <= 100

    def test_confidence_level_valid(self, valid_formulation):
        """Test confidence level is valid string."""
        result = calculate_properties(valid_formulation)
        assert result.confidence_level in ["High", "Medium", "Low"]

    def test_all_fiber_types_work(self):
        """Test calculation works for all fiber types."""
        for fiber_type in FiberType:
            formulation = Formulation(
                name="test",
                fiber_type=fiber_type,
                matrix_type=MatrixType.STARCH_CORN,
                plasticizer_type=PlasticizerType.GLYCEROL,
                fiber_content=15.0,
                matrix_content=60.0,
                plasticizer_content=25.0
            )
            result = calculate_properties(formulation)
            assert result.tensile_strength > 0

    def test_all_matrix_types_work(self):
        """Test calculation works for all matrix types."""
        for matrix_type in MatrixType:
            formulation = Formulation(
                name="test",
                fiber_type=FiberType.SISAL,
                matrix_type=matrix_type,
                plasticizer_type=PlasticizerType.GLYCEROL,
                fiber_content=15.0,
                matrix_content=60.0,
                plasticizer_content=25.0
            )
            result = calculate_properties(formulation)
            assert result.tensile_strength > 0

    def test_all_plasticizer_types_work(self):
        """Test calculation works for all plasticizer types."""
        for plas_type in PlasticizerType:
            formulation = Formulation(
                name="test",
                fiber_type=FiberType.SISAL,
                matrix_type=MatrixType.STARCH_CORN,
                plasticizer_type=plas_type,
                fiber_content=15.0,
                matrix_content=60.0,
                plasticizer_content=25.0
            )
            result = calculate_properties(formulation)
            assert result.tensile_strength > 0

    def test_higher_fiber_increases_strength(self):
        """Test that higher fiber content increases tensile strength."""
        low_fiber = Formulation(
            name="low",
            fiber_type=FiberType.SISAL,
            matrix_type=MatrixType.STARCH_CORN,
            plasticizer_type=PlasticizerType.GLYCEROL,
            fiber_content=5.0,
            matrix_content=70.0,
            plasticizer_content=25.0
        )
        high_fiber = Formulation(
            name="high",
            fiber_type=FiberType.SISAL,
            matrix_type=MatrixType.STARCH_CORN,
            plasticizer_type=PlasticizerType.GLYCEROL,
            fiber_content=30.0,
            matrix_content=45.0,
            plasticizer_content=25.0
        )

        low_result = calculate_properties(low_fiber)
        high_result = calculate_properties(high_fiber)

        assert high_result.tensile_strength > low_result.tensile_strength

    def test_plasticizer_content_affects_elongation(self):
        """Test that different plasticizer content affects elongation."""
        low_plas = Formulation(
            name="low_plas",
            fiber_type=FiberType.SISAL,
            matrix_type=MatrixType.STARCH_CORN,
            plasticizer_type=PlasticizerType.GLYCEROL,
            fiber_content=15.0,
            matrix_content=60.0,
            plasticizer_content=25.0
        )
        high_plas = Formulation(
            name="high_plas",
            fiber_type=FiberType.SISAL,
            matrix_type=MatrixType.STARCH_CORN,
            plasticizer_type=PlasticizerType.GLYCEROL,
            fiber_content=15.0,
            matrix_content=50.0,
            plasticizer_content=35.0
        )

        low_result = calculate_properties(low_plas)
        high_result = calculate_properties(high_plas)

        # Different plasticizer content should produce different elongation
        assert low_result.elongation_at_break != high_result.elongation_at_break


# =============================================================================
# TEST OPTIMIZE_FORMULATION FUNCTION
# =============================================================================

class TestOptimizeFormulation:
    """Tests for optimize_formulation() function."""

    def test_returns_list(self):
        """Test function returns a list."""
        target = TargetProperties(
            name="packaging",
            min_tensile_strength=10.0,
            max_tensile_strength=30.0,
            min_elongation=15.0,
            max_elongation=50.0,
            priority="balanced"
        )
        results = optimize_formulation(
            fiber_type=FiberType.SISAL,
            target=target
        )
        assert isinstance(results, list)

    def test_results_not_empty(self):
        """Test function returns results."""
        target = TargetProperties(
            name="flexible",
            min_tensile_strength=5.0,
            max_tensile_strength=20.0,
            min_elongation=20.0,
            max_elongation=100.0,
            priority="elongation"
        )
        results = optimize_formulation(
            fiber_type=FiberType.SISAL,
            target=target,
            n_results=5
        )
        assert len(results) > 0


# =============================================================================
# TEST COMPARE_MATERIALS FUNCTION
# =============================================================================

class TestCompareMaterials:
    """Tests for compare_materials() function."""

    def test_returns_dict(self, calculated_properties):
        """Test function returns a dictionary."""
        result = compare_materials(calculated_properties)
        assert isinstance(result, dict)

    def test_contains_reference_materials(self, calculated_properties):
        """Test comparison includes reference materials."""
        result = compare_materials(calculated_properties)
        assert len(result) > 0

    def test_comparison_has_percentage_values(self, calculated_properties):
        """Test each comparison has percentage values."""
        result = compare_materials(calculated_properties)

        for ref_name, comparison in result.items():
            assert "tensile_ratio" in comparison or "tensile_strength" in str(comparison)


# =============================================================================
# TEST GENERATE_REPORT FUNCTION
# =============================================================================

class TestGenerateReport:
    """Tests for generate_report() function."""

    def test_returns_string(self, valid_formulation, calculated_properties):
        """Test function returns a string."""
        result = generate_report(valid_formulation, calculated_properties)
        assert isinstance(result, str)

    def test_report_not_empty(self, valid_formulation, calculated_properties):
        """Test report is not empty."""
        result = generate_report(valid_formulation, calculated_properties)
        assert len(result) > 100

    def test_report_contains_properties(self, valid_formulation, calculated_properties):
        """Test report contains calculated properties."""
        result = generate_report(valid_formulation, calculated_properties)

        assert "tensile" in result.lower() or "strength" in result.lower()

    def test_report_contains_educational_disclaimer(self, valid_formulation, calculated_properties):
        """Test report contains educational disclaimer."""
        result = generate_report(valid_formulation, calculated_properties)

        # Should have some disclaimer language
        assert "educational" in result.lower() or "estimate" in result.lower() or "theoretical" in result.lower()


# =============================================================================
# TEST EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_minimum_fiber_content(self):
        """Test calculation with minimum fiber content."""
        formulation = Formulation(
            name="min_fiber",
            fiber_type=FiberType.SISAL,
            matrix_type=MatrixType.STARCH_CORN,
            plasticizer_type=PlasticizerType.GLYCEROL,
            fiber_content=5.0,  # Minimum
            matrix_content=70.0,
            plasticizer_content=25.0
        )
        result = calculate_properties(formulation)
        assert result.tensile_strength > 0

    def test_maximum_fiber_content(self):
        """Test calculation with maximum fiber content."""
        formulation = Formulation(
            name="max_fiber",
            fiber_type=FiberType.SISAL,
            matrix_type=MatrixType.STARCH_CORN,
            plasticizer_type=PlasticizerType.GLYCEROL,
            fiber_content=40.0,  # Maximum
            matrix_content=35.0,
            plasticizer_content=25.0
        )
        result = calculate_properties(formulation)
        assert result.tensile_strength > 0

    def test_minimum_plasticizer_content(self):
        """Test calculation with minimum plasticizer content."""
        formulation = Formulation(
            name="min_plasticizer",
            fiber_type=FiberType.SISAL,
            matrix_type=MatrixType.STARCH_CORN,
            plasticizer_type=PlasticizerType.GLYCEROL,
            fiber_content=15.0,
            matrix_content=80.0,
            plasticizer_content=5.0  # Minimum
        )
        result = calculate_properties(formulation)
        assert result.elongation_at_break > 0

    def test_maximum_plasticizer_content(self):
        """Test calculation with maximum plasticizer content."""
        formulation = Formulation(
            name="max_plasticizer",
            fiber_type=FiberType.SISAL,
            matrix_type=MatrixType.STARCH_CORN,
            plasticizer_type=PlasticizerType.GLYCEROL,
            fiber_content=15.0,
            matrix_content=45.0,
            plasticizer_content=40.0  # Maximum
        )
        result = calculate_properties(formulation)
        assert result.elongation_at_break > 0

    def test_all_fiber_matrix_combinations(self):
        """Test all fiber-matrix combinations work."""
        fiber_types = list(FiberType)[:3]  # Test first 3 to save time
        matrix_types = list(MatrixType)[:3]

        for fiber in fiber_types:
            for matrix in matrix_types:
                formulation = Formulation(
                    name=f"combo_{fiber.name}_{matrix.name}",
                    fiber_type=fiber,
                    matrix_type=matrix,
                    plasticizer_type=PlasticizerType.GLYCEROL,
                    fiber_content=15.0,
                    matrix_content=60.0,
                    plasticizer_content=25.0
                )
                result = calculate_properties(formulation)
                assert result.tensile_strength > 0, \
                    f"Failed for {fiber.name}/{matrix.name}"


# =============================================================================
# TEST VERSION AND METADATA
# =============================================================================

class TestMetadata:
    """Tests for version and metadata."""

    def test_version_exists(self):
        """Test VERSION constant exists."""
        assert VERSION is not None
        assert isinstance(VERSION, str)

    def test_version_format(self):
        """Test VERSION follows semantic versioning."""
        parts = VERSION.split(".")
        assert len(parts) >= 3
        # First two parts should be numeric
        assert parts[0].isdigit()
        assert parts[1].isdigit()


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
