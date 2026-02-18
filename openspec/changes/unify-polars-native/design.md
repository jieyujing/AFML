# Design: Unify Polars Native

## Context

The AFML project migrated from pandas to Polars for performance but the test suite was not updated. Tests use pandas fixtures and expect pandas return types, while implementations return Polars. This creates a fundamental incompatibility that prevents the test suite from passing.

**Current State:**
- Implementation: Fully Polars native (DataFrame, LazyFrame, Series)
- Tests: Use pandas fixtures (pd.DataFrame, pd.Series)
- Test expectations: Expect pandas return types and attributes

**Constraints:**
- Must maintain sklearn compatibility (fit/transform pattern)
- Must preserve existing API surface
- Cannot break existing user code

## Goals / Non-Goals

**Goals:**
1. Fix 70% test failure rate by aligning test infrastructure with Polars implementation
2. Add missing attributes that tests expect
3. Fix type conversion bug in labeling.py
4. Clean up unused imports

**Non-Goals:**
- Not a pandas → Polars migration (already done in implementation)
- Not adding new features
- Not changing public API behavior

## Decisions

### Decision 1: Convert Test Fixtures to Polars

**Choice:** Modify test fixtures to return Polars DataFrames instead of pandas.

**Rationale:** 
- Implementation is already Polars native
- Tests should validate Polars behavior
- Maintains consistency with project direction

**Alternative considered:** Add pandas-to-Polars conversion layer in every processor
- Rejected: Adds overhead, goes against "Polars Native" goal

### Decision 2: Add Missing Attributes to Processors

**Choice:** Add missing attributes to match test expectations (`ffd_check_stationarity`, `concurrency_`, `uniqueness_`, etc.)

**Rationale:**
- Tests check for these attributes
- Attributes can be simple pass-through or computed values
- Maintains sklearn-compatible interface

### Decision 3: Fix Labeling Type Conversion

**Choice:** Add pandas-to-Polars conversion in TripleBarrierLabeler.fit() method

**Rationale:**
- Critical bug causing AttributeError
- Simple fix: check for pandas and convert
- Prevents runtime errors

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Breaking user code if attributes change existing behavior | Add new attributes only, don't modify existing |
| Test complexity increases | Single source of truth (Polars) reduces complexity |
| Maintenance burden | Fixtures updated once, consistent thereafter |

## Migration Plan

1. **Phase 1:** Fix critical bug in labeling.py (type conversion)
2. **Phase 2:** Add missing attributes to processors
3. **Phase 3:** Rename columns to match test expectations
4. **Phase 4:** Clean up unused imports
5. **Phase 5:** Convert test fixtures to Polars

No rollback needed - changes are additive and backward compatible.
