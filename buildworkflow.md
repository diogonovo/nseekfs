# 1. Development setup
make dev          # Install dev dependencies
make install      # Build and install locally

# 2. Quality checks
make format       # Format code (Black + Rust)
make lint         # Lint code (Ruff + Clippy)
make check        # Full quality check

# 3. Build distributions
make wheels       # Build wheels for current platform
make sdist        # Build source distribution

# 4. Testing
make test-fast    # Quick functionality test
make test         # Full test suite
make benchmark    # Performance testing

# 5. Validation
python scripts/validate_distributions.py

# 6. Publishing
make upload-test  # Upload to Test PyPI
make upload       # Upload to PyPI (production)