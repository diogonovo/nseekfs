.PHONY: help install build test clean lint format check wheels sdist upload-test upload dev benchmark

# Default target
help:
	@echo "🚀 NSeekFS Build Commands"
	@echo "========================="
	@echo ""
	@echo "📦 Package Building:"
	@echo "  make build        - Build package for development"
	@echo "  make wheels       - Build wheels for current platform"
	@echo "  make sdist        - Build source distribution"
	@echo "  make clean        - Clean build artifacts"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test         - Run test suite"
	@echo "  make test-fast    - Run basic tests only"
	@echo "  make benchmark    - Run performance benchmarks"
	@echo ""
	@echo "🧹 Code Quality:"
	@echo "  make lint         - Run linting (ruff + clippy)"
	@echo "  make format       - Format code (black + rustfmt)"
	@echo "  make check        - Full quality check"
	@echo ""
	@echo "🚀 Development:"
	@echo "  make dev          - Setup development environment"
	@echo "  make install      - Install package in development mode"
	@echo ""
	@echo "📤 Publishing:"
	@echo "  make upload-test  - Upload to Test PyPI"
	@echo "  make upload       - Upload to PyPI (production)"
	@echo ""
	@echo "🔧 Utilities:"
	@echo "  make deps         - Check dependencies"
	@echo "  make info         - Show system information"

# Variables
PYTHON := python
PIP := pip
MATURIN := maturin
CARGO := cargo

# Check if maturin is installed
check-maturin:
	@which $(MATURIN) > /dev/null || (echo "❌ maturin not found. Install with: pip install maturin" && exit 1)

# Check dependencies
deps:
	@echo "🔍 Checking dependencies..."
	@$(PYTHON) --version || (echo "❌ Python not found" && exit 1)
	@$(CARGO) --version || (echo "❌ Cargo not found" && exit 1)
	@$(MATURIN) --version || (echo "❌ Maturin not found" && exit 1)
	@echo "✅ All dependencies found"

# Show system information
info:
	@echo "🖥️  System Information:"
	@echo "Platform: $$(uname -s -m)"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Rust: $$($(CARGO) --version)"
	@echo "Maturin: $$($(MATURIN) --version)"

# Setup development environment
dev: check-maturin
	@echo "🔧 Setting up development environment..."
	$(PIP) install --upgrade pip
	$(PIP) install maturin
	$(PIP) install pytest numpy black ruff mypy
	@echo "✅ Development environment ready"

# Install package in development mode
install: check-maturin
	@echo "📦 Installing package in development mode..."
	$(MATURIN) develop --release
	@echo "✅ Package installed"

# Build package for development
build: check-maturin
	@echo "🔨 Building package for development..."
	$(MATURIN) develop --release
	@echo "✅ Build complete"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf dist/
	rm -rf target/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf nseekfs/__pycache__/
	rm -f build_report.json
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.pyd" -delete
	find . -name "*.so" -delete
	find . -name "*.dylib" -delete
	find . -name "*.dll" -delete
	@echo "✅ Clean complete"

# Build wheels
wheels: check-maturin clean
	@echo "🛞 Building wheels..."
	$(PYTHON) scripts/build_wheels.py
	@echo "✅ Wheels built"

# Build source distribution
sdist: check-maturin clean
	@echo "📦 Building source distribution..."
	$(MATURIN) sdist --out dist
	@echo "✅ Source distribution built"

# Format code
format:
	@echo "🎨 Formatting code..."
	black nseekfs/ examples/ scripts/ --line-length 100
	$(CARGO) fmt --all
	@echo "✅ Formatting complete"

# Lint code
lint:
	@echo "🔍 Linting code..."
	ruff check nseekfs/ examples/ scripts/
	$(CARGO) clippy --all-targets -- -D warnings
	@echo "✅ Linting complete"

# Type check
typecheck:
	@echo "🔬 Type checking..."
	mypy nseekfs/ --ignore-missing-imports
	@echo "✅ Type checking complete"

# Full quality check
check: format lint typecheck
	@echo "✅ All quality checks passed"

# Run basic tests
test-fast: install
	@echo "🧪 Running basic tests..."
	$(PYTHON) -c "import nseekfs; print('✅ Import successful')"
	$(PYTHON) -c "
import numpy as np
import nseekfs
embeddings = np.random.randn(50, 32).astype(np.float32)
index = nseekfs.from_embeddings(embeddings)
results = index.query(embeddings[0], top_k=5)
assert len(results) == 5
assert results[0]['idx'] == 0
print('✅ Basic functionality test passed')
"
	@echo "✅ Fast tests complete"

# Run full test suite
test: install
	@echo "🧪 Running comprehensive test suite..."
	NSEEK_SKIP_200K=1 NSEEK_TEST_SMALL_N=50 NSEEK_TEST_MED_N=500 $(PYTHON) test_func.py
	@echo "✅ Full test suite complete"

# Run benchmarks
benchmark: install
	@echo "⚡ Running performance benchmarks..."
	$(PYTHON) -c "
import time
import numpy as np
import nseekfs

print('📊 Performance Benchmarks')
print('=' * 30)

sizes = [1000, 5000, 10000]
dims = 384

for n in sizes:
    print(f'Testing {n} vectors ({dims}D)...')
    embeddings = np.random.randn(n, dims).astype(np.float32)
    
    # Build time
    start = time.time()
    index = nseekfs.from_embeddings(embeddings, ann=True)
    build_time = time.time() - start
    
    # Query time (average of 10 queries)
    query = embeddings[0]
    start = time.time()
    for _ in range(10):
        results = index.query(query, top_k=10)
    avg_query_time = (time.time() - start) / 10 * 1000
    
    print(f'  Build: {build_time:.2f}s | Query: {avg_query_time:.2f}ms')

print('✅ Benchmarks complete')
"

# Upload to Test PyPI
upload-test: wheels sdist
	@echo "🧪 Uploading to Test PyPI..."
	@if [ -z "$$TEST_PYPI_TOKEN" ]; then \
		echo "❌ TEST_PYPI_TOKEN environment variable not set"; \
		echo "Set it with: export TEST_PYPI_TOKEN=pypi-..."; \
		exit 1; \
	fi
	$(PIP) install --upgrade twine
	twine upload --repository testpypi dist/* --username __token__ --password $$TEST_PYPI_TOKEN
	@echo "✅ Uploaded to Test PyPI"
	@echo "🔗 Check: https://test.pypi.org/project/nseekfs/"

# Upload to PyPI (production)
upload: wheels sdist
	@echo "🚀 Uploading to PyPI (production)..."
	@if [ -z "$$PYPI_TOKEN" ]; then \
		echo "❌ PYPI_TOKEN environment variable not set"; \
		echo "Set it with: export PYPI_TOKEN=pypi-..."; \
		exit 1; \
	fi
	@read -p "⚠️  Are you sure you want to upload to PRODUCTION PyPI? (y/N): " confirm && [ "$$confirm" = "y" ]
	$(PIP) install --upgrade twine
	twine upload dist/* --username __token__ --password $$PYPI_TOKEN
	@echo "✅ Uploaded to PyPI"
	@echo "🔗 Check: https://pypi.org/project/nseekfs/"

# Test installation from Test PyPI
test-testpypi:
	@echo "🧪 Testing installation from Test PyPI..."
	$(PIP) install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nseekfs --force-reinstall
	$(PYTHON) -c "
import nseekfs
import numpy as np
print(f'✅ Installed nseekfs {nseekfs.__version__} from Test PyPI')

# Quick test
embeddings = np.random.randn(50, 32).astype(np.float32)
index = nseekfs.from_embeddings(embeddings)
results = index.query(embeddings[0], top_k=3)
assert len(results) == 3
print('✅ Test PyPI installation working correctly')
"

# Test installation from PyPI
test-pypi:
	@echo "🧪 Testing installation from PyPI..."
	$(PIP) install nseekfs --force-reinstall
	$(PYTHON) -c "
import nseekfs
import numpy as np
print(f'✅ Installed nseekfs {nseekfs.__version__} from PyPI')

# Quick test
embeddings = np.random.randn(50, 32).astype(np.float32)
index = nseekfs.from_embeddings(embeddings)
results = index.query(embeddings[0], top_k=3)
assert len(results) == 3
print('✅ PyPI installation working correctly')
"

# Release workflow
release: check
	@echo "🚀 Starting release workflow..."
	@echo "1. Make sure all tests pass..."
	make test
	@echo "2. Build distributions..."
	make wheels
	make sdist
	@echo "3. Upload to Test PyPI..."
	make upload-test
	@echo "4. Test Test PyPI installation..."
	sleep 60  # Wait for package to be available
	make test-testpypi
	@echo ""
	@echo "🎉 Test PyPI release complete!"
	@echo "🔗 Check: https://test.pypi.org/project/nseekfs/"
	@echo ""
	@echo "To release to production PyPI:"
	@echo "  make upload"

# Development workflow
workflow: dev install test-fast
	@echo "🎉 Development workflow complete!"
	@echo "Ready for development. Try:"
	@echo "  python examples/basic_usage.py"