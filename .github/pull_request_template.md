# Pull Request

## 📋 Description

<!-- Provide a brief description of the changes in this PR -->

## 🔗 Related Issues

<!-- Link to related issues using keywords like "Fixes #123" or "Relates to #456" -->
- Fixes #
- Relates to #

## 🎯 Type of Change

<!-- Mark the relevant option with an "x" -->
- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🧹 Code cleanup/refactoring
- [ ] ⚡ Performance improvement
- [ ] 🧪 Test coverage improvement
- [ ] 🔧 Build/CI improvement

## 🧪 Testing

<!-- Describe how you tested your changes -->

### Test Environment
- [ ] Ubuntu/Linux
- [ ] macOS
- [ ] Windows
- [ ] Python 3.8
- [ ] Python 3.9
- [ ] Python 3.10
- [ ] Python 3.11
- [ ] Python 3.12

### Tests Added/Modified
- [ ] Unit tests
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Examples updated

### Manual Testing
<!-- Describe manual testing performed -->
```python
# Example test code
import nseekfs
import numpy as np

# Your test code here...
```

## 📈 Performance Impact

<!-- If applicable, describe performance impact -->
- [ ] No performance impact
- [ ] Performance improvement: <!-- describe -->
- [ ] Performance regression: <!-- describe and justify -->
- [ ] Performance testing required

**Benchmark Results** (if applicable):
```
Before: X ms
After:  Y ms
Change: Z% improvement/regression
```

## 🔄 API Changes

<!-- If this PR introduces API changes, document them -->
- [ ] No API changes
- [ ] Backward compatible API addition
- [ ] Backward compatible API modification
- [ ] Breaking API change (requires major version bump)

### API Changes Summary
<!-- List new/modified/removed methods, parameters, etc. -->

## 📚 Documentation

- [ ] Code is well-commented
- [ ] Docstrings updated
- [ ] README.md updated (if needed)
- [ ] Examples updated (if needed)
- [ ] CHANGELOG.md updated

## ✅ Checklist

<!-- Mark completed items with an "x" -->

### Code Quality
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Code is well-documented
- [ ] No debug prints or commented code left
- [ ] No unnecessary dependencies added

### Functionality
- [ ] All tests pass locally
- [ ] New tests cover the changes
- [ ] Existing functionality not broken
- [ ] Error handling implemented appropriately
- [ ] Edge cases considered

### Security & Safety
- [ ] No security vulnerabilities introduced
- [ ] Input validation implemented
- [ ] No sensitive data exposed
- [ ] Memory safety maintained (for Rust code)

### Compatibility
- [ ] Backward compatibility maintained (or breaking change justified)
- [ ] Cross-platform compatibility verified
- [ ] Python version compatibility maintained

## 📝 Additional Notes

<!-- Any additional information, context, or notes for reviewers -->

### Breaking Changes
<!-- If this is a breaking change, document migration steps -->

### Dependencies
<!-- List any new dependencies and justify their inclusion -->

### Future Work
<!-- Note any related work that should be done in the future -->

---

## 👥 Reviewer Guidelines

### Focus Areas
<!-- Suggest specific areas for reviewers to focus on -->
- [ ] Algorithm correctness
- [ ] Performance implications
- [ ] API design
- [ ] Documentation quality
- [ ] Test coverage

### Testing Instructions
<!-- Provide specific instructions for testing this PR -->

1. Install the branch: `pip install git+https://github.com/diogonovo/nseekfs.git@branch-name`
2. Run test suite: `python test_func.py`
3. Test specific functionality: <!-- provide specific test cases -->

---

**Thank you for contributing to NSeekFS! 🚀**