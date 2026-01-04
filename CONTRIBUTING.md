# Contributing to Stock Analysis Multi-Agent System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

### Our Standards

- Be respectful and constructive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/stock-agent-system-final.git
   cd stock-agent-system-final
   ```
3. **Add upstream remote:**
   ```bash
   git remote add upstream https://github.com/original-owner/stock-agent-system-final.git
   ```

## Development Setup

### Prerequisites

- Python 3.11+
- Git
- Virtual environment tool (venv, conda, etc.)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black isort flake8 mypy pre-commit

# Install pre-commit hooks
pre-commit install
```

### Environment Setup

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
nano .env
```

## How to Contribute

### Reporting Bugs

Before creating a bug report:
- Check the issue tracker for existing reports
- Verify the bug exists in the latest version
- Collect relevant information (error messages, logs, etc.)

**Bug Report Template:**

```markdown
**Description:**
A clear description of the bug.

**Steps to Reproduce:**
1. Step one
2. Step two
3. ...

**Expected Behavior:**
What you expected to happen.

**Actual Behavior:**
What actually happened.

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11.0]
- Package versions: [relevant package versions]

**Additional Context:**
Any other relevant information.
```

### Suggesting Features

Feature suggestions are welcome! Please:
- Check if the feature has already been suggested
- Provide a clear use case
- Explain why this feature would be useful
- Consider implementation complexity

**Feature Request Template:**

```markdown
**Feature Description:**
A clear description of the proposed feature.

**Use Case:**
Why this feature would be useful.

**Proposed Implementation:**
(Optional) Ideas for how to implement this.

**Alternatives Considered:**
Other approaches you've considered.
```

### Contributing Code

1. **Find or create an issue** to work on
2. **Comment on the issue** to let others know you're working on it
3. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** following coding standards
5. **Write tests** for your changes
6. **Run tests** to ensure everything works:
   ```bash
   pytest
   ```
7. **Commit your changes:**
   ```bash
   git commit -m "feat: add your feature description"
   ```
8. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```
9. **Create a Pull Request** on GitHub

## Pull Request Process

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] No merge conflicts with main branch

### PR Description Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Related Issue
Fixes #(issue number)

## Testing
Describe how you tested your changes.

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] Self-review completed
```

### Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, a maintainer will merge your PR

## Coding Standards

### Python Style Guide

Follow [PEP 8](https://pep8.org/) style guide:

```bash
# Format code with black
black .

# Sort imports with isort
isort .

# Check style with flake8
flake8 .

# Type checking with mypy
mypy .
```

### Code Formatting

- **Line length:** 88 characters (black default)
- **Indentation:** 4 spaces
- **Quotes:** Double quotes for strings
- **Imports:** Grouped and sorted (stdlib, third-party, local)

### Naming Conventions

- **Variables/Functions:** `snake_case`
- **Classes:** `PascalCase`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private methods:** `_leading_underscore`

### Documentation

Use Google-style docstrings:

```python
def analyze_symbol(symbol: str, use_supervisor: bool = False) -> Dict:
    """
    Analyze a stock symbol using multi-agent system.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        use_supervisor: Whether to use supervisor for routing
    
    Returns:
        Dictionary containing analysis results with keys:
        - recommendation: Trading recommendation
        - confidence: Confidence score (0-1)
        - reasoning: Explanation of decision
    
    Raises:
        ValueError: If symbol is invalid
        APIError: If external API call fails
    
    Example:
        >>> result = analyze_symbol('AAPL', use_supervisor=True)
        >>> print(result['recommendation'])
        'buy'
    """
    pass
```

## Testing Guidelines

### Writing Tests

- Write tests for all new features
- Maintain or improve code coverage
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)

### Test Structure

```python
def test_feature_name():
    """Test description."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_value
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_news_agent.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run with verbose output
pytest -v
```

### Test Coverage

- Aim for >80% code coverage
- Focus on critical paths
- Test edge cases and error handling

## Documentation

### Code Documentation

- Add docstrings to all public functions/classes
- Keep docstrings up to date with code changes
- Include examples in docstrings

### README Updates

Update README.md when:
- Adding new features
- Changing installation process
- Modifying usage instructions

### Documentation Files

Update relevant docs in `docs/`:
- `ARCHITECTURE.md` - For architectural changes
- `API_DOCUMENTATION.md` - For API changes
- `TRAINING.md` - For training pipeline changes
- `DEPLOYMENT.md` - For deployment changes

## Commit Message Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```bash
# Feature
git commit -m "feat(agents): add sentiment analysis to news agent"

# Bug fix
git commit -m "fix(api): resolve timeout issue in batch analysis"

# Documentation
git commit -m "docs: update API documentation with new endpoints"

# Breaking change
git commit -m "feat(api)!: change response format for analysis endpoint

BREAKING CHANGE: Response now includes additional metadata fields"
```

## Questions?

If you have questions:
- Check existing documentation
- Search closed issues
- Open a new issue with the "question" label
- Join our community discussions

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing! 🎉
