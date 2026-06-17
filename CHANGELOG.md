# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-01-XX

### Added

- Complete rewrite of GSF model as modern Python package
- Object-oriented model classes (`GSFEnergy`, `GSFRigidity`, `GSFEnergyPerNucleon`, etc.) replacing functional interface
- Modern project structure with `src/globalsplinefit/` layout
- Comprehensive test suite with pytest
- Type hints throughout codebase
- Proper error handling with custom exceptions
- Solar modulation functionality in dedicated module
- Utility functions separated into own module
- Data loading infrastructure with resource management
- PyPI packaging configuration
- Code quality tools (ruff, mypy)
- Comprehensive documentation
- CI/CD pipeline with GitHub Actions

### Changed

- **BREAKING**: Replaced functional interface with specialized model classes
- **BREAKING**: Moved data files to `src/globalsplinefit/data/{version}/` directory
- **BREAKING**: Changed import structure to `from globalsplinefit import GSFEnergy`
- Improved error messages and validation
- Enhanced solar modulation parameter handling
- Better separation of concerns across modules

### Removed

- Legacy `flux.py` functional interface
- Direct access to global variables
- Root-level data files

## [1.x.x] - Previous Versions

Legacy versions with functional interface. See git history for details.
