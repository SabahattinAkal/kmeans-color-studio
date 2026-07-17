# Contributing

Bug reports, reproducible benchmarks, documentation fixes, and focused pull requests are welcome.

1. Fork the repository and create a branch from `main`.
2. Create a Python 3.10+ virtual environment.
3. Install the project with `pip install -e ".[web]"`.
4. Run `python -m unittest discover -s tests -v` before opening a pull request.
5. Explain the behavior change, include before/after metrics when performance changes, and add a test for new logic.

Do not commit private images, secrets, model credentials, or third-party assets without a compatible license. Benchmark claims must state image size, palette range, color space, seed, and hardware context.
