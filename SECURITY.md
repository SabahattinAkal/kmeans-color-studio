# Security Policy

Supported security fixes target the latest release on `main`.

Report vulnerabilities through GitHub's private vulnerability-reporting interface when available. Do not publish API keys, private images, exploit details, or personal data in a public issue. If private reporting is unavailable, open a minimal issue requesting a private contact channel.

The demo API decodes uploaded files in memory and does not persist them. Deployers remain responsible for request-size limits, authentication, rate limiting, TLS, logging policy, dependency updates, and container isolation.
