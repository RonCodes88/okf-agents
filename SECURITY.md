# Security Policy

## Supported Versions

`okf-agents` is pre-1.0 and alpha-quality. Security fixes are made
against the latest release on the `main` branch only.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |

## Reporting a Vulnerability

Please do not open a public GitHub issue for security vulnerabilities.

Instead, report suspected vulnerabilities privately by emailing
**ronaldliyh@gmail.com** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce, or a minimal proof of concept
- The affected version(s) or commit

You should receive an acknowledgment within 5 business days. We will work
with you to understand and validate the issue, and to agree on a disclosure
timeline once a fix is available. Please allow us reasonable time to
release a patch before any public disclosure.

## Scope Notes

`okf-agents` parses local Markdown/YAML bundles and never executes
arbitrary code from bundle content. If you find a way to trigger unsafe
deserialization, path traversal outside a bundle root, or arbitrary code
execution while loading or querying a bundle, that is in scope and
especially appreciated.
