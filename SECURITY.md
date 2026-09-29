# Security and Sensitive Data

## Supported version

Security and data-handling fixes are applied to the latest tagged release.

## Reporting a problem

Do not disclose credentials, private authorization records, or restricted
market data in a public issue. Once the repository is hosted, use the host's
private security-advisory channel or contact the maintainers privately.

## Credential handling

- Treat any token pasted into chat, email, a terminal transcript, or an issue
  as exposed and rotate it at the issuing service.
- Supply credentials through environment variables.
- Never add tokens, `.env` files, private keys, or authorization records to
  version control.

## Data boundary

The public repository is code-only. Consult `DATA_ACCESS_AND_RELEASE.md` before
using or redistributing provider data, model weights trained from those data,
or timestamp-level derivatives.
