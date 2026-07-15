---
type: guide
title: Troubleshooting
tags: [troubleshooting, errors, support]
description: Common issues and resolution steps for Acme products.
---

# Troubleshooting

## Payment Failures

| Error | Cause | Fix |
|-------|-------|-----|
| `CARD_DECLINED` | Insufficient funds or card expired | Update payment method in [Billing](billing.md) |
| `BANK_REJECTED` | ACH authorization not completed | Contact your bank, then retry |
| `FRAUD_BLOCK` | Bank's fraud detection triggered | Call your bank to approve the charge |

## Login Issues

- **"Invalid credentials"** — reset your password at acme.com/reset
- **"Account locked"** — wait 30 minutes or contact support
- **Email not verified** — check spam folder, or re-send from [Account Setup](account-setup.md)

## Performance Issues

- Clear browser cache and cookies
- Try an incognito/private window
- Check our status page at status.acme.com
