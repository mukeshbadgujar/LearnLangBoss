# Northwind Analytics - Product FAQ

## Plans and Billing

**What plans are available?**
Starter (INR 4,000/user/month), Growth (INR 9,000/user/month) and Enterprise (custom).
Starter includes 3 data sources, Growth includes 15, Enterprise is unlimited.

**How does mid-cycle plan change work?**
Upgrades apply immediately and are pro-rated on the next invoice. Downgrades take effect
at the start of the next billing cycle; we do not issue partial refunds for the current
cycle.

**What is the trial period?**
14 days on the Growth plan, no credit card required. Trials can be extended once by 7
days through the support team.

## Data Sources and Integrations

**Which sources are supported natively?**
PostgreSQL, MySQL, Snowflake, BigQuery, Redshift, Salesforce, HubSpot, Google Sheets and
S3 (CSV/Parquet).

**How often does data refresh?**
Starter refreshes every 24 hours, Growth every 3 hours, Enterprise every 15 minutes or on
demand via the API.

**What happens when a credential expires?**
The connector retries 3 times with exponential backoff, then marks the source as
"needs attention" and emails the workspace admins. Data is retained, not deleted.

## API and Limits

**What are the API rate limits?**
Starter: 60 requests/minute. Growth: 600 requests/minute. Enterprise: 6,000
requests/minute. Bursts up to 2x the limit are allowed for 10 seconds.

**Is there a webhook system?**
Yes. Webhooks are signed with HMAC-SHA256. Signatures must be verified using the secret
shown once at creation time. Failed deliveries retry for 24 hours.

## Security and Compliance

**Where is data stored?**
Primary region is ap-south-1 (Mumbai). EU customers can request eu-west-1 (Ireland).
Enterprise customers can request a dedicated single-tenant deployment.

**Do you have SOC 2?**
Yes, SOC 2 Type II, audited annually. Reports are available under NDA through the
security portal.

**How long is data retained after cancellation?**
30 days, after which all customer data is permanently deleted. Export can be requested
at any point within those 30 days.

## Support

**What are the support SLAs?**
Starter: 48 business hours first response. Growth: 8 business hours. Enterprise: 1 hour
for critical issues, 24x7.

**How do I escalate?**
Reply to the ticket with "ESCALATE" in the subject, or contact your customer success
manager (Enterprise only).
