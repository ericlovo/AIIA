/**
 * OpenClaw Skill: AIIA PII Scanner
 *
 * Scan any text for PII/PHI using the local model.
 * Data never leaves the machine — perfect for compliance.
 *
 * Usage (via messaging):
 *   "Scan this for PII: [paste text]"
 *   "Check this email for sensitive data"
 *   "PII check: John Smith, SSN 123-45-6789, lives at..."
 */

export const skill = {
  name: "aiia-pii-scan",
  description: "Scan text for PII/PHI locally — data never leaves your machine",
  triggers: ["scan for pii", "pii check", "check for sensitive", "phi scan", "privacy scan"],

  parameters: {
    text: { type: "string", description: "Text to scan for PII", required: true },
  },

  async execute(context, { text }) {
    const BRAIN_URL = process.env.LOCAL_BRAIN_URL || "http://localhost:8100";

    try {
      const response = await fetch(`${BRAIN_URL}/v1/scan-pii`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          categories: ["ssn", "email", "phone", "address", "dob", "medical", "financial"],
        }),
      });

      if (!response.ok) {
        throw new Error(`Local Brain returned ${response.status}`);
      }

      const r = await response.json();

      const riskEmoji = {
        none: "✅",
        low: "🟡",
        medium: "🟠",
        high: "🔴",
        critical: "🚨",
      };

      let report = `**PII Scan Result** (${r.latency_ms?.toFixed(0)}ms, local)\n\n`;
      report += `${riskEmoji[r.risk_level] || "❓"} Risk Level: **${r.risk_level.toUpperCase()}**\n\n`;

      if (r.findings && r.findings.length > 0) {
        report += `Found ${r.findings.length} item(s):\n`;
        r.findings.forEach(f => {
          report += `  - **${f.category}**: ${f.value} (${f.location || "in text"})\n`;
        });
        report += `\n⚠️ This data should be redacted before sharing.`;
      } else {
        report += `No PII detected.`;
      }

      return { message: report, data: r };
    } catch (e) {
      return {
        message: `PII scan failed: ${e.message}. Is the Local Brain running?`,
        error: true,
      };
    }
  },
};
