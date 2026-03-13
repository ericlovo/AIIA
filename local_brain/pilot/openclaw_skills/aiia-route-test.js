/**
 * OpenClaw Skill: AIIA Route Test
 *
 * Test the Smart Conductor by sending it queries and seeing
 * how it classifies them. Useful for tuning and debugging routing.
 *
 * Usage (via messaging):
 *   "Route test: compare Q3 revenue to industry benchmarks"
 *   "How would the brain route: I'm worried about cash flow"
 *   "Test routing: help me draft a will"
 */

export const skill = {
  name: "aiia-route-test",
  description: "Test Smart Conductor routing — see how the local brain classifies any query",
  triggers: ["route test", "test routing", "how would the brain route", "classify this"],

  parameters: {
    query: { type: "string", description: "Query to classify", required: true },
    tenant: { type: "string", description: "Tenant ID", default: "default" },
  },

  async execute(context, { query, tenant = "default" }) {
    const BRAIN_URL = process.env.LOCAL_BRAIN_URL || "http://localhost:8100";

    try {
      const response = await fetch(`${BRAIN_URL}/v1/route`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          tenant_id: tenant,
          has_documents: false,
          document_count: 0,
        }),
      });

      if (!response.ok) {
        throw new Error(`Local Brain returned ${response.status}`);
      }

      const r = await response.json();

      const eqEmoji = {
        analyst: "📊",
        guide: "🧭",
        supporter: "💛",
        advocate: "🚨",
      };

      const pathEmoji = {
        local: "🏠",
        eos: "⚡",
        rlm: "🧠",
      };

      let report = `**Smart Conductor Result**\n\n`;
      report += `Query: "${query}"\n\n`;
      report += `${eqEmoji[r.eq_mode] || "❓"} **EQ**: Level ${r.eq_level} (${r.eq_mode.toUpperCase()})\n`;
      report += `🎯 **Domain**: ${r.domain}\n`;
      report += `📈 **Complexity**: ${r.complexity_score}\n`;
      report += `${pathEmoji[r.recommended_path] || "❓"} **Path**: ${r.recommended_path.toUpperCase()}\n`;
      report += `🔮 **Confidence**: ${(r.confidence * 100).toFixed(0)}%\n`;
      report += `⏱️ **Latency**: ${r.latency_ms?.toFixed(0)}ms\n`;
      report += `\n💭 ${r.reasoning}`;

      return { message: report, data: r };
    } catch (e) {
      return {
        message: `Route test failed: ${e.message}. Is the Local Brain running?`,
        error: true,
      };
    }
  },
};
