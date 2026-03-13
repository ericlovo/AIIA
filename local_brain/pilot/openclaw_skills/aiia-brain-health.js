/**
 * OpenClaw Skill: AIIA Brain Health
 *
 * Monitors the Local Brain stack and reports status.
 * OpenClaw can run this proactively on a schedule or on demand.
 *
 * Usage (via messaging):
 *   "How's the brain?"
 *   "Check system health"
 *   "Is Ollama running?"
 */

export const skill = {
  name: "aiia-brain-health",
  description: "Check the health of the AIIA Local Brain stack (Ollama + Local Brain API)",
  triggers: ["brain health", "system health", "is ollama running", "how's the brain", "check status"],

  async execute(context) {
    const BRAIN_URL = process.env.LOCAL_BRAIN_URL || "http://localhost:8100";
    const OLLAMA_URL = process.env.LOCAL_LLM_URL || "http://localhost:11434";

    const results = {
      timestamp: new Date().toISOString(),
      ollama: null,
      localBrain: null,
      models: [],
      issues: [],
    };

    // Check Ollama
    try {
      const ollamaResp = await fetch(`${OLLAMA_URL}/api/tags`, { signal: AbortSignal.timeout(5000) });
      const ollamaData = await ollamaResp.json();
      const models = ollamaData.models || [];
      results.ollama = "online";
      results.models = models.map(m => `${m.name} (${(m.size / 1e9).toFixed(1)}GB)`);
    } catch (e) {
      results.ollama = "offline";
      results.issues.push("Ollama is not running. Run: brew services start ollama");
    }

    // Check Local Brain API
    try {
      const brainResp = await fetch(`${BRAIN_URL}/health`, { signal: AbortSignal.timeout(5000) });
      const brainData = await brainResp.json();
      results.localBrain = brainData.status || "online";
      results.features = brainData.features || {};
    } catch (e) {
      results.localBrain = "offline";
      results.issues.push("Local Brain API is not running. Run: brain start");
    }

    // Build human-readable report
    let report = `**AIIA Local Brain Status**\n`;
    report += `Ollama: ${results.ollama === "online" ? "running" : "DOWN"}\n`;
    report += `Local Brain API: ${results.localBrain === "online" ? "running" : "DOWN"}\n`;

    if (results.models.length > 0) {
      report += `\nModels loaded:\n`;
      results.models.forEach(m => { report += `  - ${m}\n`; });
    }

    if (results.issues.length > 0) {
      report += `\nIssues:\n`;
      results.issues.forEach(i => { report += `  - ${i}\n`; });
    } else {
      report += `\nAll systems operational.`;
    }

    return { message: report, data: results };
  },
};
