/**
 * OpenClaw Skill: AIIA Summarize
 *
 * Summarize any text using the local LLM — no API cost.
 * You can paste text, send a URL, or reference a file.
 *
 * Usage (via messaging):
 *   "Summarize this: [paste text]"
 *   "Give me bullet points on this article"
 *   "TLDR this document"
 */

export const skill = {
  name: "aiia-summarize",
  description: "Summarize text using the local LLM (free, private, fast)",
  triggers: ["summarize", "tldr", "bullet points", "give me the gist", "sum up"],

  parameters: {
    text: { type: "string", description: "Text to summarize", required: true },
    style: {
      type: "string",
      description: "Summary style: concise, detailed, or bullet_points",
      default: "concise",
    },
  },

  async execute(context, { text, style = "concise" }) {
    const BRAIN_URL = process.env.LOCAL_BRAIN_URL || "http://localhost:8100";

    try {
      const response = await fetch(`${BRAIN_URL}/v1/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          style: style,
          max_length: 200,
        }),
      });

      if (!response.ok) {
        throw new Error(`Local Brain returned ${response.status}`);
      }

      const data = await response.json();

      return {
        message: `**Summary** (${style}, ${data.latency_ms?.toFixed(0)}ms, local):\n\n${data.summary}`,
        data: data,
      };
    } catch (e) {
      return {
        message: `Could not summarize locally: ${e.message}. Is the Local Brain running?`,
        error: true,
      };
    }
  },
};
