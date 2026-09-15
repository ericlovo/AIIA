export function runTokens(run: { input_tokens?: number | null; output_tokens?: number | null }): string {
  if (run.input_tokens == null || run.output_tokens == null) return 'Tokens unrecorded'
  return `${(run.input_tokens + run.output_tokens).toLocaleString('en-US')} tokens`
}
