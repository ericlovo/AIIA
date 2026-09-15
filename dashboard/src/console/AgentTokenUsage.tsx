import type { Agent, StudioActivity } from '../lib/api'

interface Props {
  data?: StudioActivity
  agents: Agent[]
  day: string
  status: string
  selectedAgentId: string
  onSelectAgent: (id: string) => void
}

export function AgentTokenUsage({ data, agents, day, status, selectedAgentId, onSelectAgent }: Props) {
  const usage = data?.usage_by_agent
  return <section className="sb-tokens" aria-label="Agent token attribution">
    <div className="sb-section-title"><div><h2>Agent tokens</h2><p>{day || (data ? `${data.start} to ${data.today}` : 'Loading period')} UTC / {status || 'all outcomes'} / {selectedAgentId ? 'selected agent' : 'all agents'}</p></div></div>
    {!usage ? <p className="sb-coverage">{data ? 'Agent attribution is unavailable from this backend.' : 'Loading agent usage...'}</p> : <>
      <div className="sb-token-table"><table>
        <thead><tr><th scope="col">Agent</th><th scope="col">Input</th><th scope="col">Output</th><th scope="col">Measured runs</th></tr></thead>
        <tbody>{usage.map(row => <tr key={row.agent_id}>
          <th scope="row"><button type="button" className="text-left text-cyan-200" aria-pressed={row.agent_id === selectedAgentId} onClick={() => onSelectAgent(row.agent_id)}>{agents.find(agent => agent.id === row.agent_id)?.name ?? row.agent_name}</button></th>
          <td>{row.input_tokens == null ? '--' : row.input_tokens.toLocaleString('en-US')}</td>
          <td>{row.output_tokens == null ? '--' : row.output_tokens.toLocaleString('en-US')}</td>
          <td>{row.measured_runs} / {row.runs}</td>
        </tr>)}</tbody>
      </table></div>
      {usage.length === 0 && <p className="sb-coverage">No recorded runs match these filters.</p>}
      <p className="sb-coverage">Measured input and output from saved runs, including failed outputs when usage was returned. Older runs and missing responses are unrecorded, not zero. Totals follow the activity filters and include all matches, not just the latest 200.</p>
    </>}
  </section>
}
