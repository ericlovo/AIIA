import type { Agent } from '../lib/api'
import { pickerGroups, pickerOptionLabel, resolveUseWhen } from './agentRoster'

export function AgentPicker({
  agents,
  value,
  onChange,
  placeholder,
  excludeIds,
  className,
  id,
  'aria-label': ariaLabel,
}: {
  agents: Agent[]
  value: string
  onChange: (agentId: string) => void
  placeholder: string
  excludeIds?: string[]
  className?: string
  id?: string
  'aria-label'?: string
}) {
  const groups = pickerGroups(agents, excludeIds)
  return (
    <select id={id} aria-label={ariaLabel} className={className} value={value} onChange={event => onChange(event.target.value)}>
      <option value="">{placeholder}</option>
      {groups.map(group => (
        <optgroup key={group.id || 'unsorted'} label={group.label}>
          {group.agents.map(agent => (
            <option key={agent.id} value={agent.id} title={resolveUseWhen(agent)}>
              {pickerOptionLabel(agent)}
            </option>
          ))}
        </optgroup>
      ))}
    </select>
  )
}
