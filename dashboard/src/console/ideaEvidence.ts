export function groupIdeaEvidence<T extends { source: string; project?: string; text: string }>(ideas: T[]): T[][] {
  const groups = new Map<string, T[]>()
  for (const idea of ideas) {
    const claim = idea.text.replace(/\n\nFound by the code review loop in commit [a-f0-9]+\.\s*$/, '')
    const key = JSON.stringify([idea.source, idea.project, claim])
    groups.set(key, [...(groups.get(key) ?? []), idea])
  }
  return [...groups.values()]
}
