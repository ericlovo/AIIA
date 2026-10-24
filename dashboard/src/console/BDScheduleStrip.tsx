import { plannedWaiting, scheduleTime, type BDSnapshot } from './bdModel'

export function BDScheduleStrip({ data, now }: { data: BDSnapshot; now: number }) {
  const schedule = data.schedule
  if (!schedule) return null
  const waiting = plannedWaiting(schedule, now)
  const state = waiting ? 'Planned start' : schedule.state === 'paused' || data.kill_switch && schedule.state !== 'expired' ? 'Paused' : schedule.state === 'expired' ? 'Window ended' : schedule.state === 'scheduled' ? 'Scheduled' : schedule.state === 'waiting' ? 'Awaiting schedule update' : 'Disabled'
  return <section className="bd-schedule-strip" aria-label="Research schedule">
    <div className="bd-schedule-state"><span className="bd-machine-label">RESEARCH SCHEDULE</span><strong>{state}</strong><small>{schedule.timezone}</small></div>
    <dl><div><dt>{waiting ? 'Planned first run' : 'Next planned run'}</dt><dd>{scheduleTime(schedule.next_run_at, schedule.timezone)}</dd></div><div><dt>Scheduled stop</dt><dd>{scheduleTime(schedule.stop_at, schedule.timezone)}</dd></div></dl>
    <p>{waiting && data.kill_switch ? 'Currently paused. Planned times do not resume paused work.' : data.kill_switch ? 'Currently paused. Planned times do not resume paused work.' : 'Planned research window.'}</p>
  </section>
}
