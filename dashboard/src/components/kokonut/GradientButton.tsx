import type { ComponentProps } from 'react'

// Emerald variant adapted from Kokonut UI's gradient-button registry component.
export function GradientButton({ children, className = '', ...props }: ComponentProps<'button'>) {
  return <button type="button" {...props} className={`group relative min-h-11 overflow-hidden rounded-xl border-2 border-transparent bg-linear-to-b from-[#336C4F] via-[#0C1F21] to-[#0D6437] bg-origin-border px-4 py-2 text-sm font-semibold text-[#a8f2d4] transition-transform duration-150 active:scale-[.98] disabled:opacity-50 motion-reduce:transition-none motion-reduce:active:scale-100 ${className}`}>
    <span aria-hidden="true" className="pointer-events-none absolute inset-0 rounded-[inherit] bg-linear-to-br from-[#274e40] via-[#102c30] to-[#163c30] shadow-[inset_0_0_10px_rgba(135,246,183,0.1)]" />
    <span className="relative inline-flex items-center justify-center gap-2">{children}</span>
  </button>
}
