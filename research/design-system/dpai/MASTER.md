# DPAI Design System (Approved Direction — Reconciled)

**Concept:** Audio Inpainting Research Platform. Two modes:
- **Light administrative shell** (landing, dashboard, history, settings) — clean, SaaS-like
- **Focused audio workbench** (workspace, results) — lab instrument, dark-mode capable

The waveform and spectrogram are the hero in the workbench. Everything else is quiet.

---

## Color

### Light Theme (Default)
| Role | Hex | CSS Variable |
|---|---|---|
| Page Background | #F5F5F0 | --bg |
| Card Surface | #FFFFFF | --surface |
| Raised/Hover | #FAFAF8 | --raised |
| Border | rgba(0,0,0,0.08) | --border |
| Border Strong | rgba(0,0,0,0.14) | --border-strong |
| Primary Text | #1D1D1B | --fg |
| Muted Text | #6B6B66 | --fg-muted |
| **Primary (Purple)** | #6D28D9 | --primary |
| Primary Hover | #5B21B6 | --primary-hover |
| Primary Foreground | #FFFFFF | --primary-foreground |
| Manual Missing | #6D28D9 | --manual |
| Auto-Detected Missing | #0D9488 | --missing |
| Reconstructed | #EA580C | --recon |
| Danger | #DC2626 | --danger |
| Focus Ring | #6D28D9 | --ring |

### Dark Theme (Workbench Mode)
| Role | Hex | CSS Variable |
|---|---|---|
| Page Background | #0E0F11 | --bg |
| Card Surface | #16181B | --surface |
| Raised/Hover | #1D2024 | --raised |
| Border | rgba(255,255,255,0.07) | --border |
| Border Strong | rgba(255,255,255,0.14) | --border-strong |
| Primary Text | #E8E6E1 | --fg |
| Muted Text | #8A8F98 | --fg-muted |
| **Primary (Purple)** | #8B5CF6 | --primary |
| Primary Hover | #A78BFA | --primary-hover |
| Primary Foreground | #0E0F11 | --primary-foreground |
| Manual Missing | #8B5CF6 | --manual |
| Auto-Detected Missing | #3DD6C6 | --missing |
| Reconstructed | #FFB020 | --recon |
| Danger | #E5484D | --danger |
| Focus Ring | #8B5CF6 | --ring |

**Rules:**
- Only `--missing`/`--recon`/`--manual` are saturated accent colours
- No gradients in UI (spectrogram colormap is magma)
- Keep surrounding UI neutral
- Region colours always have legend; never rely on colour alone

---

## Typography

- **UI + Headings:** Inter (400, 500, 600)
- **Numbers, IDs, timestamps, metrics:** IBM Plex Mono, tabular figures (`font-feature-settings: 'tnum'`)
- **Small uppercase mono labels** (11px, letter-spacing .08em) for panel titles
- Import: `https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap`

---

## Shape & Depth

| Element | Radius |
|---|---|
| Buttons | 10–12px (`rounded-md` → `rounded-lg`) |
| Cards/Panels | 12px (`rounded-lg`) |
| Inputs | 8px (`rounded-md`) |
| Badges/Pills | 4px (`rounded-xs`) |

- Subtle shadows: `shadow-card` (0 1px 3px), `shadow-elevated` (0 4px 6px)
- Separate regions with 1px borders + surface tone
- Workbench panels share edges like instrument rack (no card gaps)

---

## Spacing Scale

4 / 8 / 12 / 16 / 24 / 32 px. Panel padding 12–16px. Consistent 4px base unit.

---

## Components

| Component | Light Spec | Dark Spec (Workbench) |
|---|---|---|
| Primary Button | `--primary` bg, white text, 600 weight, 10px radius | `--primary` bg, `--bg` text, 600 weight, 10px radius |
| Secondary Button | `--surface` bg, 1px `--border`, `--fg` text; hover `--raised` | `--raised` bg, 1px `--border`, `--fg` text; hover `--border-strong` |
| Inputs | `--bg` bg, 1px `--border`, mono for numbers; focus 2px `--ring` | `--surface` bg, 1px `--border`, mono for numbers; focus 2px `--ring` |
| Gap Regions (waveform) | Manual: `--manual` 25% fill + 1px edge | Manual: `--manual` 25% fill + 1px edge |
|  | Auto: `--missing` 25% fill + 1px edge | Auto: `--missing` 25% fill + 1px edge |
|  | Reconstructed: `--recon` 25% fill + 1px edge | Reconstructed: `--recon` 25% fill + 1px edge |
| Progress Bar | 2px `--primary` | 2px `--recon` |
| Metrics | Large mono numeral, small muted label | Large mono numeral, small muted label |

---

## Motion

- 120–180ms ease-out for hover/focus/transitions
- Progress and loss curve update live (60fps target)
- No bouncing, no parallax
- Respect `prefers-reduced-motion`

---

## Layout

### Administrative Shell (Light, Default)
- Topbar: page context, backend status, running job indicator
- Collapsible sidebar (icon-only when collapsed, drawer on mobile)
- Main content: max-width 7xl, generous padding
- Cards with subtle shadows, clear visual hierarchy

### Audio Workbench (Dark-Capable)
- Two-column: ~60-65% waveform/editor, 30-35% config
- WaveSurfer.js waveform with custom region interactions
- Synchronized spectrogram below waveform (unlinkable)
- Docked inspector: presets → basic → advanced → telemetry → metrics
- Bottom: A/B player + spectrogram triptych + loss curve

---

## Accessibility Checklist

- [ ] Contrast 4.5:1 minimum (both themes)
- [ ] Region colours have legend + edge + label (never colour alone)
- [ ] Visible focus on every control (`focus-visible: ring-2 ring-[var(--ring)]`)
- [ ] Keyboard-operable gap editor (arrow keys, Enter/Space, Delete)
- [ ] `cursor:pointer` on all clickable elements
- [ ] `prefers-reduced-motion` respected
- [ ] Tested at 375, 768, 1024, 1440px; no horizontal page scroll
- [ ] Empty, loading, error, cancelled, demo-only states designed
- [ ] Semantic HTML: `<nav>`, `<main>`, `<aside>`, `<header>`, `<footer>`
- [ ] ARIA labels on icon-only buttons
- [ ] Audio controls keyboard/touch accessible

---

## Anti-Patterns (Both Modes)

- No purple/pink gradients, no glassmorphism, no neumorphism
- No emoji icons (use Lucide React SVG)
- No border radius above 12px (cards) or 12px (buttons)
- No pill buttons (use rounded-md/lg)
- No hover transforms that shift layout
- No fake data/metrics — empty states are honest

---

## Reconciliation Notes

| Previous (Workbench MASTER) | Approved Spec | Resolution |
|---|---|---|
| Dark-only (#0E0F11 bg) | Light default, dark for workbench | **Dual-mode**: light shell, dark workbench |
| 2px radius everywhere | 10-12px buttons, 12px cards | **Tiered radius** by component |
| No shadows | Subtle shadows | **shadow-card / shadow-elevated** |
| Dense instrument rack | Generous admin spacing | **Mode-aware spacing** |
| `--recon` as primary | Purple (`#6D28D9`) as primary | **Purple primary**, `--recon` for recon regions only |
| IBM Plex Sans only | Inter for UI, Plex Mono for data | **Inter + Plex Mono** |
| No landing page | Public landing required | **Added** |
| No experiment history UI | Required | **Added** |

---

## Chart Tokens (Recharts)

```js
const CHART_COLORS = {
  primary: 'hsl(var(--primary))',      // purple
  missing: 'hsl(var(--missing))',      // teal
  recon: 'hsl(var(--recon))',          // orange
  manual: 'hsl(var(--manual))',        // purple
  muted: 'hsl(var(--fg-muted))',
  grid: 'hsl(var(--border))',
  bg: 'hsl(var(--bg))',
  surface: 'hsl(var(--surface))',
};
```

- Axis ticks: `fontFamily: 'IBM Plex Mono'`, size 11-12px
- Tooltips: `--surface` bg, `--border` border, `--fg` text
- Legend: 12px Inter, `--fg-muted`
- ResponsiveContainer fills parent