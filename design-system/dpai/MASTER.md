# DPAI Design System (hand-written override)

**Concept:** Audio-restoration workbench. A lab instrument, not a SaaS page.
The waveform and spectrogram are the hero. Everything else is quiet.

## Color
| Role | Hex | Variable |
|---|---|---|
| Background | #0E0F11 | --bg |
| Surface | #16181B | --surface |
| Raised | #1D2024 | --raised |
| Border | rgba(255,255,255,0.07) | --border |
| Foreground | #E8E6E1 | --fg |
| Muted text | #8A8F98 | --fg-muted |
| Missing / gap (matches spec PNG overlay) | #3DD6C6 | --missing |
| Reconstruction | #FFB020 | --recon |
| Danger | #E5484D | --danger |
| Focus ring | #FFB020 | --ring |

Rules: only --missing and --recon are saturated. No gradients. Spectrogram
colormap is magma, so keep surrounding UI neutral.

## Typography
- UI + headings: IBM Plex Sans (400, 500, 600)
- Numbers, labels, metrics, timestamps: IBM Plex Mono, tabular figures
- Small uppercase mono labels (11px, letter-spacing .08em) for panel titles
- Import: https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap

## Shape and depth
- Radius: 2px (inputs, buttons), 4px (panels). Nothing larger.
- No drop shadows. Separate regions with 1px borders and surface tone.
- Dense layout: panels share edges like an instrument rack, not floating cards.

## Spacing
4 / 8 / 12 / 16 / 24 / 32 px. Panel padding 12-16px.

## Components
- Primary button: --recon background, #0E0F11 text, 600 weight, 2px radius.
- Secondary button: transparent, 1px --border, --fg text; hover raises to --raised.
- Inputs: --surface bg, 1px --border, mono font for numeric fields;
  focus = 2px --ring outline offset 2px.
- Gap regions on waveform: --missing at 25% fill with 1px solid edges and
  drag handles.
- Progress: thin 2px bar in --recon, mono percentage and epoch count beside it.
- Metrics: large mono numeral, small muted label above.

## Motion
120-180ms ease-out for hover/focus. Progress and loss curve update live.
No bouncing, no parallax. Respect prefers-reduced-motion.

## Layout (single-screen workbench, desktop)
Top bar (title, device, queue status)
Main left: waveform with gap editor, below it spectrogram
Main right: preset + epochs + seed + run/cancel, live loss curve, metrics
Bottom: A/B player (reconstruction / corrupted / clean), then examples gallery
Mobile: stack in the same order; controls collapse into one panel.

## Anti-patterns
- No purple/pink gradients, no glassmorphism, no neumorphism
- No emoji icons (use Lucide/Phosphor SVG)
- No centered hero, no three-feature card grid, no landing-page sections
- No border radius above 4px, no pill buttons
- No hover transforms that shift layout

## Checklist
- [ ] Contrast 4.5:1 minimum; gap color not the only cue (add edge + label)
- [ ] Visible focus on every control, keyboard-operable gap editor
- [ ] cursor:pointer on clickable elements
- [ ] prefers-reduced-motion respected
- [ ] Tested at 375, 768, 1024, 1440px; no horizontal page scroll
- [ ] Empty, loading, error, cancelled, and demo-only states designed