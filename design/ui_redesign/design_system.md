# FlexPose Vision Design System — Prototype 0.1

## Direction

**Deep Graphite Industrial Vision** uses restrained graphite surfaces, precise
typography, weak separators and a single blue interaction accent. Images and
detection results carry the visual weight; controls recede until needed.

## Tokens

| Role | Value |
| --- | --- |
| App background | `#0B1018` |
| Navigation | `#0E1520` |
| Panel | `#131C28` |
| Elevated panel | `#182332` |
| Canvas | `#060A10` |
| Border | `#263244` |
| Divider | `#202B3A` |
| Primary | `#3B82F6` |
| Primary hover | `#5795F8` |
| Text primary | `#F1F5F9` |
| Text secondary | `#94A3B8` |
| Text muted | `#64748B` |
| Text disabled | `#475569` |
| Success / Warning / Error / Info | `#22C55E` / `#F59E0B` / `#EF4444` / `#38BDF8` |

## Typography

- Application: 21 px semibold
- Page title: 24 px semibold
- Section title: 13 px semibold with restrained letter spacing
- Body/value: 12 px
- Field label/secondary: 10–11 px
- Primary metric: 30 px semibold
- Preferred families: `Microsoft YaHei UI`, fallback `Segoe UI`

## Geometry

- Spacing scale: `4, 8, 12, 16, 20, 24, 32`
- Control heights: compact `28`, default `34`, primary `38`
- Radius: `4` for fields, `6` for panels, `8` only for major surfaces
- Borders are limited to viewers, inputs and structural separators.

## Components

- **Navigation item:** 3 px active indicator, brighter text and only a subtle
  active surface. No large blue selection fill.
- **Button:** one blue primary action per task cluster; secondary and toolbar
  actions are neutral or icon-only.
- **Field:** graphite input surface, one-pixel border, blue focus ring.
- **Status:** a colored dot plus concise label. Color is semantic, never
  decorative.
- **Metric:** label, large value and optional unit with no boxed KPI treatment.
- **Table:** no cell grid, weak row separators, quiet hover, thin selection.
- **Image viewer:** near-black canvas, compact title row, text view switcher,
  icon-only zoom actions and a low-emphasis metadata strip.
- **Inspector:** elevated background, section rhythm and dividers instead of
  nested group boxes.

## Icon system

The prototype renders a coherent set of inline SVG line icons at 16, 18 and
20 px. Names follow product intent (`image`, `camera`, `scan`, `save`,
`settings`, `sliders`, `database`, `activity`, `zoom-in`, `zoom-out`, `fit`,
`play`, `stop`, `layers`, `mask`). These can later be replaced one-for-one by
the approved Lucide SVG asset set.

## Interaction model

- Template workflow selection changes the active step; view tabs switch the
  mock viewer between Original, Mask and Overlay.
- Live Run/Stop changes production state without opening a device.
- Device Settings opens a right-side settings drawer; operational camera space
  remains clean while the drawer is closed.
- Results table selection represents choosing a detected object and updates the
  selected-object inspection concept.
