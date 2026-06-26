// Visual system for the coverage grid (light theme). WHITE = zero coverage = the page
// background, so an absent cell and a zero-coverage cell look identical. Coverage DARKENS
// each cell toward its column's colour: a feature group -> BLUE, a raw tape layer -> SLATE.
// There is no "trust" distinction — parity is by design (live and backfill run the same
// code), so a feature is a feature; the colour only tells raw-substrate from feature group.

export const COLORS = {
  bg: "#ffffff", // the page / grid background == a zero-coverage cell
  panel: "#f6f8fa",
  panelAlt: "#eef1f5",
  border: "#d6dbe1",
  borderSoft: "#e7ebf0",
  text: "#1b2430",
  textDim: "#4a5563",
  muted: "#7a8694",
  groupDark: "#0b3d91", // feature group, fully covered
  rawDark: "#3a4554", // raw tape layer, fully covered (neutral slate)
  accent: "#1f6feb",
} as const;

// Canvas cell sizing (CSS px before devicePixelRatio scaling). A hairline gap reads as
// crisp tiles; the date axis scrolls vertically.
export const CELL = {
  w: 19,
  h: 9,
  gap: 1,
} as const;

export type ColumnKind = "raw" | "group";

type Rgb = [number, number, number];

const WHITE: Rgb = [255, 255, 255];

function hexToRgb(hex: string): Rgb {
  const value = parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

const GROUP_RGB = hexToRgb(COLORS.groupDark);
const RAW_RGB = hexToRgb(COLORS.rawDark);

function mix(a: Rgb, b: Rgb, t: number): string {
  const r = Math.round(a[0] + (b[0] - a[0]) * t);
  const g = Math.round(a[1] + (b[1] - a[1]) * t);
  const bl = Math.round(a[2] + (b[2] - a[2]) * t);
  return `rgb(${r},${g},${bl})`;
}

// The dark end for a column: a feature group -> blue, a raw layer -> slate.
export function columnDark(kind: ColumnKind): Rgb {
  return kind === "raw" ? RAW_RGB : GROUP_RGB;
}

// The fill a cell paints: WHITE at zero coverage -> the column's dark colour at full.
// Returns null at byte 0 so an absent cell paints nothing. A gentle gamma lift keeps thin
// coverage visible against white.
export function cellColor(coverageByte: number, kind: ColumnKind): string | null {
  if (coverageByte <= 0) return null;
  const t = Math.pow(coverageByte / 255, 0.7);
  return mix(WHITE, columnDark(kind), t);
}

// Sample CSS for the legend swatches (full-coverage dark end of each kind).
export const LEGEND = {
  groupDark: `rgb(${GROUP_RGB.join(",")})`,
  rawDark: `rgb(${RAW_RGB.join(",")})`,
} as const;
