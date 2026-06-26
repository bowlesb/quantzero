import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchMatrix } from "./api";
import type { StoreGridMatrix } from "./types";
import { CanvasHeatmap, type DisplayColumn, type HoverCell } from "./CanvasHeatmap";
import { Tooltip } from "./Tooltip";
import { LatencyView } from "./LatencyView";

// Two views: the feature-store coverage grid (default) and the per-feature latency chart.
type View = "grid" | "latency";

const VIEW_TITLES: Record<View, string> = {
  grid: "Feature-store coverage",
  latency: "Feature latency expectations",
};

function formatAsOf(generatedAt: string): string {
  const then = new Date(generatedAt);
  const hhmm = then.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return `as of ${hhmm}`;
}

export function App() {
  const [matrix, setMatrix] = useState<StoreGridMatrix | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<HoverCell | null>(null);
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());
  const [view, setView] = useState<View>("grid");

  useEffect(() => {
    fetchMatrix()
      .then(setMatrix)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  const toggleExpand = useCallback((groupKey: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(groupKey)) next.delete(groupKey);
      else next.add(groupKey);
      return next;
    });
  }, []);

  // Display columns mirror the heatmap's expansion (group column + its feature sub-columns when expanded).
  const displayCols = useMemo<DisplayColumn[]>(() => {
    if (!matrix) return [];
    const out: DisplayColumn[] = [];
    matrix.columns.forEach((col, idx) => {
      if (col.kind === "group") {
        const expanded = expandedGroups.has(col.key);
        out.push({
          kind: "group",
          key: col.key,
          label: col.label,
          trusted: col.trusted,
          coverageCol: idx,
          groupKey: col.key,
          expandable: col.features.length > 0,
          expanded,
        });
        if (expanded) {
          col.features.forEach((feature) =>
            out.push({
              kind: "feature",
              key: `${col.key}::${feature}`,
              label: feature,
              trusted: col.trusted,
              coverageCol: idx,
              groupKey: col.key,
              expandable: false,
              expanded: false,
            }),
          );
        }
      } else {
        out.push({
          kind: "raw",
          key: col.key,
          label: col.label,
          trusted: false,
          coverageCol: idx,
          groupKey: null,
          expandable: false,
          expanded: false,
        });
      }
    });
    return out;
  }, [matrix, expandedGroups]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-left">
          <span className="brand-mark" />
          <h1>{VIEW_TITLES[view]}</h1>
        </div>
        <div className="topbar-center">
          {view === "grid" && matrix && (
            <span className="asof">{formatAsOf(matrix.generated_at)}</span>
          )}
        </div>
        <nav className="view-tabs">
          <button
            className={`view-tab${view === "grid" ? " active" : ""}`}
            onClick={() => setView("grid")}
          >
            Coverage grid
          </button>
          <button
            className={`view-tab${view === "latency" ? " active" : ""}`}
            onClick={() => setView("latency")}
          >
            Latency
          </button>
        </nav>
      </header>

      {view === "latency" ? (
        <LatencyView />
      ) : (
        <>
          <div className="controls">
            {matrix && (
              <div className="legend">
                <span className="legend-item">
                  <span className="sw sw-empty" /> none
                </span>
                <span className="legend-item">
                  <span className="ramp ramp-trusted" /> feature group
                </span>
                <span className="legend-item">
                  <span className="ramp ramp-raw" /> raw tape layer
                </span>
                <span className="legend-item">
                  click a feature-group header to expand its columns
                </span>
                <span className="legend-uni">
                  darkness = % of the {matrix.universe_size.toLocaleString()}-ticker universe covered
                </span>
              </div>
            )}
          </div>

          {error && <div className="banner-error">{error}</div>}

          <div className="grid-region">
            {matrix && (
              <CanvasHeatmap
                matrix={matrix}
                expandedGroups={expandedGroups}
                highlightCol={null}
                onHoverChange={setHover}
                onToggleExpand={toggleExpand}
                onOpenDetail={() => {}}
              />
            )}
          </div>

          {matrix && <Tooltip hover={hover} matrix={matrix} displayCols={displayCols} />}
        </>
      )}
    </div>
  );
}
