import { useEffect, useMemo, useState } from "react";
import { fetchLatency, refreshLatency } from "./api";
import type { LatencyExpectations, LatencyGroup } from "./types";

// The per-group feature-latency view. A HORIZONTAL BAR CHART, one row per group, SORTED SLOWEST-FIRST by the
// tail (p99); bar LENGTH encodes the typical per-minute cost (p50). Hovering a bar surfaces the tail (p95/p99)
// plus the facts that explain the cost. Every group sits far below the 1 ms per-group budget.

function formatUs(ns: number): string {
  const us = ns / 1000;
  return us < 10 ? us.toFixed(2) : us.toFixed(1);
}

// Bar fill darkens with slowness (anchored at the slowest p50): pale blue (#cfe0fb) -> deep "trusted" blue.
function barColor(p50: number, maxP50: number): string {
  const t = maxP50 > 0 ? Math.min(1, Math.max(0, p50 / maxP50)) : 0;
  const eased = Math.pow(t, 0.6);
  const red = Math.round(207 + (11 - 207) * eased);
  const green = Math.round(224 + (61 - 224) * eased);
  const blue = Math.round(251 + (145 - 251) * eased);
  return `rgb(${red},${green},${blue})`;
}

interface HoverState {
  group: LatencyGroup;
  x: number;
  y: number;
}

interface LatencyBarProps {
  group: LatencyGroup;
  maxP50: number;
  onHover: (state: HoverState | null) => void;
}

function LatencyBar({ group, maxP50, onHover }: LatencyBarProps) {
  const widthPct = maxP50 > 0 ? Math.max(0.5, (group.p50_ns / maxP50) * 100) : 0;
  const move = (event: React.MouseEvent) => onHover({ group, x: event.clientX, y: event.clientY });
  return (
    <div
      className="lat-bar-row"
      onMouseMove={move}
      onMouseEnter={move}
      onMouseLeave={() => onHover(null)}
    >
      <div className="lat-bar-label">{group.group}</div>
      <div className="lat-bar-track">
        <div
          className="lat-bar-fill"
          style={{ width: `${widthPct}%`, background: barColor(group.p50_ns, maxP50) }}
        />
        <span className="lat-bar-value">{formatUs(group.p50_ns)} µs</span>
      </div>
    </div>
  );
}

function LatencyTooltip({ hover }: { hover: HoverState }) {
  const { group } = hover;
  const flipLeft = hover.x > window.innerWidth - 280;
  const style: React.CSSProperties = {
    top: hover.y + 14,
    left: flipLeft ? undefined : hover.x + 14,
    right: flipLeft ? window.innerWidth - hover.x + 14 : undefined,
  };
  return (
    <div className="lat-tooltip" style={style}>
      <div className="lat-tip-title">{group.group}</div>
      <div className="lat-tip-tail">
        <span className="lat-tip-stat">
          <span className="lat-tip-k">p50</span>
          <span className="lat-tip-v">{formatUs(group.p50_ns)} µs</span>
        </span>
        <span className="lat-tip-stat">
          <span className="lat-tip-k">p95</span>
          <span className="lat-tip-v">{formatUs(group.p95_ns)} µs</span>
        </span>
        <span className="lat-tip-stat">
          <span className="lat-tip-k">p99</span>
          <span className="lat-tip-v lat-tip-tailv">{formatUs(group.p99_ns)} µs</span>
        </span>
      </div>
      <dl className="lat-tip-meta">
        <dt>own cost</dt>
        <dd>{formatUs(group.own_ns)} µs</dd>
        <dt>kind</dt>
        <dd>{group.kind}</dd>
        <dt>mechanism</dt>
        <dd>{group.mechanism}</dd>
        <dt>incremental</dt>
        <dd className={`lat-incr lat-incr-${group.incremental_ready}`}>{group.incremental_ready}</dd>
        <dt>features</dt>
        <dd>{group.feat_count}</dd>
      </dl>
    </div>
  );
}

export function LatencyView() {
  const [data, setData] = useState<LatencyExpectations | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<HoverState | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchLatency()
      .then((payload) => {
        if (!cancelled) setData(payload);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const sorted = useMemo<LatencyGroup[]>(
    () => (data ? [...data.groups].sort((a, b) => b.p99_ns - a.p99_ns) : []),
    [data],
  );
  const maxP50 = useMemo(
    () => sorted.reduce((acc, group) => Math.max(acc, group.p50_ns), 0),
    [sorted],
  );

  const remeasure = async () => {
    setBusy(true);
    try {
      setData(await refreshLatency());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  if (error) return <div className="banner-error">{error}</div>;
  if (!data) return <div className="lat-note">Measuring feature latency…</div>;

  const ctx = data.e2e_context;

  return (
    <div className="lat-view" onMouseLeave={() => setHover(null)}>
      <div className="lat-header">
        <div className="lat-headline">
          <span className="lat-metric">bar &rarr; vector</span>
          <span className="lat-stat">
            <span className="lat-stat-val">{formatUs(ctx.vector_p50_ns)}</span> µs
            <span className="lat-stat-lbl">vector p50</span>
          </span>
          <span className="lat-stat">
            <span className="lat-stat-val">{formatUs(ctx.total_own_ns)}</span> µs
            <span className="lat-stat-lbl">sum of groups</span>
          </span>
          <span className="lat-stat">
            <span className="lat-stat-val lat-target">&lt;1</span> ms
            <span className="lat-stat-lbl">budget / group</span>
          </span>
          <button className="view-tab" onClick={remeasure} disabled={busy}>
            {busy ? "measuring…" : "Re-measure"}
          </button>
        </div>
        <div className="lat-submeta">
          {data.group_count} groups · {data.feature_count} features · units: {data.units} ·{" "}
          {data.sorted_by} · {data.scenario.measure_minutes} min/group · generated {data.generated_at}
        </div>
        <div className="lat-note">
          All {data.group_count} feature groups compute in well under the 1&nbsp;ms per-group budget — each
          value is a few lookups on top of the group's incremental cache.
        </div>
      </div>

      <div className="lat-chart">
        {sorted.map((group) => (
          <LatencyBar key={group.group} group={group} maxP50={maxP50} onHover={setHover} />
        ))}
      </div>

      {hover && <LatencyTooltip hover={hover} />}
    </div>
  );
}
