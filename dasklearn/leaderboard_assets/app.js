/* Read-only dashboard. Use textContent for every value originating in results. */
"use strict";
const $ = id => document.getElementById(id);
const colors = ["#72e4bf", "#7bb9ff", "#ffc27b", "#df9fff", "#ff8c9a", "#91d6e6", "#d2dc7f", "#b8a1ff"];
let current;
const number = (value, digits = 2) => value == null ? "—" : value.toLocaleString(undefined, {maximumFractionDigits: digits, minimumFractionDigits: digits});
function cell(row, value) { const td = document.createElement("td"); td.textContent = value; row.append(td); }
function svgElement(tag, attributes, content) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (content != null) node.textContent = content;
  return node;
}
function curves(data) {
  const variants = [...new Set(data.runs.map(run => run.variant))];
  return variants.map((variant, i) => {
    const times = new Map();
    for (const run of data.runs.filter(run => run.variant === variant && ["running", "completed"].includes(run.status))) {
      for (const point of run.curve) {
        if (!times.has(point.time)) times.set(point.time, []);
        times.get(point.time).push(point);
      }
    }
    const points = [...times].sort((a,b) => a[0] - b[0]).map(([time, values]) => ({time,
      accuracy: values.reduce((sum,p) => sum + p.accuracy, 0) / values.length,
      bytes_sent: values.reduce((sum,p) => sum + p.bytes_sent, 0) / values.length, seeds: values.length}));
    return {variant, color: i < colors.length ? colors[i] : `hsl(${(i * 137.508) % 360} 72% 72%)`, points};
  });
}
function chart(id, series, field, label) {
  const svg = $(id); svg.replaceChildren();
  const left=56, top=18, width=602, height=260;
  const maxX = Math.max(1, ...series.flatMap(s => s.points.map(p => p[field])));
  const x = value => left + value / maxX * width;
  const y = value => top + height - value / 100 * height;
  for (let i=0; i<=4; i++) {
    const value = i * 25;
    svg.append(svgElement("line", {x1:left, x2:left+width, y1:y(value), y2:y(value), stroke:"#304150"}));
    svg.append(svgElement("text", {x:left-10, y:y(value)+4, fill:"#a9bac8", "font-size":11, "text-anchor":"end"}, `${value}%`));
    svg.append(svgElement("text", {x:x(maxX*i/4), y:height+top+22, fill:"#a9bac8", "font-size":11, "text-anchor":"middle"}, number(maxX*i/4, field === "time" ? 0 : 1)));
  }
  svg.append(svgElement("text", {x:left+width/2, y:324, fill:"#a9bac8", "font-size":12, "text-anchor":"middle"}, label));
  for (const seriesItem of series) {
    const points = seriesItem.points.map(point => `${x(point[field])},${y(point.accuracy)}`).join(" ");
    svg.append(svgElement("polyline", {points, fill:"none", stroke:seriesItem.color, "stroke-width":2}));
    for (const point of seriesItem.points) {
      const circle = svgElement("circle", {cx:x(point[field]), cy:y(point.accuracy), r:3, fill:seriesItem.color});
      circle.append(svgElement("title", {}, `${seriesItem.variant}: ${number(point.accuracy)}%, t=${point.time}s, ${point.seeds} seed(s)`));
      svg.append(circle);
    }
  }
}
function render(data) {
  $("title").textContent = data.name;
  $("status").textContent = data.status;
  $("scenario").textContent = `${data.scenario.dataset} · ${data.scenario.model} · ${data.scenario.peers} clients · ${data.scenario.partitioner} · ${data.network.profile} capacities`;
  $("completed").textContent = `${data.runs.filter(run => run.status === "completed").length} / ${data.runs.length}`;
  $("split").textContent = data.evaluation_split;
  $("updated").textContent = new Date(data.updated_at).toLocaleTimeString();
  const select = $("budget");
  if ([...select.options].map(option => +option.value).join() !== data.budgets.join()) {
    select.replaceChildren(...data.budgets.map(budget => new Option(`${budget} seconds`, budget)));
  }
  const table = data.tables.find(table => table.budget === +select.value);
  $("budget-label").textContent = `${table.budget} s`;
  $("ranking-note").textContent = table.provisional ? "Provisional results · Official ranks appear after every variant has the full paired seed set. Partial or failed runs never receive a final rank." : "Complete paired comparison · Sorted by mean accuracy at the selected simulated budget. Small differences may fall within seed variation.";
  $("rows").replaceChildren(); $("targets").replaceChildren();
  for (const row of table.rows) {
    const tr = document.createElement("tr");
    const accuracy = row.accuracy ? `${number(row.accuracy.mean)} ± ${row.accuracy.sd == null ? "—" : number(row.accuracy.sd)}` : "—";
    const values = [row.rank ?? "—", row.variant, `${row.seeds.length}/${row.expected_seeds}${row.failed_runs ? " · failed" : ""}`, accuracy,
      number(row.average_accuracy?.mean), number(row.worst_decile_accuracy?.mean), number(row.bytes_sent?.mean == null ? null : row.bytes_sent.mean/1e6),
      number(row.best_observed_accuracy?.mean), number(row.execution_wall_seconds?.mean, 1)];
    values.forEach(value => cell(tr, value)); $("rows").append(tr);
    const targetRow = document.createElement("tr"); cell(targetRow, ""); cell(targetRow, row.variant);
    for (const target of data.accuracy_targets) {
      const metric = row.time_to_accuracy[String(target)];
      cell(targetRow, metric.evaluated ? `${metric.mean_seconds == null ? "Not reached" : number(metric.mean_seconds, 1) + " s"} · ${metric.reached}/${metric.evaluated} reached` : "—");
    }
    $("targets").append(targetRow);
  }
  const head = document.createElement("tr");
  for (const label of ["", "Algorithm / variant", ...data.accuracy_targets.map(target => `${target}% accuracy`)]) {
    const th = document.createElement("th"); th.textContent=label; head.append(th);
  }
  $("target-head").replaceChildren(head);
  const series = curves(data);
  chart("time-chart", series, "time", "Simulated protocol time (seconds)");
  chart("traffic-chart", series.map(s => ({...s, points:s.points.map(p => ({...p, bytes_sent:p.bytes_sent/1e6}))})), "bytes_sent", "Total sent traffic (MB)");
  $("legend").replaceChildren(...series.map(s => {
    const label=document.createElement("span"), dot=document.createElement("i"); dot.className="dot"; dot.style.backgroundColor=s.color;
    label.append(dot, document.createTextNode(s.variant)); return label;
  }));
  $("runs").replaceChildren(...data.runs.map(run => {
    const box=document.createElement("div"); box.className=`run${run.error ? " error" : ""}`;
    const name=document.createElement("strong"), status=document.createElement("div"); name.textContent=run.name;
    status.textContent=`${run.status} · ${run.curve.at(-1)?.time ?? 0}s evaluated${run.wall_seconds == null ? "" : " · " + number(run.wall_seconds,1) + "s wall"}`;
    box.append(name,status); if (run.error) {const error=document.createElement("div");error.textContent=run.error;box.append(error);} return box;
  }));
  $("interpretation").textContent = data.evaluation_split === "test" ? "Exploratory test-set comparison. Best observed accuracy is a descriptive maximum over checkpoints; it is not an independently evaluated, validation-selected checkpoint or an estimate of unlimited-time performance." : "Best observed accuracy is measured on validation checkpoints within the declared budget. Final held-out test evaluation and automatic hyperparameter tuning are separate steps.";
  $("provenance").textContent=JSON.stringify({scenario:data.scenario,network:data.network,provenance:data.provenance,runs:data.runs.map(({curve,...run}) => run)}, null, 2);
}
async function refresh() {
  try {
    const response = await fetch("/leaderboard.json", {cache:"no-store"});
    if (!response.ok) throw new Error(`Waiting for results (HTTP ${response.status})`);
    current = await response.json(); render(current); $("connection").textContent="Refreshes every 2 seconds. Results remain available after the benchmark finishes.";
  } catch (error) {$("connection").textContent=error.message;}
  setTimeout(refresh, 2000);
}
$("budget").addEventListener("change", () => {if (current) render(current);});
refresh();
