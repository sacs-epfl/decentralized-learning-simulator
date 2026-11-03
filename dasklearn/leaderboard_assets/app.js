/* Read-only dashboard. Use textContent for every value originating in results. */
"use strict";
const $ = id => document.getElementById(id);
const colors = ["#72e4bf", "#7bb9ff", "#ffc27b", "#df9fff", "#ff8c9a", "#91d6e6", "#d2dc7f", "#b8a1ff"];
let current;
let selectedSetting, renderedSetting, refreshTimer, refreshGeneration = 0;
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
  const globalModel = data.evaluation_model === "global";
  $("title").textContent = data.name;
  $("status").textContent = data.pipeline ? `${data.pipeline.phase} · ${data.status}` : data.status;
  $("tuning-section").hidden = !data.pipeline;
  if (data.pipeline) {
    const spec=data.pipeline.tuning, selected=data.pipeline.selected;
    $("tuning-title").textContent = selected ? "Frozen learning rates · held-out evaluation" : "Equal-budget learning-rate search";
    $("tuning-rule").textContent = `${spec.trials_per_algorithm} trials per algorithm · ${spec.budget}s per trial · validation only · ${spec.evaluation_examples || "all"} examples per checkpoint. Maximize mean ${spec.metric}; lower LR breaks exact ties. Final seeds: ${data.pipeline.final_seeds.join(", ")}.${data.pipeline.error ? " " + data.pipeline.error : ""}`;
    $("tuning-rows").replaceChildren(...Object.entries(spec.learning_rates).map(([variant,rates]) => {
      const row=document.createElement("tr"); cell(row,variant); cell(row,rates.join(", ")); cell(row,selected?.[variant] ?? "Pending"); return row;
    }));
  }
  const partition = data.scenario.partitioner === "dirichlet" ? `Dirichlet α = ${data.scenario.alpha}` : data.scenario.partitioner;
  $("scenario").textContent = `${data.scenario.dataset} · ${data.scenario.model} · ${data.scenario.peers} clients · ${partition} · ${data.network.profile} capacities · ${globalModel ? "global model" : "mean client accuracy"}`;
  $("worst-client-header").hidden = globalModel;
  $("metric-note").textContent = globalModel ? "Each checkpoint averages all client parameters with equal weights, then evaluates that global model once. Aggregation is an offline diagnostic: collection time and traffic are excluded from protocol costs. Average accuracy integrates the global learning curve. Full-run wall time includes evaluation and any concurrent worker contention." : "Accuracy averages individual client accuracies. Worst 10% averages the lowest-scoring clients. Average accuracy integrates the learning curve. Traffic counts sent bytes, including partial transfers. Full-run wall time includes evaluation and any concurrent worker contention.";
  $("completed").textContent = `${data.runs.filter(run => run.status === "completed").length} / ${data.runs.length}`;
  $("split").textContent = data.evaluation_split;
  $("updated").textContent = new Date(data.updated_at).toLocaleTimeString();
  const select = $("budget");
  const settingKey = selectedSetting ?? data.name;
  if (renderedSetting !== settingKey || [...select.options].map(option => +option.value).join() !== data.budgets.join()) {
    const previous = renderedSetting === settingKey ? +select.value : null;
    select.replaceChildren(...data.budgets.map(budget => new Option(`${budget} seconds`, budget)));
    select.value = previous !== null && data.budgets.includes(previous) ? previous : data.budgets.includes(900) ? 900 : Math.max(...data.budgets);
  }
  renderedSetting = settingKey;
  $("export").href = `/leaderboard.csv?setting=${encodeURIComponent(selectedSetting ?? "default")}`;
  const table = data.tables.find(table => table.budget === +select.value);
  $("budget-label").textContent = `${table.budget} s`;
  $("ranking-note").textContent = table.provisional ? "Provisional results · Official ranks appear after every variant has the full paired seed set. Partial or failed runs never receive a final rank." : "Complete paired comparison · Sorted by mean accuracy at the selected simulated budget. Small differences may fall within seed variation.";
  if (data.pipeline?.phase === "tuning") $("ranking-note").textContent = "Validation search trials · These candidate scores select learning rates. The five-algorithm comparison starts after selection, using fresh seeds and test data.";
  $("rows").replaceChildren(); $("targets").replaceChildren();
  for (const row of table.rows) {
    const tr = document.createElement("tr");
    const accuracy = row.accuracy ? `${number(row.accuracy.mean)} ± ${row.accuracy.sd == null ? "—" : number(row.accuracy.sd)}` : "—";
    const values = [row.rank ?? "—", row.variant, row.learning_rate ?? "—", `${row.seeds.length}/${row.expected_seeds}${row.failed_runs ? " · failed" : ""}`, accuracy,
      number(row.average_accuracy?.mean), ...(globalModel ? [] : [number(row.worst_decile_accuracy?.mean)]), number(row.bytes_sent?.mean == null ? null : row.bytes_sent.mean/1e6),
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
  if (data.pipeline) $("interpretation").textContent = data.pipeline.phase === "tuning" ? "Each algorithm receives the same number of candidates, tuning seeds, simulated seconds, and validation examples. Candidate ranges are declared in advance; no test scores enter LR selection." : "Learning rates were frozen using validation data on separate seeds. Test accuracy at the declared budgets is the held-out comparison. Best observed test accuracy is descriptive and must not be used to select checkpoints or retune settings.";
  $("provenance").textContent=JSON.stringify({scenario:data.scenario,network:data.network,provenance:data.provenance,runs:data.runs.map(({curve,...run}) => run)}, null, 2);
}
async function refresh() {
  const generation = ++refreshGeneration;
  clearTimeout(refreshTimer);
  try {
    const settingsResponse = await fetch("/settings.json", {cache:"no-store"});
    if (!settingsResponse.ok) throw new Error(`Cannot load settings (HTTP ${settingsResponse.status})`);
    const catalog = await settingsResponse.json();
    if (generation !== refreshGeneration) return;
    if (selectedSetting === undefined) selectedSetting = new URLSearchParams(window.location.search).get("setting") ?? catalog.default;
    if (!catalog.settings.some(setting => setting.id === selectedSetting)) selectedSetting = catalog.default;
    const picker = $("setting");
    if ([...picker.options].map(option => option.value).join() !== catalog.settings.map(setting => setting.id).join()) {
      picker.replaceChildren(...catalog.settings.map(setting => new Option(setting.label, setting.id)));
    }
    picker.value = selectedSetting;
    $("setting-picker").hidden = catalog.settings.length < 2;
    const response = await fetch(`/leaderboard.json?setting=${encodeURIComponent(selectedSetting)}`, {cache:"no-store"});
    if (!response.ok) throw new Error(`Waiting for results (HTTP ${response.status})`);
    const result = await response.json();
    if (generation !== refreshGeneration) return;
    current = result; render(current); $("connection").textContent="Refreshes every 2 seconds. Remote experiments continue while this page is closed.";
  } catch (error) {if (generation === refreshGeneration) $("connection").textContent=error.message;}
  finally {if (generation === refreshGeneration) refreshTimer = setTimeout(refresh, 2000);}
}
$("budget").addEventListener("change", () => {if (current) render(current);});
$("setting").addEventListener("change", () => {
  selectedSetting = $("setting").value; current = null;
  const query = new URLSearchParams(window.location.search); query.set("setting", selectedSetting);
  window.history.replaceState(null, "", `${window.location.pathname}?${query}${window.location.hash}`);
  for (const id of ["rows", "targets", "runs", "legend", "time-chart", "traffic-chart"]) $(id).replaceChildren();
  for (const id of ["completed", "budget-label", "split", "updated"]) $(id).textContent = "—";
  $("status").textContent = "Loading"; $("title").textContent = "Loading selected setting…"; $("scenario").textContent = "";
  refresh();
});
refresh();
