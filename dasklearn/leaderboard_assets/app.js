/* Read-only dashboard. Use textContent for every value originating in results. */
"use strict";
const $ = id => document.getElementById(id);
const colors = ["#72e4bf", "#7bb9ff", "#ffc27b", "#df9fff", "#ff8c9a", "#91d6e6", "#d2dc7f", "#b8a1ff"];
let current;
let selectedSetting, renderedSetting, refreshTimer, refreshGeneration = 0, budgetChosen = false;
let selectedWorkload = new URLSearchParams(window.location.search).get("workload");
const workloadSuffix = () => selectedWorkload ? `&workload=${encodeURIComponent(selectedWorkload)}` : "";
const number = (value, digits = 2) => value == null ? "—" : value.toLocaleString(undefined, {maximumFractionDigits: digits, minimumFractionDigits: digits});
function cell(row, value) { const td = document.createElement("td"); td.textContent = value; row.append(td); }
function algorithmCell(row, variant, data) {
  const entry = data.algorithms?.find(entry => entry.id === variant || entry.variants.includes(variant));
  if (!entry) {cell(row, variant); return;}
  const td=document.createElement('td'), a=document.createElement('a');
  a.textContent=entry.label;
  a.href=`/algorithm/${encodeURIComponent(entry.id)}?setting=${encodeURIComponent(selectedSetting ?? 'default')}${workloadSuffix()}`;
  td.append(a);row.append(td);
}
function svgElement(tag, attributes, content) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (content != null) node.textContent = content;
  return node;
}
function curves(data, metric = "accuracy", budget = Infinity) {
  const variants = [...new Set(data.runs.map(run => run.variant))];
  return variants.map((variant, i) => {
    const times = new Map();
    for (const run of data.runs.filter(run => run.variant === variant && ["running", "completed"].includes(run.status))) {
      for (const point of run.curve) {
        if (point.time > budget || !Number.isFinite(point[metric])) continue;
        if (!times.has(point.time)) times.set(point.time, []);
        times.get(point.time).push(point);
      }
    }
    const points = [...times].sort((a,b) => a[0] - b[0]).map(([time, values]) => ({time,
      [metric]: values.reduce((sum,p) => sum + p[metric], 0) / values.length,
      bytes_sent: values.reduce((sum,p) => sum + p.bytes_sent, 0) / values.length, seeds: values.length}));
    return {variant, color: i < colors.length ? colors[i] : `hsl(${(i * 137.508) % 360} 72% 72%)`, points};
  });
}
function chart(id, series, field, label, metric = "accuracy", reference = null, budget = null) {
  const svg = $(id); svg.replaceChildren();
  const left=56, top=18, width=602, height=260;
  const maxX = field === "time" && budget != null ? budget : Math.max(1, ...series.flatMap(s => s.points.map(p => p[field])));
  const referenceValue = reference?.status === "completed" && Number.isFinite(reference[metric]?.mean) ? reference[metric].mean : null;
  const maxY = metric === "accuracy" ? 100 : Math.max(0.001, referenceValue ?? 0, ...series.flatMap(s => s.points.map(p => p[metric]))) * 1.08;
  const x = value => left + value / maxX * width;
  const y = value => top + height - value / maxY * height;
  for (let i=0; i<=4; i++) {
    const value = i * maxY / 4;
    svg.append(svgElement("line", {x1:left, x2:left+width, y1:y(value), y2:y(value), stroke:"#304150"}));
    svg.append(svgElement("text", {x:left-10, y:y(value)+4, fill:"#a9bac8", "font-size":11, "text-anchor":"end"}, metric === "accuracy" ? `${value}%` : number(value, maxY < 0.01 ? 4 : 2)));
    svg.append(svgElement("text", {x:x(maxX*i/4), y:height+top+22, fill:"#a9bac8", "font-size":11, "text-anchor":"middle"}, number(maxX*i/4, field === "time" ? 0 : 1)));
  }
  svg.append(svgElement("text", {x:left+width/2, y:324, fill:"#a9bac8", "font-size":12, "text-anchor":"middle"}, label));
  for (const seriesItem of series) {
    const points = seriesItem.points.map(point => `${x(point[field])},${y(point[metric])}`).join(" ");
    svg.append(svgElement("polyline", {points, fill:"none", stroke:seriesItem.color, "stroke-width":2}));
    for (const point of seriesItem.points) {
      const circle = svgElement("circle", {cx:x(point[field]), cy:y(point[metric]), r:3, fill:seriesItem.color});
      const value = metric === "accuracy" ? `${number(point[metric])}%` : `loss ${number(point[metric], 3)}`;
      circle.append(svgElement("title", {}, `${seriesItem.variant}: ${value}, t=${point.time}s, ${point.seeds} seed(s)`));
      svg.append(circle);
    }
  }
  if (referenceValue !== null) {
    const value = metric === "accuracy" ? `${number(referenceValue)}%` : `loss ${number(referenceValue, 3)}`;
    const line = svgElement("line", {x1:left, x2:left+width, y1:y(referenceValue), y2:y(referenceValue),
      stroke:"#fff0ae", "stroke-width":2, "stroke-dasharray":"8 5", "data-reference":metric});
    line.append(svgElement("title", {}, `Centralized: ${value} · ${number(reference.steps, 0)} SGD steps · ${reference.checkpoint_rule ?? "Validation-selected checkpoints"} · ${reference.seeds.length} seeds`));
    svg.append(line);
    svg.append(svgElement("text", {x:left+width-4, y:Math.max(top+12, y(referenceValue)-8),
      fill:"#fff0ae", "font-size":12, "text-anchor":"end"}, `Centralized ${value}`));
  }
}
function render(data) {
  const globalModel = data.evaluation_model === "global";
  const reference = data.reference;
  $("title").textContent = data.name;
  $("campaign-note").hidden = !data.campaign_notice;
  $("campaign-note").textContent = data.campaign_notice ?? '';
  $("status").textContent = data.pipeline ? `${data.pipeline.phase} · ${data.status}` : data.status;
  $("extension-note").hidden = !data.extension;
  if (data.extension) {
    const extension = data.extension;
    const variants = extension.added_variants ?? Object.keys(extension.tuning.learning_rates);
    const names = variants.join(", ");
    const tuning = `Adding ${variants.length} algorithms (${names}) · ${extension.status} · equal-budget validation LR search: ${extension.tuning_completed}/${extension.tuning_total} trials finished${extension.reused_tuning_trials ? " (" + extension.reused_tuning_trials + " reused)" : ""}${extension.diverged_tuning_trials ? " · " + extension.diverged_tuning_trials + " divergent candidates excluded" : ""}. New runs wait for LR selection. Select the corresponding LR tuning setting to inspect candidate curves.`;
    const evaluation = `${variants.length} additional algorithms (${names}) · ${extension.status} · ${extension.phase === "waiting_for_baselines" ? "waiting for existing experiments to finish" : "frozen validation-selected LRs"}${extension.current_setting ? " · executing setting: " + extension.current_setting : ""}. Algorithm-specific settings are held fixed throughout tuning and final evaluation.`;
    $("extension-note").textContent = (extension.phase === "tuning" ? tuning : evaluation) + (extension.following_baselines ? " Existing experiment results are refreshed independently." : "") + (extension.error ? " " + extension.error : "");
  }
  $("tuning-section").hidden = !data.pipeline;
  if (data.pipeline) {
    const spec=data.pipeline.tuning, selected=data.pipeline.selected;
    const adaptive=spec.method === 'tpe';
    $("tuning-title").textContent = selected ? "Frozen parameters · held-out evaluation" : adaptive ? "TPE parameter search" : "Equal-budget learning-rate search";
    $("tuning-rule").textContent = `${spec.trials_per_algorithm} trials per algorithm · ${spec.budget}s per trial · validation only · ${spec.evaluation_examples || "all"} examples per checkpoint. ${spec.metric === 'loss' ? 'Minimize' : 'Maximize'} mean ${spec.metric}; lower LR breaks exact ties. Final seeds: ${data.pipeline.final_seeds.join(", ")}.${data.pipeline.error ? " " + data.pipeline.error : ""}`;
    $("tuning-rows").replaceChildren(...Object.entries(adaptive ? spec.search_spaces : spec.learning_rates).map(([variant,space]) => {
      const row=document.createElement("tr"); cell(row,variant); cell(row,adaptive ? JSON.stringify(space) : space.join(", ")); cell(row,adaptive && data.selection?.selected_parameters?.[variant] ? JSON.stringify(data.selection.selected_parameters[variant]) : selected?.[variant] ?? "Pending"); return row;
    }));
  }
  const partition = data.scenario.partitioner === "dirichlet" ? `Dirichlet α = ${data.scenario.alpha}` : data.scenario.partitioner;
  $("scenario").textContent = `${data.scenario.dataset} · ${data.scenario.model} · ${data.scenario.peers} clients · ${partition} · ${data.network.profile} capacities · ${globalModel ? "global model" : "mean client accuracy"}`;
  $("worst-client-header").hidden = globalModel;
  $("metric-note").textContent = globalModel ? "Each checkpoint averages all client parameters with equal weights, then evaluates that global model once. Aggregation is an offline diagnostic: collection time and traffic are excluded from protocol costs. Average accuracy integrates the global learning curve. Full-run wall time includes evaluation and any concurrent worker contention." : "Accuracy averages individual client accuracies. Worst 10% averages the lowest-scoring clients. Average accuracy integrates the learning curve. Traffic counts sent bytes, including partial transfers. Full-run wall time includes evaluation and any concurrent worker contention.";
  $("completed").textContent = `${data.runs.filter(run => ["completed", "diverged"].includes(run.status)).length} / ${data.runs.length}`;
  $("split").textContent = data.evaluation_split;
  $("updated").textContent = new Date(data.updated_at).toLocaleTimeString();
  const select = $("budget");
  const primaryTime = data.evaluation?.primary_time ?? data.selection?.selected_time;
  const settingKey = `${selectedWorkload ?? ''}/${selectedSetting ?? data.name}`;
  if (renderedSetting !== settingKey || [...select.options].map(option => +option.value).join() !== data.budgets.join()) {
    const previous = renderedSetting === settingKey && budgetChosen ? +select.value : null;
    select.replaceChildren(...data.budgets.map(budget => new Option(`${budget} seconds`, budget)));
    select.value = previous !== null && data.budgets.includes(previous) ? previous : data.budgets.includes(primaryTime) ? primaryTime : data.budgets.includes(900) ? 900 : Math.max(...data.budgets);
  }
  if (!budgetChosen && data.budgets.includes(primaryTime)) select.value = primaryTime;
  renderedSetting = settingKey;
  $("export").href = `/leaderboard.csv?setting=${encodeURIComponent(selectedSetting ?? "default")}${workloadSuffix()}`;
  const table = data.tables.find(table => table.budget === +select.value);
  $("budget-label").textContent = `${table.budget} s`;
  $("ranking-note").textContent = table.provisional ? "Provisional results · Official ranks appear after every variant has the full paired seed set. Partial or failed runs never receive a final rank." : "Complete paired comparison · Sorted by mean accuracy at the selected simulated budget. Small differences may fall within seed variation.";
  if (data.evaluation?.policy === "fixed_endpoint") $("ranking-note").textContent += ` Primary evaluation uses the fixed endpoint at ${primaryTime} s. Hyperparameters come from the earlier validation search; other budget views describe the learning curve.`;
  else if (data.selection?.selected_time != null) $("ranking-note").textContent += ` Primary evaluation uses the validation-selected checkpoint at ${data.selection.selected_time} s; other budget views describe the learning curve.`;
  if (data.pipeline?.phase === "tuning") $("ranking-note").textContent = "Validation search trials · These candidate scores select learning rates. The algorithm comparison starts after selection, using fresh seeds and test data.";
  $("rows").replaceChildren(); $("targets").replaceChildren();
  for (const row of table.rows) {
    const tr = document.createElement("tr");
    const accuracy = row.accuracy ? `${number(row.accuracy.mean)} ± ${row.accuracy.sd == null ? "—" : number(row.accuracy.sd)}` : "—";
    const loss = row.loss ? `${number(row.loss.mean, 3)} ± ${row.loss.sd == null ? "—" : number(row.loss.sd, 3)}` : "—";
    const values = [row.rank ?? "—", row.variant, `${row.seeds.length}/${row.expected_seeds}${row.failed_runs ? " · failed" : ""}`, accuracy,
      loss, number(row.average_accuracy?.mean), ...(globalModel ? [] : [number(row.worst_decile_accuracy?.mean)]), number(row.bytes_sent?.mean == null ? null : row.bytes_sent.mean/1e6),
      number(row.best_observed_accuracy?.mean), number(row.execution_wall_seconds?.mean, 1)];
    values.forEach((value,index) => index === 1 ? algorithmCell(tr, row.variant, data) : cell(tr, value)); $("rows").append(tr);
    const targetRow = document.createElement("tr"); cell(targetRow, ""); algorithmCell(targetRow, row.variant, data);
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
  const series = curves(data, "accuracy", table.budget);
  chart("time-chart", series, "time", "Simulated protocol time (seconds)", "accuracy", reference, table.budget);
  chart("traffic-chart", series.map(s => ({...s, points:s.points.map(p => ({...p, bytes_sent:p.bytes_sent/1e6}))})), "bytes_sent", "Total sent traffic (MB)", "accuracy", reference);
  const lossSeries = curves(data, "loss", table.budget);
  chart("loss-time-chart", lossSeries, "time", "Simulated protocol time (seconds)", "loss", reference, table.budget);
  chart("loss-traffic-chart", lossSeries.map(s => ({...s, points:s.points.map(p => ({...p, bytes_sent:p.bytes_sent/1e6}))})), "bytes_sent", "Total sent traffic (MB)", "loss", reference);
  $("loss-note").textContent = `Mean ${data.evaluation_split} loss per example of ${globalModel ? "the aggregated global model" : "individual client models"}, averaged across available seeds at each checkpoint. Lower is better. Loss uses its own vertical scale; colors match the accuracy charts. Hover over a point for its loss and seed count.`;
  if (!lossSeries.some(s => s.points.length)) $("loss-note").textContent = "Waiting for checkpoints with evaluation loss. Missing loss values are omitted.";
  $("legend").replaceChildren(...series.map(s => {
    const label=document.createElement("span"), dot=document.createElement("i"); dot.className="dot"; dot.style.backgroundColor=s.color;
    label.append(dot, document.createTextNode(s.variant)); return label;
  }));
  if (reference?.status === "completed") {
    const label=document.createElement("span"), dash=document.createElement("i");
    dash.className="reference-swatch";
    label.append(dash, document.createTextNode(`Centralized · ${number(reference.steps, 0)} steps`)); $("legend").append(label);
  }
  $("interpretation").textContent = data.evaluation_split === "test" ? "Exploratory test-set comparison. Best observed accuracy is a descriptive maximum over checkpoints; it is not an independently evaluated, validation-selected checkpoint or an estimate of unlimited-time performance." : "Best observed accuracy is measured on validation checkpoints within the declared budget. Final held-out test evaluation and automatic hyperparameter tuning are separate steps.";
  if (data.pipeline) $("interpretation").textContent = data.pipeline.phase === "tuning" ? "Each algorithm receives the same number of candidates, tuning seeds, simulated seconds, and validation examples. Candidate ranges are declared in advance; no test scores enter LR selection." : "Learning rates were frozen using validation data on separate seeds. Test accuracy at the declared budgets is the held-out comparison. Best observed test accuracy is descriptive and must not be used to select checkpoints or retune settings.";
  $("provenance").textContent=JSON.stringify({scenario:data.scenario,network:data.network,provenance:data.provenance,reference:data.reference,runs:data.runs.map(({curve,...run}) => run)}, null, 2);
}
async function refresh() {
  const generation = ++refreshGeneration;
  clearTimeout(refreshTimer);
  try {
    const settingsResponse = await fetch(`/settings.json${selectedWorkload ? '?workload='+encodeURIComponent(selectedWorkload) : ''}`, {cache:"no-store"});
    if (!settingsResponse.ok) throw new Error(`Cannot load settings (HTTP ${settingsResponse.status})`);
    const catalog = await settingsResponse.json();
    if (generation !== refreshGeneration) return;
    selectedWorkload = catalog.workload ?? selectedWorkload;
    const workloads = catalog.workloads ?? [];
    const workloadPicker = $("workload");
    if ([...workloadPicker.options].map(option=>option.value).join() !== workloads.map(item=>item.id).join()) {
      workloadPicker.replaceChildren(...workloads.map(item=>new Option(item.label,item.id)));
    }
    workloadPicker.value = selectedWorkload ?? '';
    $("workload-picker").hidden = workloads.length < 2;
    if (selectedSetting === undefined) selectedSetting = new URLSearchParams(window.location.search).get("setting") ?? catalog.default;
    if (!catalog.settings.some(setting => setting.id === selectedSetting)) selectedSetting = catalog.default;
    const picker = $("setting");
    if ([...picker.options].map(option => option.value).join() !== catalog.settings.map(setting => setting.id).join()) {
      picker.replaceChildren(...catalog.settings.map(setting => new Option(setting.label, setting.id)));
    }
    picker.value = selectedSetting;
    $("setting-picker").hidden = catalog.settings.length < 2;
    const response = await fetch(`/leaderboard.json?setting=${encodeURIComponent(selectedSetting)}${workloadSuffix()}`, {cache:"no-store"});
    if (!response.ok) throw new Error(`Waiting for results (HTTP ${response.status})`);
    const result = await response.json();
    if (generation !== refreshGeneration) return;
    current = result; render(current); $("connection").textContent="Refreshes every 2 seconds. Remote experiments continue while this page is closed.";
  } catch (error) {if (generation === refreshGeneration) $("connection").textContent=error.message;}
  finally {if (generation === refreshGeneration) refreshTimer = setTimeout(refresh, 2000);}
}
$("budget").addEventListener("change", () => {budgetChosen = true;if (current) render(current);});
function changeSelection() {
  current = null; budgetChosen = false;
  const query = new URLSearchParams(window.location.search);
  if (selectedSetting === undefined)query.delete("setting");else query.set("setting", selectedSetting);
  if (selectedWorkload)query.set("workload", selectedWorkload);
  window.history.replaceState(null, "", `${window.location.pathname}?${query}${window.location.hash}`);
  for (const id of ["rows", "targets", "legend", "time-chart", "traffic-chart", "loss-time-chart", "loss-traffic-chart"]) $(id).replaceChildren();
  $("loss-note").textContent = "";
  for (const id of ["completed", "budget-label", "split", "updated"]) $(id).textContent = "—";
  $("status").textContent = "Loading"; $("title").textContent = "Loading selected setting…"; $("scenario").textContent = "";
  refresh();
}
$("setting").addEventListener("change", () => {selectedSetting = $("setting").value;changeSelection();});
$("workload").addEventListener("change", () => {selectedWorkload = $("workload").value;selectedSetting = undefined;changeSelection();});
refresh();
