"use strict";
const $ = id => document.getElementById(id);
const identifier = window.location.pathname.split("/").pop();
let selectedSetting = new URLSearchParams(window.location.search).get("setting");
const workload = new URLSearchParams(window.location.search).get("workload");
const workloadSuffix = workload ? `&workload=${encodeURIComponent(workload)}` : "";
let generation = 0, timer;
const format = (value, digits = 2) => value == null ? "—" : Number(value).toLocaleString(undefined, {maximumFractionDigits:digits, minimumFractionDigits:digits});
function duration(seconds) {
  if(seconds == null || !Number.isFinite(seconds) || seconds < 0)return "—";
  if(seconds < 60)return `${format(seconds,1)}s`;
  const total=Math.floor(seconds),hours=Math.floor(total/3600),minutes=Math.floor(total%3600/60),remaining=total%60;
  return `${hours ? hours+'h ' : ''}${minutes}m ${remaining}s`;
}
function cell(row, value) {const td=document.createElement("td");td.textContent=value ?? "—";row.append(td);return td;}
function learningRateCell(row, value) {
  const td=cell(row,value == null ? "—" : Number(value).toLocaleString(undefined,{maximumSignificantDigits:4,useGrouping:false}));
  if(value != null)td.title=String(value);
  return td;
}
function statusCell(row, status, selected = false) {
  const td=cell(row, "");
  const dot=document.createElement("span");
  const color=status === "completed" ? "green" : ["failed","interrupted","diverged"].includes(status) ? "red" : "orange";
  const label=selected ? `Selected (${status})` : status;
  dot.className=`status-dot status-dot-${color}`;
  dot.setAttribute("role","img");dot.setAttribute("aria-label",label);dot.title=label;
  td.append(dot);return td;
}
function textRows(id, rows, keys) {
  $(id).replaceChildren(...rows.map(row => {const tr=document.createElement("tr");for(const key of keys){if(key === "status")statusCell(tr,row[key]);else if(key === "learning_rate")learningRateCell(tr,row[key]);else if(key === "wall_seconds")cell(tr,duration(row[key]));else cell(tr,row[key]);}return tr;}));
}
function link(label, url, external = true) {
  const a=document.createElement("a");a.className="button";a.textContent=label;a.href=url;
  if(external){a.target="_blank";a.rel="noopener noreferrer";}return a;
}
function list(id, values) {$(id).replaceChildren(...values.map(value=>{const li=document.createElement("li");li.textContent=value;return li;}));}
function renderOptuna(search) {
  const progress=search.progress;
  $("optuna-progress").hidden=!progress;
  if(!progress)return;
  const phases=[...new Set(progress.studies.map(study=>study.phase))].join(', ');
  const stats=[['Configurations',`${progress.completed} / ${progress.total} complete · ${progress.proposed} suggested`],
    ['Sampling phase',phases],['Seed runs',`${progress.completed_seeds} complete · ${progress.running_seeds} running`],
    ['Completed trial budgets',`${format(progress.completed_simulated_seconds,0)} / ${format(progress.allocated_simulated_seconds,0)} simulated s`]];
  $("optuna-stats").replaceChildren(...stats.map(([label,value])=>{
    const card=document.createElement('div');card.className='tuning-stat';
    const title=document.createElement('span');title.textContent=label;
    const metric=document.createElement('strong');metric.textContent=value;
    card.append(title,metric);return card;
  }));
  const best=progress.studies.map(study=>study.best_candidate).filter(Boolean);
  $("optuna-best").textContent=best.length ? best.map(row=>`Best so far: ${row.id}, ${format(row.mean_score,3)}${search.metric==='loss' ? ' NLL' : '% accuracy'}, parameters ${JSON.stringify(row.parameters)}${row.selected_time != null ? ', shared checkpoint '+row.selected_time+' s' : ''}.`).join(' ')+(progress.completed<progress.total ? ' This is provisional until the search completes.' : ' Search complete.') : 'Waiting for the first configuration to finish all paired seeds.';
  const host=$("optuna-chart");host.replaceChildren();
  const history=progress.history.filter(row=>row.best_score != null);
  if(!history.length){const empty=document.createElement('p');empty.className='notice';empty.textContent='The score history will appear after the first finite full-budget configuration.';host.append(empty);return;}
  const svgNS='http://www.w3.org/2000/svg';
  const svg=document.createElementNS(svgNS,'svg');svg.setAttribute('viewBox','0 0 800 280');svg.setAttribute('role','img');
  const title=document.createElementNS(svgNS,'title');title.textContent='Optuna validation scores and best score by completed configuration. The full values are in the tuning table.';svg.append(title);
  const values=history.flatMap(row=>[row.score,row.best_score]).filter(value=>value != null);
  let low=Math.min(...values),high=Math.max(...values);const pad=Math.max((high-low)*.1,.1);low-=pad;high+=pad;
  const maxX=Math.max(2,...progress.studies.map(study=>study.total));
  const x=number=>65+(number-1)/(maxX-1)*710,y=value=>225-(value-low)/(high-low)*190;
  function element(tag,attrs,text){const node=document.createElementNS(svgNS,tag);for(const [key,value] of Object.entries(attrs))node.setAttribute(key,value);if(text != null)node.textContent=text;svg.append(node);return node;}
  for(let tick=0;tick<=4;tick++){const value=low+(high-low)*tick/4;const at=y(value);element('line',{x1:65,x2:775,y1:at,y2:at,stroke:'var(--line)'});element('text',{x:57,y:at+4,'text-anchor':'end','font-size':12,fill:'currentColor'},format(value,2));}
  for(const number of [...new Set([1,Math.ceil(maxX/2),maxX])])element('text',{x:x(number),y:246,'text-anchor':'middle','font-size':12,fill:'currentColor'},number);
  element('text',{x:420,y:272,'text-anchor':'middle','font-size':13,fill:'currentColor'},'Configuration number');
  element('text',{x:65,y:18,'font-size':13,fill:'currentColor'},search.metric==='loss' ? 'Validation NLL ↓' : 'Mean validation accuracy % ↑');
  const colors=['#72e4bf','#83bfff','#dda7ff','#ffbf86'];
  progress.studies.forEach((study,index)=>{
    const rows=history.filter(row=>row.study===study.study),color=colors[index%colors.length];
    element('polyline',{points:rows.map(row=>`${x(row.number)},${y(row.best_score)}`).join(' '),fill:'none',stroke:color,'stroke-width':2.5});
    rows.filter(row=>row.score != null).forEach(row=>{
      const dot=element('circle',{cx:x(row.number),cy:y(row.score),r:4,fill:color,opacity:.65});
      const tip=document.createElementNS(svgNS,'title');tip.textContent=`${row.candidate_id}: ${row.score}; best so far ${row.best_score}; ${row.sampling_phase}`;dot.append(tip);
    });
  });
  host.append(svg);
}
function render(data) {
  document.title=`${data.label} · Methodology`;
  $("title").textContent=data.label;$("description").textContent=data.description;
  $("status").textContent=`${data.phase} · ${data.status}`;
  $("paper-links").replaceChildren(...(data.paper ? [link("Read paper ↗",data.paper.url)] : []));
  if(data.author_code) $("paper-links").append(link("Authors’ code ↗",data.author_code.url));
  else if(data.paper) {const button=document.createElement("button");button.textContent="Authors’ code · unverified";button.disabled=true;$("paper-links").append(button);}
  if(data.supplement)$("paper-links").append(link("Authors’ supplement ↗",data.supplement.url));
  $("citation").textContent=data.paper ? `${data.paper.title} — ${data.paper.authors}. ${data.paper.venue}.` : "In-repository research protocol; no separate publication is claimed.";
  $("code-note").textContent=data.author_code_note ?? "";
  list("steps",data.steps);
  $("implementation-note").textContent=data.implementation_note;
  $("implementation-links").replaceChildren(...data.implementation_files.map((path,i)=>link(`Simulator source: ${path.split("/").pop()}`,`/implementation?algorithm=${encodeURIComponent(data.id)}&file=${i}${workloadSuffix}`,false)));
  textRows("parameters",data.experiment_parameters,["parameter","value"]);
  textRows("fixed-parameters",data.parameter_notes,["parameter","value","role"]);
  const search=data.tuning;
  const demo=data.id === "demo";
  const teleportation=data.id === "teleportation";
  const conflux=data.id === "conflux-v2";
  $("parameter-notes-title").textContent=demo || teleportation || conflux ? "Algorithm parameters and search ranges" : "Fixed algorithm parameters";
  $("search-title").textContent=demo ? "Joint learning-rate, accumulation and compression search" : teleportation ? "Joint learning-rate, local-step, active-client and topology search" : conflux ? "Joint learning-rate, local-step, active-client, chunk and reducer search" : "Joint learning-rate and local-step search";
  const tuningData=search.partitioner === 'dirichlet' ? `non-IID data (Dirichlet α = ${search.alpha})` : 'IID data';
  $("tuning-setting-note").textContent=`Tuning uses ${tuningData} and ${search.hardware_profile === 'heterogeneous' ? 'heterogeneous' : 'homogeneous'} hardware. The displayed setting describes the subsequent evaluation; it does not start a separate search.`;
  renderOptuna(search);
  $("tuning-progress").textContent=`${search.completed_trials} / ${search.total_trials} trials complete`;
  const localSteps=[...new Set(search.groups.map(row=>row.local_steps))];
  const criterion=search.metric === 'loss' ? 'lowest mean validation NLL' : 'highest mean validation accuracy';
  const timing=search.metric === 'best_mean_accuracy' ? `at one shared scheduled training checkpoint within ${search.budget} seconds; initialization is a reference only. Earliest checkpoint breaks within-pair ties` : 'at the exact endpoint';
  const adaptive=search.method === 'tpe';
  const method=adaptive ? `TPE adaptive search, at least ${search.n_startup_trials} startup configurations, batches of ${search.batch_size}. Search spaces: ${JSON.stringify(search.search_spaces)}.` : `Joint grid: local steps H ∈ {${localSteps.join(', ')}} × declared learning rates.`;
  $("tuning-rule").textContent=`${method} Each configuration uses ${search.seeds.length} seeds (${search.seeds.join(", ")}) and the full ${search.budget} simulated seconds per seed, without early stopping. Select the ${criterion} ${timing}; lower learning rate, then variant name${adaptive ? ' and candidate ID' : ''}, break between-candidate ties. Freeze all selected parameters for every evaluation setting; retain the validation checkpoint as provenance. The leaderboard reports each evaluation seed’s best accuracy within the selected budget and its corresponding loss, averaged across seeds.`;
  $("selected-pair").textContent=search.selected_variant ? `Frozen parameters: ${JSON.stringify(search.selected_parameters ?? {local_steps:search.selected_local_steps,learning_rate:search.selected_learning_rate})}${search.selected_time != null ? ', checkpoint = '+search.selected_time+' s' : ''}. Selected using validation data only.` : "No configuration has been selected. All trials must finish before selection and test evaluation.";
  $("selected-pair").textContent=$("selected-pair").textContent.replace(", checkpoint = ", ", original validation checkpoint = ");
  if (data.evaluation?.primary_time != null) $("selected-pair").textContent += ` Primary evaluation budget: ${data.evaluation.primary_time} s; report the best accuracy within that budget for each seed.`;
  $("boundary-note").hidden=!search.boundary_winner;
  const head=document.createElement("tr");
  const extraParameters=demo ? [["demo_decay","Residual decay"],["demo_chunk","DCT block size"],["demo_topk","Coefficients / block"]] : teleportation ? [["sample_size","Active clients"],["teleportation_topology","Topology"]] : conflux ? [["sample_size","Active clients"],["chunks_in_sample","Chunks"],["relay_count","Reducer cap"]] : [];
  for(const label of [...(adaptive ? ["Candidate"] : []),demo ? "Accumulation H" : "Local steps H","Learning rate",...extraParameters.map(item=>item[1]),"Checkpoint s","Accuracy % ± SD","Loss ± SD","Seeds","Wall time / seed","Status"]){const th=document.createElement("th");th.scope="col";th.textContent=label;head.append(th);}
  $("search-head").replaceChildren(head);
  $("search-rows").replaceChildren(...search.groups.map(row=>{
    const tr=document.createElement("tr");if(row.selected)tr.className="selected-row";
    const estimate=(v,digits=2)=>v ? `${format(v.mean,digits)} ± ${format(v.sd,digits)}` : "—";
    for(const value of [...(adaptive ? [row.candidate_id] : []),row.local_steps])cell(tr,value);
    learningRateCell(tr,row.learning_rate);
    for(const [key] of extraParameters)cell(tr,row.parameters?.[key]);
    cell(tr,row.selected_time);
    const accuracy=cell(tr,estimate(row.accuracy));
    accuracy.title=search.seeds.map((seed,index)=>{
      const value=search.metric === 'loss' ? search.trials.find(trial=>trial.name===`${row.candidate_id}-s${seed}`)?.accuracy : row.seed_scores[index];
      return `Seed ${seed}: ${value == null ? 'Pending' : format(value)+'%'}`;
    }).join('\n');
    for(const value of [estimate(row.loss,3),`${row.completed_seeds}/${row.expected_seeds}`])cell(tr,value);
    const wallTime=cell(tr,duration(row.wall_seconds));
    wallTime.title=`Mean recorded wall-clock time per seed (${row.timed_seeds ?? 0}/${row.expected_seeds} seeds); running trials use their latest elapsed time.`;
    statusCell(tr,row.status,row.selected);
    return tr;
  }));
  textRows("trial-rows",search.trials.map(row=>({...row,accuracy:format(row.accuracy),loss:format(row.loss,3)})),["local_steps","learning_rate","seed","wall_seconds","status","accuracy","loss","error"]);
  $("trial-rows").closest("table").querySelector("th").textContent=demo ? "Accumulation H" : "Local steps H";
  $("back").href=`/?setting=${encodeURIComponent(selectedSetting)}${workloadSuffix}`;
  $("config-link").href=`/algorithm-config.json?algorithm=${encodeURIComponent(data.id)}${workloadSuffix}`;
  $("configuration").textContent=JSON.stringify(data.configuration,null,2);
}
async function refresh() {
  const own=++generation;clearTimeout(timer);
  try {
    const response=await fetch(`/settings.json${workload ? '?workload='+encodeURIComponent(workload) : ''}`,{cache:"no-store"});if(!response.ok)throw Error(`Cannot load settings (${response.status})`);
    const catalog=await response.json();if(own!==generation)return;
    selectedSetting=selectedSetting ?? catalog.default;
    if(!catalog.settings.some(entry=>entry.id===selectedSetting))selectedSetting=catalog.default;
    const picker=$("setting");
    if([...picker.options].map(o=>o.value).join()!==catalog.settings.map(s=>s.id).join())picker.replaceChildren(...catalog.settings.map(s=>new Option(s.label,s.id)));
    picker.value=selectedSetting;
    const result=await fetch(`/algorithm.json?algorithm=${encodeURIComponent(identifier)}&setting=${encodeURIComponent(selectedSetting)}${workloadSuffix}`,{cache:"no-store"});
    if(!result.ok)throw Error(`Cannot load algorithm evidence (${result.status})`);
    const data=await result.json();if(own!==generation)return;
    render(data);$("connection").textContent=data.error ? `Campaign error: ${data.error}` : "Refreshes every 2 seconds. Only recorded experiment observations populate the tuning tables.";
  } catch(error){if(own===generation)$("connection").textContent=error.message;}
  finally {if(own===generation)timer=setTimeout(refresh,2000);}
}
$("setting").addEventListener("change",()=>{selectedSetting=$("setting").value;const q=new URLSearchParams(window.location.search);q.set("setting",selectedSetting);window.history.replaceState(null,"",`${window.location.pathname}?${q}`);refresh();});
refresh();
