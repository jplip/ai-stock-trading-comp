(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const esc = value => String(value == null ? '' : value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const number = (value, digits = 1) => finite(value) ? value.toLocaleString('en-US', {minimumFractionDigits:digits, maximumFractionDigits:digits}) : '—';
  const pct = (value, digits = 1, signed = false) => finite(value) ? `${signed && value > 0 ? '+' : ''}${number(value * 100, digits)}%` : '—';
  const cash = value => finite(value) ? value.toLocaleString('en-US', {style:'currency', currency:'USD', maximumFractionDigits:0}) : '—';
  const shortDate = value => {
    if (!value) return '—';
    const parsed = new Date(String(value).slice(0,10) + 'T12:00:00Z');
    return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric',timeZone:'UTC'});
  };
  const text = (id, value) => { $(id).textContent = value; };
  const state = {bars:[], sourceName:'Synthetic demo', synthetic:true, warnings:[], dataVersion:0, result:null, applied:null, mode:'equity', running:false, chart:null, hoverIndex:0, toastTimer:null};
  const controls = ['threshold','fees','exposure','capital'];

  function setTab(name, replaceHash = true) {
    if (!['overview','notes','data'].includes(name)) name='overview';
    document.querySelectorAll('.nav-tab').forEach(button => {
      const active=button.dataset.tab===name;
      button.classList.toggle('active',active);
      if (active) button.setAttribute('aria-current','page'); else button.removeAttribute('aria-current');
    });
    document.querySelectorAll('.tab-panel').forEach(panel => {panel.hidden=panel.id!=='section-'+name;});
    if (replaceHash && location.hash!=='#'+name) {
      try {history.replaceState(null,'','#'+name);} catch (_) { /* File previews may restrict history. */ }
    }
    if (name==='overview' && state.result) renderChart();
  }

  function values() {
    return {initialCapital:Number($('capital').value),feeBps:Number($('fees').value),threshold:Number($('threshold').value),maxExposure:Number($('exposure').value),targetVol:.15,trainFraction:.6,validationFraction:.2,retrainEvery:20,sourceName:state.sourceName,synthetic:state.synthetic};
  }
  function fingerprint() { return JSON.stringify({options:values(),dataVersion:state.dataVersion}); }
  function updateDirty() {
    text('threshold-value',pct(Number($('threshold').value),0));
    if (state.running) return;
    const dirty=state.applied!==null&&state.applied!==fingerprint();
    $('run-state').classList.toggle('stale',dirty);
    if (dirty) text('run-state','Changes pending · run to apply');
    else if (state.result) text('run-state','Experiment complete · '+number(state.result.meta.elapsedMs||0,0)+' ms');
    else text('run-state','Ready to run');
    $('staged-notice').hidden=!dirty;
    if (dirty) text('staged-notice',`Pending experiment: ${state.sourceName} · ${state.bars.length.toLocaleString('en-US')} daily rows. Results below belong to the last completed run.`);
  }
  function error(message) {
    text('error-text',message);
    $('error-banner').hidden=false;
    $('error-banner').scrollIntoView({block:'nearest',behavior:'auto'});
  }
  function clearError() { $('error-banner').hidden=true; }
  function toast(message) {
    clearTimeout(state.toastTimer);
    text('toast',message);$('toast').hidden=false;
    state.toastTimer=setTimeout(()=>{$('toast').hidden=true;},3500);
  }
  function dataStatus() {
    const warnings=state.warnings.length?' '+state.warnings.join(' '):'';
    text('data-status',`${state.sourceName} · ${state.bars.length.toLocaleString('en-US')} daily rows loaded.${warnings}`);
  }
  function loadDemo() {
    if (!window.SignalEngine) {error('The research engine did not load. Keep index.html, app.js, style.css, and engine.js in the same folder, then reopen index.html.');return;}
    state.bars=SignalEngine.generateDemo(42,760);state.sourceName='Synthetic demo';state.synthetic=true;state.warnings=[];state.dataVersion++;
    $('csv-file').value='';clearError();dataStatus();updateDirty();
    if (state.result) toast('Reproducible demo loaded. Run experiment to apply.');
  }
  async function loadFile(file) {
    if (!file) return;
    clearError();
    if (file.size>5*1024*1024) {error('This file is larger than 5 MB. Use a CSV containing 180–5,000 daily rows for a single stock.');return;}
    try {
      text('data-status','Reading '+file.name+'…');
      const input=await file.text();
      const parsed=SignalEngine.parseCSV(input);
      state.bars=parsed.bars;state.sourceName=file.name;state.synthetic=false;state.warnings=parsed.warnings||[];state.dataVersion++;
      dataStatus();updateDirty();toast('CSV loaded. Run experiment to test it.');
    } catch (e) {dataStatus();error(e.message||'The selected CSV could not be read.');}
    finally {$('csv-file').value='';}
  }
  async function run() {
    if (state.running) return;
    clearError();
    if (!window.SignalEngine) {error('The research engine is missing. Reopen index.html from the complete Signal Lab folder.');return;}
    for (const id of ['fees','capital']) {
      if (!$(id).checkValidity() || !$(id).value.trim()) {error(id==='fees'?'Enter a fee from 0 to 100 basis points.':'Enter starting capital from $100 to $1,000,000,000.');$(id).focus();return;}
    }
    const options=values(),applied=fingerprint(),input=state.bars.slice(),warnings=state.warnings.slice();
    state.running=true;document.body.classList.add('loading');
    $('run-button').disabled=true;['upload-button','data-upload-button','use-demo',...controls].forEach(id=>{$(id).disabled=true;});
    text('run-text','Testing models…');text('run-state','Training → validation → test');$('run-state').classList.remove('stale');
    await new Promise(resolve=>setTimeout(resolve,35));
    try {
      const result=SignalEngine.runExperiment(input,options);
      result.meta.warnings=[...new Set([...(result.meta.warnings||[]),...warnings])];
      state.result=result;state.applied=applied;
      render(result);
      ['export-signal','export-predictions','export-report'].forEach(id=>{$(id).disabled=false;});
    } catch (e) {error(e.message||'The experiment could not be completed.');}
    finally {
      state.running=false;document.body.classList.remove('loading');
      $('run-button').disabled=false;['upload-button','data-upload-button','use-demo',...controls].forEach(id=>{$(id).disabled=false;});
      text('run-text','Run experiment');updateDirty();
    }
  }
  function render(r) {
    const m=r.metrics,b=r.baselineMetrics;
    $('source-badge').className='badge '+(r.meta.synthetic?'synthetic':'imported');
    $('source-badge').innerHTML='<span aria-hidden="true">◈</span> '+(r.meta.synthetic?'SYNTHETIC DEMO':'IMPORTED DATA');
    text('source-detail',r.meta.sourceName+' · '+r.meta.rows.toLocaleString('en-US')+' daily rows');
    text('test-window',shortDate(r.meta.testRange[0])+' — '+shortDate(r.meta.testRange[1]));
    text('metric-return',pct(m.totalReturn,2,true));$('metric-return').className='metric-number '+(m.totalReturn>=0?'positive':'negative');
    const delta=m.totalReturn-b.totalReturn;
    $('metric-return-note').innerHTML=`<span class="${delta>=0?'positive':'negative'}">${delta>0?'+':''}${esc(number(delta*100,2))} pp</span> vs. buy &amp; hold (${esc(pct(b.totalReturn,2,true))})`;
    text('metric-drawdown',pct(-m.maxDrawdown,2));text('metric-sharpe',number(m.sharpe,2));text('metric-accuracy',pct(m.accuracy,1));
    text('metric-accuracy-note',r.meta.testCount.toLocaleString('en-US')+' test predictions · 50% decision cutoff');
    text('ending-equity',cash(m.endingEquity));
    const latest=r.latest;
    text('signal-probability',pct(latest.probability,1));
    $('probability-ring').style.background=`conic-gradient(var(--mint) 0% ${Math.max(0,Math.min(100,latest.probability*100))}%,#2a3945 0% 100%)`;
    text('signal-action',latest.position>0?'STOCK + CASH':'HOLD CASH');$('signal-action').className='signal-pill'+(latest.position>0?'':' cash');
    text('signal-date',shortDate(latest.date));text('signal-allocation',pct(latest.position,1));
    $('allocation-fill').style.width=pct(latest.position,2);
    text('signal-disagreement',number(latest.disagreement*100,1)+' percentage points');text('signal-volatility',pct(latest.volatility,1));
    $('model-cards').innerHTML=r.models.map((model,i)=>`<article class="model-card"><div class="model-card-top"><span class="model-icon" aria-hidden="true">${['∑','⋈','∴'][i]}</span><span class="model-index">0${i+1}</span></div><h3>${esc(model.name)}</h3><div class="model-weight">${esc(pct(model.weight,1))}<small>weight</small></div><div class="model-weight-track"><span style="width:${Math.max(0,Math.min(100,model.weight*100))}%"></span></div><div class="model-stat"><span>Validation Brier</span><strong>${esc(number(model.validationBrier,3))}</strong></div><div class="model-stat"><span>Test accuracy</span><strong>${esc(pct(model.testAccuracy,1))}</strong></div><div class="model-stat"><span>Latest up-score</span><strong>${esc(pct(model.latestProbability,1))}</strong></div><p class="model-description">${esc(model.description)}</p></article>`).join('');
    const sorted=r.features.slice().sort((a,b)=>b.importance-a.importance).slice(0,6),signed=sorted.every(f=>finite(f.coefficient));
    const max=Math.max(.00001,...sorted.map(f=>signed?Math.abs(f.coefficient):f.importance));
    $('feature-list').innerHTML=sorted.map(f=>{
      const value=signed?f.coefficient:f.importance,negative=value<0;
      return `<div class="feature-row" title="${esc(f.description)}"><span class="feature-name">${esc(f.name)}</span><div class="feature-bar"><span class="${negative?'negative':''}" style="width:${Math.min(100,Math.abs(value)/max*100)}%"></span></div><span class="feature-value ${negative?'negative':''}">${esc(signed?((value>0?'+':'')+number(value,2)):pct(value,0))}</span></div>`;
    }).join('');
    document.querySelector('.feature-panel .panel-footnote').textContent=signed?'Top six standardized logistic coefficients from the latest fit. Positive means a higher model up-score; these are associations, not causes.':'Top six shares of absolute standardized logistic coefficients. Larger means more influence within the logistic model, not the ensemble; these are associations, not causes.';
    const total=r.meta.trainCount+r.meta.validationCount+r.meta.testCount;
    text('row-count',total.toLocaleString('en-US')+' eligible observations');
    $('split-labels').innerHTML=[['Training',r.meta.trainRange,r.meta.trainCount],['Validation',r.meta.validationRange,r.meta.validationCount],['Final test',r.meta.testRange,r.meta.testCount]].map(([name,range,count])=>`<div><strong>${esc(shortDate(range[0]))} — ${esc(shortDate(range[1]))}</strong>${esc(count.toLocaleString('en-US'))} ${esc(name.toLowerCase())} observations</div>`).join('');
    $('audit-list').innerHTML=r.audit.map(a=>`<span class="audit-item${a.passed?'':' failed'}" title="${esc(a.detail)}"><span aria-hidden="true">${a.passed?'✓':'!'}</span><span class="sr-only">${a.passed?'Passed: ':'Failed: '}</span>${esc(a.label)}</span>`).join('');
    text('trade-count',m.tradeCount.toLocaleString('en-US')+' allocation changes');text('average-exposure',pct(m.exposure,1)+' average stock exposure');
    $('trade-rows').innerHTML=r.trades.length?r.trades.slice(-20).reverse().map(t=>`<tr><td>${esc(t.signalDate)}</td><td>${esc(t.executionDate)}</td><td><span class="table-action${t.action==='SELL'||t.action==='LIQUIDATE'?' cash':''}">${esc(t.action)}</span></td><td class="numeric">${esc(pct(t.weight,1))}</td><td class="numeric">${esc(pct(t.probability,1))}</td><td class="numeric ${t.netReturn>=0?'return-positive':'return-negative'}">${esc(pct(t.netReturn,2,true))}</td><td class="numeric">${esc(cash(t.equity))}</td></tr>`).join(''):'<tr><td colspan="7" class="empty-row">No allocation changes. The strategy stayed in cash for this test.</td></tr>';
    if (r.meta.warnings.length) text('data-status',r.meta.sourceName+' · '+r.meta.warnings.join(' '));
    renderChart();
  }

  function renderChart() {
    if (!state.result) return;
    const r=state.result,cap=r.options.initialCapital;
    const initial={date:r.predictions[0]?.executionDate||r.meta.testRange[0],equity:cap,baseline:cap,drawdown:0};
    const rows=[initial,...r.series];let basePeak=cap;
    const points=rows.map(row=>{basePeak=Math.max(basePeak,row.baseline);return {...row,baselineDrawdown:1-row.baseline/basePeak};});
    const drawdown=state.mode==='drawdown',W=900,H=285,left=11,right=68,top=20,bottom=32,pw=W-left-right,ph=H-top-bottom;
    const vals=points.flatMap(p=>drawdown?[-p.drawdown,-p.baselineDrawdown]:[p.equity,p.baseline]);
    let lo=Math.min(...vals),hi=Math.max(...vals),span=hi-lo;
    if (span<.000001) {span=drawdown?.01:Math.max(1,cap*.04);lo-=span*.5;hi+=span*.5;}
    else {lo-=span*.12;hi+=span*.12;}
    if (drawdown) hi=Math.max(hi,0);
    const x=i=>left+i/(points.length-1)*pw,y=v=>top+(hi-v)/(hi-lo)*ph;
    const get=(p,baseline=false)=>drawdown?-(baseline?p.baselineDrawdown:p.drawdown):(baseline?p.baseline:p.equity);
    const path=base=>points.map((p,i)=>(i?'L':'M')+x(i).toFixed(2)+','+y(get(p,base)).toFixed(2)).join(' ');
    const strategyPath=path(false),area=strategyPath+` L${x(points.length-1)},${top+ph} L${left},${top+ph} Z`;
    const axisLabel=v=>drawdown?pct(v,1):(Math.abs(v)>=1000000?'$'+number(v/1000000,1)+'m':Math.abs(v)>=1000?'$'+number(v/1000,1)+'k':cash(v));
    let grid='';
    for (let i=0;i<5;i++) {const v=lo+(hi-lo)*i/4,py=y(v);grid+=`<line class="chart-grid" x1="${left}" x2="${W-right+4}" y1="${py}" y2="${py}"/><text x="${W-right+14}" y="${py+3}">${esc(axisLabel(v))}</text>`;}
    let labels='';
    for(let j=0;j<5;j++){const i=Math.round((points.length-1)*j/4),date=new Date(points[i].date+'T12:00:00Z'),label=date.toLocaleDateString('en-US',{month:'short',day:'numeric',timeZone:'UTC'});labels+=`<text x="${x(i)}" y="${H-9}" text-anchor="${j===0?'start':j===4?'end':'middle'}">${esc(label)}</text>`;}
    $('equity-chart').innerHTML=`<title>${drawdown?'Drawdown':'Simulated account value'}: Signal Lab versus buy and hold</title><desc>The final test window, after modeled fees. Use left and right arrow keys to inspect individual dates.</desc><defs><linearGradient id="equity-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#c0f4b1" stop-opacity=".12"/><stop offset="100%" stop-color="#c0f4b1" stop-opacity="0"/></linearGradient></defs>${grid}<path d="${area}" fill="url(#equity-fill)"/><path class="baseline-line" d="${path(true)}"/><path class="strategy-line" d="${strategyPath}"/>${labels}<g id="chart-hover" visibility="hidden"><line id="hover-line" class="hover-line" y1="${top}" y2="${top+ph}"/><circle id="hover-strategy" r="4" fill="#c0f4b1" stroke="#10171e" stroke-width="2"/><circle id="hover-baseline" r="3" fill="#8dc9ee" stroke="#10171e" stroke-width="2"/></g>`;
    $('equity-chart').setAttribute('aria-label',(drawdown?'Drawdown':'Equity')+' comparison chart. Focus and use arrow keys to inspect dates.');
    $('chart-empty').hidden=true;$('chart-tooltip').hidden=true;
    state.chart={points,x,y,get,pw,left,W};state.hoverIndex=Math.min(state.hoverIndex,points.length-1);
    $('equity-view').classList.toggle('selected',!drawdown);$('drawdown-view').classList.toggle('selected',drawdown);
    $('equity-view').setAttribute('aria-pressed',String(!drawdown));$('drawdown-view').setAttribute('aria-pressed',String(drawdown));
  }
  function showChartPoint(index) {
    if (!state.chart) return;
    const c=state.chart;index=Math.max(0,Math.min(c.points.length-1,index));state.hoverIndex=index;
    const p=c.points[index],px=c.x(index),sy=c.y(c.get(p)),by=c.y(c.get(p,true));
    $('chart-hover').setAttribute('visibility','visible');$('hover-line').setAttribute('x1',px);$('hover-line').setAttribute('x2',px);
    $('hover-strategy').setAttribute('cx',px);$('hover-strategy').setAttribute('cy',sy);$('hover-baseline').setAttribute('cx',px);$('hover-baseline').setAttribute('cy',by);
    const format=state.mode==='drawdown'?v=>pct(v,2):cash;
    $('chart-tooltip').innerHTML=`<strong>${esc(shortDate(p.date))}</strong><div class="tip-strategy"><span>Signal Lab</span><span>${esc(format(c.get(p)))}</span></div><div class="tip-baseline"><span>Buy &amp; hold</span><span>${esc(format(c.get(p,true)))}</span></div>`;
    $('chart-tooltip').hidden=false;
    const wrap=$('chart-wrap'),width=wrap.getBoundingClientRect().width,tipWidth=$('chart-tooltip').offsetWidth;
    $('chart-tooltip').style.left=Math.max(0,Math.min(width-tipWidth,px/c.W*width-tipWidth/2))+'px';$('chart-tooltip').style.top='1px';
    $('equity-chart').setAttribute('aria-label',`${shortDate(p.date)}. Signal Lab ${format(c.get(p))}. Buy and hold ${format(c.get(p,true))}. Use arrow keys for other dates.`);
  }
  function hideChartPoint() {if($('chart-hover'))$('chart-hover').setAttribute('visibility','hidden');$('chart-tooltip').hidden=true;}

  function download(content,filename,type) {
    const blob=new Blob([content],{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=filename;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('Downloaded '+filename);
  }
  const csvCell = v => /[",\r\n]/.test(String(v))?'"'+String(v).replace(/"/g,'""')+'"':String(v);
  function csv(rows) {return rows.map(row=>row.map(csvCell).join(',')).join('\r\n')+'\r\n';}
  function dataCSV(bars) {return csv([['Date','Open','High','Low','Close','Volume'],...bars.map(b=>[b.date,b.open,b.high,b.low,b.close,b.volume])]);}
  function exportPredictions() {
    if (!state.result) return;
    download(csv([['signal_date','execution_date','outcome_date','up_score_uncalibrated','stock_allocation','actual_open_to_open_return'],...state.result.predictions.map(p=>[p.signalDate,p.executionDate,p.exitDate,p.probability,p.position,p.actualReturn])]),'signal-lab-predictions.csv','text/csv;charset=utf-8');
  }
  function exportReport() {
    if (!state.result) return;
    download(JSON.stringify({application:'Signal Lab',version:SignalEngine.version,notice:'Educational simulation. Synthetic demo results are not market evidence. Probabilities are uncalibrated. No live orders.',...state.result},null,2),'signal-lab-report.json','application/json');
  }
  function exportSignal() {
    if (!state.result) return;
    const r=state.result;
    download(JSON.stringify({application:'Signal Lab',type:'proposed_next_open_allocation',sourceName:r.meta.sourceName,synthetic:r.meta.synthetic,signalAtClose:r.latest.date,entry:'Next available session open',holdingWindow:'Next open to the following open',stockAllocation:r.latest.position,cashAllocation:1-r.latest.position,action:r.latest.position>0?'TARGET_LONG_ALLOCATION':'TARGET_CASH',modelUpScore:r.latest.probability,probabilitiesCalibrated:false,threshold:r.options.threshold,maxExposure:r.options.maxExposure,estimatedAnnualVolatility:r.latest.volatility,notice:'Browser research diagnostic only. No live order was placed. This JSON is not a competition submission; use the separate .pkl model and .ipynb notebook from the competition kit.'},null,2),'signal-lab-next-signal.json','application/json');
  }

  document.querySelectorAll('.nav-tab').forEach(button=>button.addEventListener('click',()=>setTab(button.dataset.tab)));
  document.querySelectorAll('[data-go-tab]').forEach(link=>link.addEventListener('click',event=>{event.preventDefault();setTab(link.dataset.goTab);$('section-'+link.dataset.goTab).scrollIntoView({block:'start'});}));
  document.querySelector('.site-header .brand').addEventListener('click',event=>{event.preventDefault();setTab('overview');});
  window.addEventListener('hashchange',()=>setTab(location.hash.slice(1),false));
  ['upload-button','data-upload-button'].forEach(id=>$(id).addEventListener('click',()=>$('csv-file').click()));
  $('csv-file').addEventListener('change',event=>loadFile(event.target.files[0]));
  $('use-demo').addEventListener('click',loadDemo);$('run-button').addEventListener('click',run);$('dismiss-error').addEventListener('click',clearError);
  controls.forEach(id=>{$(id).addEventListener('input',updateDirty);$(id).addEventListener('change',updateDirty);});
  $('equity-view').addEventListener('click',()=>{state.mode='equity';renderChart();});$('drawdown-view').addEventListener('click',()=>{state.mode='drawdown';renderChart();});
  $('equity-chart').addEventListener('pointermove',event=>{if(!state.chart)return;const rect=event.currentTarget.getBoundingClientRect(),c=state.chart,svgX=(event.clientX-rect.left)/rect.width*c.W;showChartPoint(Math.round((svgX-c.left)/c.pw*(c.points.length-1)));});
  $('equity-chart').addEventListener('pointerleave',hideChartPoint);$('equity-chart').addEventListener('blur',hideChartPoint);
  $('equity-chart').addEventListener('keydown',event=>{if(!state.chart)return;if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){event.preventDefault();showChartPoint(event.key==='Home'?0:event.key==='End'?state.chart.points.length-1:state.hoverIndex+(event.key==='ArrowRight'?1:-1));}else if(event.key==='Escape')hideChartPoint();});
  $('export-predictions').addEventListener('click',exportPredictions);$('export-report').addEventListener('click',exportReport);$('export-signal').addEventListener('click',exportSignal);
  $('download-data').addEventListener('click',()=>download(dataCSV(state.bars),state.synthetic?'signal-lab-synthetic-demo.csv':'signal-lab-loaded-data.csv','text/csv;charset=utf-8'));
  $('download-template').addEventListener('click',()=>download('Date,Open,High,Low,Close,Volume\r\n2024-01-02,100,102,99,101,1200000\r\n2024-01-03,101,103,100,102,1350000\r\n2024-01-04,102,104,101,103,1280000\r\n','signal-lab-format-template.csv','text/csv;charset=utf-8'));
  setTab(location.hash.slice(1)||'overview',false);loadDemo();run();
})();
