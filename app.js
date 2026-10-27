'use strict';
const DATA = window.HARMONY_DATA;
const TASKS = {add:'Addition',remove:'Removal',replace:'Replacement',volume:'Volume adjustment'};
const METHODS = Object.keys(DATA.methods).filter(method => !['input','harmony'].includes(method));
const escapeHTML = value => String(value).replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
const formatTime = value => `${Math.floor(Math.max(0,value)/60)}:${String(Math.floor(Math.max(0,value)%60)).padStart(2,'0')}`;
const caseMap = new Map(DATA.cases.map(item => [item.id,item]));
let selectedTask = 'all';
let context;
let masterGain;
let source;
let playing = false;
let startedAt = 0;
let offset = 0;
let active = null;
let loading = false;
let requestToken = 0;
let toastTimer;
const buffers = new Map();
const playSVG = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1L11 6 2 11Z"/></svg>';
const pauseSVG = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1h3v10H2zM7 1h3v10H7z"/></svg>';

function wave(item, track) {
  const scale = Math.max(...Object.values(item.tracks).map(value => value.peak),0.001);
  const bars = track.peaks.map((peak,index) => {
    const height = Math.max(0.9,Math.min(1,peak/scale)*30);
    return `M${index*3},${(32-height)/2}h1.8v${height}h-1.8Z`;
  }).join('');
  return `<svg viewBox="0 0 480 32" preserveAspectRatio="none" aria-hidden="true"><path class="wave-base" d="${bars}"/><path class="wave-played" d="${bars}"/></svg>`;
}
function trackRow(item, method) {
  const track = item.tracks[method];
  return `<div class="track ${method==='harmony'?'ours':''}" data-case="${item.id}" data-method="${method}"><button class="play-track" aria-label="Play ${escapeHTML(track.label)} for ${escapeHTML(item.title)}">${playSVG}</button><div class="track-label"><strong>${escapeHTML(track.label)}</strong></div><button class="wave-button" aria-label="Play or seek ${escapeHTML(track.label)}" title="Click to play from this point">${wave(item,track)}</button><span class="track-duration">${formatTime(track.duration)}</span><a class="download-track" href="${track.src}" download aria-label="Download ${escapeHTML(track.label)}" title="Download audio">↓</a></div>`;
}
function renderCase(item) {
  const index = DATA.cases.indexOf(item)+1;
  return `<article class="case" id="case-${item.id}" data-id="${item.id}"><div class="case-head"><h3>${TASKS[item.task]}</h3><span class="case-number">${String(index).padStart(2,'0')}</span></div><blockquote class="instruction">${escapeHTML(item.instruction)}</blockquote><div class="main-tracks">${['input','harmony',...METHODS].map(method => trackRow(item,method)).join('')}</div></article>`;
}
function render() {
  const items = DATA.cases.filter(item => selectedTask==='all'||item.task===selectedTask);
  document.querySelector('#case-list').innerHTML = items.map(renderCase).join('');
  document.querySelector('#empty-state').hidden = Boolean(items.length);
  updatePlayerUI();
}
function showToast(message) {
  const element = document.querySelector('#toast');
  element.textContent = message;
  element.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { element.hidden = true; },4000);
}
function ensureContext() {
  if (!context) {
    context = new (window.AudioContext || window.webkitAudioContext)();
    masterGain = context.createGain();
    masterGain.connect(context.destination);
  }
  masterGain.gain.value = Number(document.querySelector('#volume').value) * (DATA.playbackHeadroom || 1);
  return context.resume();
}
function decodeWav(arrayBuffer) {
  const data = new DataView(arrayBuffer);
  const text = (start,length) => String.fromCharCode(...new Uint8Array(arrayBuffer,start,length));
  if (text(0,4)!=='RIFF'||text(8,4)!=='WAVE') throw new Error('Unsupported audio container');
  let format,channels,rate,bits,blockAlign,dataStart,dataLength;
  let cursor = 12;
  while (cursor+8<=data.byteLength) {
    const id = text(cursor,4);
    const size = data.getUint32(cursor+4,true);
    const position = cursor+8;
    if (id==='fmt ') {
      format = data.getUint16(position,true);
      channels = data.getUint16(position+2,true);
      rate = data.getUint32(position+4,true);
      blockAlign = data.getUint16(position+12,true);
      bits = data.getUint16(position+14,true);
      if (format===65534) format = data.getUint16(position+24,true);
    } else if (id==='data') {
      dataStart = position;
      dataLength = Math.min(size,data.byteLength-position);
    }
    cursor = position+size+(size%2);
  }
  if (!dataStart||!channels||!blockAlign||![1,3].includes(format)) throw new Error('Unsupported WAV encoding');
  const frames = Math.floor(dataLength/blockAlign);
  const result = context.createBuffer(channels,frames,rate);
  const bytes = bits/8;
  for (let channel=0;channel<channels;channel++) {
    const output = result.getChannelData(channel);
    for (let frame=0;frame<frames;frame++) {
      const position = dataStart+frame*blockAlign+channel*bytes;
      let value;
      if (format===3&&bits===32) value = data.getFloat32(position,true);
      else if (format===3&&bits===64) value = data.getFloat64(position,true);
      else if (bits===16) value = data.getInt16(position,true)/32768;
      else if (bits===24) { let sample=data.getUint8(position)|(data.getUint8(position+1)<<8)|(data.getUint8(position+2)<<16);if(sample&0x800000)sample|=0xff000000;value=sample/8388608; }
      else if (bits===32) value = data.getInt32(position,true)/2147483648;
      else if (bits===8) value = (data.getUint8(position)-128)/128;
      else throw new Error('Unsupported WAV bit depth');
      output[frame] = value;
    }
  }
  return result;
}
async function loadTrack(track) {
  if (buffers.has(track.src)) {
    const buffer = buffers.get(track.src);
    buffers.delete(track.src);buffers.set(track.src,buffer);
    return buffer;
  }
  const response = await fetch(track.src);
  if (!response.ok) throw new Error(`Audio request failed (${response.status})`);
  const buffer = decodeWav(await response.arrayBuffer());
  buffers.set(track.src,buffer);
  if (buffers.size>6) buffers.delete(buffers.keys().next().value);
  return buffer;
}
function currentTime() {
  if (!active) return 0;
  return Math.min(active.buffer?.duration || active.track.duration,offset+(playing?context.currentTime-startedAt:0));
}
function stopSource() {
  if (source) { source.onended=null;try { source.stop(); } catch (_) {}source.disconnect();source=null; }
}
function pause() {
  offset = currentTime();stopSource();playing=false;updatePlayerUI();
}
function startAt(position) {
  if (!active?.buffer) return;
  stopSource();
  offset = Math.max(0,Math.min(position,active.buffer.duration));
  if (offset>=active.buffer.duration-0.01) offset=0;
  source=context.createBufferSource();source.buffer=active.buffer;source.connect(masterGain);
  source.onended=() => { offset=active.buffer.duration;playing=false;updatePlayerUI(); };
  startedAt=context.currentTime;playing=true;source.start(0,offset);updatePlayerUI();
}
async function playTrack(caseId,method,position) {
  const item=caseMap.get(caseId),track=item.tracks[method];
  if (active?.caseId===caseId&&active.method===method&&active.buffer&&!loading&&position===undefined) {
    if (playing) pause();else { await ensureContext();startAt(offset); }
    return;
  }
  const seek = position!==undefined?position:active?.caseId===caseId?currentTime():0;
  const token=++requestToken;
  pause();
  active={caseId,method,track,title:item.title,buffer:null};offset=seek;loading=true;updatePlayerUI();
  try {
    await ensureContext();
    const buffer=await loadTrack(track);
    if (token!==requestToken) return;
    active.buffer=buffer;loading=false;
    if (seek>=buffer.duration && seek>0) {
      offset=buffer.duration;playing=false;updatePlayerUI();
      showToast('This output ends before the selected point. Restart to hear it from the beginning.');
    } else startAt(seek);
  } catch (error) {
    if (token!==requestToken) return;
    loading=false;playing=false;updatePlayerUI();showToast(`Could not play this track: ${error.message}`);
  }
}
function updatePlayerUI() {
  document.querySelector('#master-play').disabled=!active||loading;
  document.querySelector('#master-play').textContent=loading?'…':playing?'Ⅱ':'▶';
  document.querySelector('#now-method').textContent=active?`${active.track.label}${loading?' · Loading…':''}`:'Select an audio track';
  document.querySelector('#now-title').textContent=active?active.title:'Original · Harmony · Baselines';
  document.querySelector('#seek').disabled=!active?.buffer;
  if (active) {
    document.querySelector('#seek').max=active.buffer?.duration||active.track.duration;
    document.querySelector('#duration').textContent=formatTime(active.buffer?.duration||active.track.duration);
  }
  document.querySelectorAll('.track').forEach(row => {
    const match=active&&row.dataset.case===active.caseId&&row.dataset.method===active.method;
    row.classList.toggle('playing',Boolean(match&&playing));
    row.querySelector('.play-track').innerHTML=match&&loading?'…':match&&playing?pauseSVG:playSVG;
    row.querySelector('.play-track').setAttribute('aria-label',`${match&&playing?'Pause':'Play'} ${DATA.methods[row.dataset.method]}`);
  });
}
function animate() {
  const time=currentTime();
  document.querySelector('#elapsed').textContent=formatTime(time);
  if (document.activeElement!==document.querySelector('#seek')) document.querySelector('#seek').value=time;
  if (active) document.querySelectorAll(`.track[data-case="${active.caseId}"]`).forEach(row => {
    const duration=caseMap.get(active.caseId).tracks[row.dataset.method].duration;
    row.querySelector('.wave-played').style.clipPath=`inset(0 ${100-Math.min(100,time/duration*100)}% 0 0)`;
  });
  requestAnimationFrame(animate);
}
document.querySelector('#case-list').addEventListener('click',event => {
  const row=event.target.closest('.track');
  if (event.target.closest('.play-track')&&row) playTrack(row.dataset.case,row.dataset.method);
  if (event.target.closest('.wave-button')&&row) {
    const button=event.target.closest('.wave-button'),bounds=button.getBoundingClientRect();
    const ratio=event.detail===0?0:Math.max(0,Math.min(1,(event.clientX-bounds.left)/bounds.width));
    playTrack(row.dataset.case,row.dataset.method,ratio*caseMap.get(row.dataset.case).tracks[row.dataset.method].duration);
  }
 });
document.querySelectorAll('.tab').forEach(button => button.addEventListener('click',() => {
  selectedTask=button.dataset.task;
  document.querySelectorAll('.tab').forEach(other => {
    other.classList.toggle('selected',other===button);
    other.setAttribute('aria-pressed',String(other===button));
  });
  render();
}));
document.querySelector('#master-play').addEventListener('click',async () => {if(playing)pause();else if(active?.buffer){await ensureContext();startAt(offset);}});
document.querySelector('#restart').addEventListener('click',async () => {if(active?.buffer){await ensureContext();startAt(0);}});
document.querySelector('#seek').addEventListener('input',event => {if(active?.buffer){if(playing)startAt(Number(event.target.value));else offset=Number(event.target.value);}});
document.querySelector('#volume').addEventListener('input',event => {if(masterGain)masterGain.gain.value=Number(event.target.value)*(DATA.playbackHeadroom||1);});
document.addEventListener('keydown',event => {
  if(event.target.closest('button,a,input,select,textarea')||event.altKey||event.ctrlKey||event.metaKey)return;
  if(event.code==='Space'&&active){event.preventDefault();document.querySelector('#master-play').click();}
  if(active&&['1','2','3'].includes(event.key)){event.preventDefault();playTrack(active.caseId,{'1':'input','2':'harmony','3':METHODS[0]}[event.key]);}
});
function openHash() {
  const id=location.hash.replace('#case-','');
  if (!caseMap.has(id)) return;
  selectedTask='all';
  document.querySelectorAll('.tab').forEach(button => {
    button.classList.toggle('selected',button.dataset.task==='all');
    button.setAttribute('aria-pressed',String(button.dataset.task==='all'));
  });
  render();
  requestAnimationFrame(()=>document.getElementById(`case-${id}`)?.scrollIntoView({behavior:'smooth',block:'start'}));
}
window.addEventListener('hashchange',openHash);
render();openHash();requestAnimationFrame(animate);
