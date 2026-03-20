import * as d3 from 'https://cdn.jsdelivr.net/npm/d3@7.9.0/+esm';

// ── State ──────────────────────────────────────────────
let extractor = null;
let ideas = [];
let simulation = null;
let clusterCenters = {};
let subclusterCenters = {};

// Two thresholds define the hierarchy:
//   - GROUP_THRESHOLD: broad groups (lower = more merging)
//   - SUBGROUP_THRESHOLD: tight subclusters within groups (higher = stricter)
let GROUP_THRESHOLD = 0.50;
let SUBGROUP_THRESHOLD = 0.65;

const PALETTE = [
  '#6366f1', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6',
  '#ec4899', '#14b8a6', '#f97316', '#06b6d4', '#84cc16',
];

// ── DOM ────────────────────────────────────────────────
const svgEl = document.getElementById('viz');
const svg = d3.select(svgEl);
const container = svg.append('g');
const hullGroup = container.append('g').attr('class', 'hulls');
const linkGroup = container.append('g').attr('class', 'links');
const nodeGroup = container.append('g').attr('class', 'nodes');
const labelGroup = container.append('g').attr('class', 'cluster-labels');
const tooltip = document.getElementById('tooltip');
const statusEl = document.getElementById('status');
const inputEl = document.getElementById('idea-input');
const submitBtn = document.getElementById('submit-btn');
const emptyState = document.getElementById('empty-state');

let width = svgEl.clientWidth;
let height = svgEl.clientHeight;

// ── Zoom ───────────────────────────────────────────────
const zoom = d3.zoom()
  .scaleExtent([0.3, 3])
  .on('zoom', (event) => container.attr('transform', event.transform));
svg.call(zoom);

// ── Resize ─────────────────────────────────────────────
window.addEventListener('resize', () => {
  width = svgEl.clientWidth;
  height = svgEl.clientHeight;
  if (simulation) {
    simulation
      .force('centerX', d3.forceX(width / 2).strength(0.01))
      .force('centerY', d3.forceY(height / 2).strength(0.01));
    updateClusterForces();
    simulation.alpha(0.2).restart();
  }
});

// ── Model Loading ──────────────────────────────────────
async function getExtractor() {
  if (extractor) return extractor;
  setStatus('Loading AI model (first time only)...');
  const { pipeline, env } = await import(
    'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.1'
  );
  env.allowLocalModels = false;
  extractor = await pipeline('feature-extraction', 'Xenova/bge-small-en-v1.5', {
    progress_callback: (p) => {
      if (p.status === 'progress') {
        setStatus(`Downloading model: ${Math.round(p.progress)}%`);
      }
    },
  });
  setStatus('Ready');
  return extractor;
}

// ── Embeddings ─────────────────────────────────────────
function cosineSimilarity(a, b) {
  let dot = 0, nA = 0, nB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    nA += a[i] * a[i];
    nB += b[i] * b[i];
  }
  return dot / (Math.sqrt(nA) * Math.sqrt(nB));
}

// ── Hierarchical Agglomerative Clustering ──────────────
// Runs once, records the full merge tree, then cuts at two thresholds
function clusterIdeas() {
  const n = ideas.length;
  if (n === 0) return { numGroups: 0, numSubgroups: 0 };
  if (n === 1) {
    ideas[0].group = 0;
    ideas[0].subgroup = '0-0';
    return { numGroups: 1, numSubgroups: 1 };
  }

  // Precompute similarity matrix
  const sim = Array.from({ length: n }, () => new Float32Array(n));
  console.group('Pairwise similarities');
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const s = cosineSimilarity(ideas[i].embedding, ideas[j].embedding);
      sim[i][j] = s;
      sim[j][i] = s;
      console.log(`${s.toFixed(3)}  "${ideas[i].text}" ↔ "${ideas[j].text}"`);
    }
  }
  console.groupEnd();

  // Run full agglomerative clustering, recording merge history
  // assignments[i] = current cluster label for idea i
  const assignments = ideas.map((_, i) => i);
  const merges = []; // { a, b, similarity } — b merged into a at this similarity

  while (true) {
    const clusterMap = {};
    for (let i = 0; i < n; i++) {
      const c = assignments[i];
      if (!clusterMap[c]) clusterMap[c] = [];
      clusterMap[c].push(i);
    }
    const clusterIds = Object.keys(clusterMap).map(Number);
    if (clusterIds.length <= 1) break;

    // Find most similar pair (average linkage)
    let bestSim = -Infinity, bestA = -1, bestB = -1;
    for (let ci = 0; ci < clusterIds.length; ci++) {
      for (let cj = ci + 1; cj < clusterIds.length; cj++) {
        const membersA = clusterMap[clusterIds[ci]];
        const membersB = clusterMap[clusterIds[cj]];
        let total = 0;
        for (const a of membersA) {
          for (const b of membersB) {
            total += sim[a][b];
          }
        }
        const avgSim = total / (membersA.length * membersB.length);
        if (avgSim > bestSim) {
          bestSim = avgSim;
          bestA = clusterIds[ci];
          bestB = clusterIds[cj];
        }
      }
    }

    // Record this merge (even below threshold — we need the full tree)
    merges.push({ a: bestA, b: bestB, similarity: bestSim });

    // Merge bestB into bestA
    for (let i = 0; i < n; i++) {
      if (assignments[i] === bestB) assignments[i] = bestA;
    }
  }

  // Now cut the dendrogram at two thresholds by replaying merges
  // Subgroups: only apply merges with similarity >= SUBGROUP_THRESHOLD
  const subAssign = ideas.map((_, i) => i);
  for (const m of merges) {
    if (m.similarity >= SUBGROUP_THRESHOLD) {
      for (let i = 0; i < n; i++) {
        if (subAssign[i] === m.b) subAssign[i] = m.a;
      }
    }
  }

  // Groups: apply merges with similarity >= GROUP_THRESHOLD
  const groupAssign = ideas.map((_, i) => i);
  for (const m of merges) {
    if (m.similarity >= GROUP_THRESHOLD) {
      for (let i = 0; i < n; i++) {
        if (groupAssign[i] === m.b) groupAssign[i] = m.a;
      }
    }
  }

  // Renumber groups to 0..k-1
  const uniqueGroups = [...new Set(groupAssign)];
  const groupRemap = {};
  uniqueGroups.forEach((c, i) => (groupRemap[c] = i));

  // Renumber subgroups within each group
  const subRemap = {};
  let subId = 0;
  for (let g = 0; g < uniqueGroups.length; g++) {
    const groupIdx = uniqueGroups[g];
    const memberIndices = groupAssign
      .map((ga, i) => (ga === groupIdx ? i : -1))
      .filter((i) => i >= 0);
    const subsInGroup = [...new Set(memberIndices.map((i) => subAssign[i]))];
    for (const s of subsInGroup) {
      subRemap[s] = { group: g, sub: subId++ };
    }
  }

  for (let i = 0; i < n; i++) {
    ideas[i].group = groupRemap[groupAssign[i]];
    const sr = subRemap[subAssign[i]];
    ideas[i].subgroup = `${sr.group}-${sr.sub}`;
  }

  console.log('Groups:', uniqueGroups.length, 'Subgroups:', subId);
  return { numGroups: uniqueGroups.length, numSubgroups: subId };
}

// ── Cluster Centers ────────────────────────────────────
function computeClusterCenters(numGroups) {
  if (numGroups <= 1) {
    return { 0: { x: width / 2, y: height / 2 } };
  }
  const radius = Math.min(width, height) * 0.28;
  const centers = {};
  for (let i = 0; i < numGroups; i++) {
    const angle = (2 * Math.PI * i) / numGroups - Math.PI / 2;
    centers[i] = {
      x: width / 2 + radius * Math.cos(angle),
      y: height / 2 + radius * Math.sin(angle),
    };
  }
  return centers;
}

function computeSubclusterCenters(numGroups) {
  // Offset subclusters slightly from their parent group center
  const centers = {};
  const subsByGroup = {};
  for (const idea of ideas) {
    if (!subsByGroup[idea.group]) subsByGroup[idea.group] = new Set();
    subsByGroup[idea.group].add(idea.subgroup);
  }
  for (let g = 0; g < numGroups; g++) {
    const subs = [...(subsByGroup[g] || [])];
    const parent = clusterCenters[g] || { x: width / 2, y: height / 2 };
    if (subs.length <= 1) {
      if (subs[0]) centers[subs[0]] = { x: parent.x, y: parent.y };
      continue;
    }
    const subRadius = 50 + subs.length * 10;
    subs.forEach((s, i) => {
      const angle = (2 * Math.PI * i) / subs.length;
      centers[s] = {
        x: parent.x + subRadius * Math.cos(angle),
        y: parent.y + subRadius * Math.sin(angle),
      };
    });
  }
  return centers;
}

// ── Cluster Labels ─────────────────────────────────────
const STOP_WORDS = new Set([
  'a','an','the','and','or','but','in','on','at','to','for','of','with',
  'by','from','is','it','its','are','was','were','be','been','being',
  'have','has','had','do','does','did','will','would','could','should',
  'may','might','can','this','that','these','those','i','me','my','we',
  'our','you','your','he','she','they','them','their','what','which',
  'who','how','when','where','why','not','no','so','if','then','than',
  'very','just','about','up','out','into','over','after','before','between',
  'all','each','every','both','few','more','most','other','some','such',
  'as','also','back','even','still','new','old','get','go','make','like',
]);

function generateLabel(members) {
  if (members.length === 0) return '';
  if (members.length === 1) return members[0].text;

  const wordFreq = {};
  for (const idea of members) {
    const words = idea.text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/);
    const seen = new Set();
    for (const w of words) {
      if (w.length < 2 || STOP_WORDS.has(w) || seen.has(w)) continue;
      seen.add(w);
      wordFreq[w] = (wordFreq[w] || 0) + 1;
    }
  }

  const shared = Object.entries(wordFreq)
    .filter(([, count]) => count > 1)
    .sort((a, b) => b[1] - a[1])
    .map(([w]) => w);

  if (shared.length > 0) {
    return shared
      .slice(0, 3)
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(' & ');
  }

  // Fallback: most central idea
  let bestIdx = 0, bestAvg = -Infinity;
  for (let i = 0; i < members.length; i++) {
    let sum = 0;
    for (let j = 0; j < members.length; j++) {
      if (i !== j) sum += cosineSimilarity(members[i].embedding, members[j].embedding);
    }
    const avg = sum / (members.length - 1);
    if (avg > bestAvg) { bestAvg = avg; bestIdx = i; }
  }
  const label = members[bestIdx].text;
  return label.length > 40 ? label.slice(0, 39) + '…' : label;
}

// ── Links ──────────────────────────────────────────────
function computeLinks() {
  const links = [];
  for (let i = 0; i < ideas.length; i++) {
    for (let j = i + 1; j < ideas.length; j++) {
      if (ideas[i].subgroup === ideas[j].subgroup) {
        // Same subgroup — strong link
        links.push({ source: ideas[i], target: ideas[j], type: 'sub' });
      } else if (ideas[i].group === ideas[j].group) {
        // Same group, different subgroup — weak link
        links.push({ source: ideas[i], target: ideas[j], type: 'group' });
      }
    }
  }
  return links;
}

// ── D3 Simulation ──────────────────────────────────────
function updateClusterForces() {
  if (!simulation) return;
  // Two-level pull: weak toward group center, stronger toward subcluster center
  simulation
    .force('groupX', d3.forceX((d) => clusterCenters[d.group]?.x ?? width / 2).strength(0.06))
    .force('groupY', d3.forceY((d) => clusterCenters[d.group]?.y ?? height / 2).strength(0.06))
    .force('subX', d3.forceX((d) => subclusterCenters[d.subgroup]?.x ?? width / 2).strength(0.12))
    .force('subY', d3.forceY((d) => subclusterCenters[d.subgroup]?.y ?? height / 2).strength(0.12));
}

let clusterLabels = [];

function updateSimulation() {
  const { numGroups } = clusterIdeas();
  clusterCenters = computeClusterCenters(numGroups);
  subclusterCenters = computeSubclusterCenters(numGroups);
  const links = computeLinks();

  // Generate labels for groups (2+ members)
  clusterLabels = [];
  for (let g = 0; g < numGroups; g++) {
    const members = ideas.filter((d) => d.group === g);
    if (members.length >= 2) {
      clusterLabels.push({
        id: 'g-' + g,
        label: generateLabel(members),
        group: g,
        type: 'group',
      });
    }
    // Subgroup labels — shown when a group splits into multiple subgroups
    const subs = [...new Set(members.map((d) => d.subgroup))];
    if (subs.length > 1) {
      for (const s of subs) {
        const subMembers = members.filter((d) => d.subgroup === s);
        clusterLabels.push({
          id: 's-' + s,
          label: generateLabel(subMembers),
          group: g,
          subgroup: s,
          type: 'subgroup',
        });
      }
    }
  }

  if (!simulation) {
    simulation = d3.forceSimulation(ideas)
      .velocityDecay(0.55)
      .alphaDecay(0.015)
      .force('charge', d3.forceManyBody().strength(-80))
      .force('collide', d3.forceCollide((d) => d.radius + 4).strength(0.7))
      .force('link', d3.forceLink(links.filter((l) => l.type === 'sub')).distance(40).strength(0.12))
      .force('groupLink', d3.forceLink(links.filter((l) => l.type === 'group')).distance(90).strength(0.03))
      .force('centerX', d3.forceX(width / 2).strength(0.01))
      .force('centerY', d3.forceY(height / 2).strength(0.01))
      .on('tick', ticked);
    updateClusterForces();
  } else {
    simulation.nodes(ideas);
    simulation.force('link', d3.forceLink(links.filter((l) => l.type === 'sub')).distance(40).strength(0.12));
    simulation.force('groupLink', d3.forceLink(links.filter((l) => l.type === 'group')).distance(90).strength(0.03));
    simulation.force('collide', d3.forceCollide((d) => d.radius + 4).strength(0.7));
    updateClusterForces();
    simulation.alpha(0.8).restart();
  }

  render(links);
}

// ── Rendering ──────────────────────────────────────────
const colorScale = (group) => PALETTE[group % PALETTE.length];

function subgroupColor(group, subgroup) {
  const base = d3.color(colorScale(group));
  // Shift lightness per subgroup to differentiate within a group
  const subIdx = parseInt(subgroup.split('-')[1], 10) || 0;
  const offset = (subIdx % 3) * 0.15 - 0.15; // -0.15, 0, +0.15
  return base.brighter(offset).toString();
}

function truncate(text, max) {
  return text.length > max ? text.slice(0, max - 1) + '…' : text;
}

function wrapText(text, maxChars) {
  if (text.length <= maxChars) return [text];
  const words = text.split(' ');
  const lines = [];
  let current = '';
  for (const word of words) {
    if ((current + ' ' + word).trim().length > maxChars && current) {
      lines.push(current);
      current = word;
    } else {
      current = current ? current + ' ' + word : word;
    }
    if (lines.length >= 2) {
      current = truncate(current, maxChars);
      break;
    }
  }
  if (current) lines.push(current);
  return lines;
}

function render(links) {
  // Links — clear and redraw
  linkGroup.selectAll('line').remove();
  for (const link of links) {
    linkGroup.append('line')
      .datum(link)
      .attr('class', link.type === 'sub' ? 'subgroup-link' : 'group-link');
  }

  // Bubbles
  const groups = nodeGroup.selectAll('g.bubble').data(ideas, (d) => d.id);
  groups.exit().transition().duration(300).attr('opacity', 0).remove();

  const enter = groups.enter().append('g').attr('class', 'bubble');

  enter
    .append('circle')
    .attr('r', 0)
    .attr('fill', (d) => subgroupColor(d.group, d.subgroup))
    .attr('opacity', 0.85)
    .transition()
    .duration(600)
    .attr('r', (d) => d.radius);

  enter
    .append('text')
    .attr('class', 'bubble-label')
    .attr('text-anchor', 'middle')
    .attr('dy', '0.35em');

  // Delete button — hidden by default, shown on hover
  const delGroup = enter.append('g')
    .attr('class', 'delete-btn')
    .attr('opacity', 0)
    .style('cursor', 'pointer');

  delGroup.append('circle')
    .attr('r', 10)
    .attr('fill', '#ef4444')
    .attr('stroke', '#fff')
    .attr('stroke-width', 1.5);

  delGroup.append('text')
    .attr('text-anchor', 'middle')
    .attr('dy', '0.35em')
    .attr('font-size', '12px')
    .attr('font-weight', '700')
    .attr('fill', '#fff')
    .attr('pointer-events', 'none')
    .text('\u00d7');

  // Position delete button at top-right of bubble
  delGroup.attr('transform', (d) => `translate(${d.radius * 0.7}, ${-d.radius * 0.7})`);

  delGroup.on('click', (event, d) => {
    event.stopPropagation();
    deleteIdea(d.id);
  });

  enter
    .on('mouseenter', function (event, d) {
      tooltip.textContent = d.text;
      tooltip.style.opacity = '1';
      d3.select(this).select('.delete-btn')
        .transition().duration(150).attr('opacity', 1);
    })
    .on('mousemove', (event) => {
      const rect = svgEl.getBoundingClientRect();
      tooltip.style.left = event.clientX - rect.left + 12 + 'px';
      tooltip.style.top = event.clientY - rect.top - 8 + 'px';
    })
    .on('mouseleave', function () {
      tooltip.style.opacity = '0';
      d3.select(this).select('.delete-btn')
        .transition().duration(150).attr('opacity', 0);
    });

  enter.call(drag());

  const merged = enter.merge(groups);
  merged
    .select('circle')
    .transition()
    .duration(600)
    .attr('fill', (d) => subgroupColor(d.group, d.subgroup))
    .attr('r', (d) => d.radius);

  merged.select('text').each(function (d) {
    const lines = wrapText(d.text, Math.floor(d.radius / 3.5));
    const el = d3.select(this);
    el.selectAll('tspan').remove();
    lines.forEach((line, i) => {
      el.append('tspan')
        .attr('x', 0)
        .attr('dy', i === 0 ? `${-(lines.length - 1) * 0.3}em` : '1.1em')
        .text(line);
    });
  });

  // Labels
  const labelSel = labelGroup.selectAll('text.cluster-label').data(clusterLabels, (d) => d.id);
  labelSel.exit().transition().duration(300).attr('opacity', 0).remove();
  const labelEnter = labelSel.enter().append('text')
    .attr('class', 'cluster-label')
    .attr('text-anchor', 'middle')
    .attr('opacity', 0);
  labelEnter.transition().duration(600).attr('opacity', (d) => d.type === 'group' ? 0.8 : 0.55);
  const labelMerged = labelEnter.merge(labelSel);
  labelMerged
    .text((d) => d.label)
    .attr('fill', (d) => colorScale(d.group))
    .attr('font-size', (d) => d.type === 'group' ? '14px' : '11px')
    .attr('font-weight', (d) => d.type === 'group' ? '700' : '500')
    .attr('letter-spacing', (d) => d.type === 'group' ? '0.5px' : '0.3px');
}

// ── Convex Hulls for Groups ────────────────────────────
function drawHulls(numGroups) {
  hullGroup.selectAll('path.group-hull').remove();
  for (let g = 0; g < numGroups; g++) {
    const members = ideas.filter((d) => d.group === g);
    if (members.length < 2) continue;
    const points = members.map((d) => [d.x, d.y]);
    // Need at least 3 points for a hull; for 2, draw nothing (links show it)
    if (points.length < 3) continue;
    const hull = d3.polygonHull(points);
    if (!hull) continue;
    hullGroup.append('path')
      .attr('class', 'group-hull')
      .attr('d', 'M' + hull.map((p) => p.join(',')).join('L') + 'Z')
      .attr('fill', colorScale(g))
      .attr('fill-opacity', 0.05)
      .attr('stroke', colorScale(g))
      .attr('stroke-opacity', 0.12)
      .attr('stroke-width', 1.5)
      .attr('rx', 20);
  }
}

function ticked() {
  nodeGroup
    .selectAll('g.bubble')
    .attr('transform', (d) => `translate(${d.x},${d.y})`);

  linkGroup
    .selectAll('line')
    .attr('x1', (d) => d.source.x)
    .attr('y1', (d) => d.source.y)
    .attr('x2', (d) => d.target.x)
    .attr('y2', (d) => d.target.y);

  // Hulls around top-level groups
  const numGroups = new Set(ideas.map((d) => d.group)).size;
  drawHulls(numGroups);

  // Position labels
  labelGroup.selectAll('text.cluster-label').each(function (d) {
    let members;
    if (d.type === 'group') {
      members = ideas.filter((idea) => idea.group === d.group);
    } else {
      members = ideas.filter((idea) => idea.subgroup === d.subgroup);
    }
    if (members.length === 0) return;
    let minY = Infinity, sumX = 0;
    for (const m of members) {
      if (m.y - m.radius < minY) minY = m.y - m.radius;
      sumX += m.x;
    }
    const offset = d.type === 'group' ? 28 : 14;
    d3.select(this)
      .attr('x', sumX / members.length)
      .attr('y', minY - offset);
  });
}

// ── Drag Behavior ──────────────────────────────────────
function drag() {
  return d3
    .drag()
    .on('start', function (event, d) {
      if (!event.active) simulation.alphaTarget(0.15).restart();
      d.fx = d.x;
      d.fy = d.y;
      d3.select(this).classed('dragging', true);
    })
    .on('drag', (event, d) => {
      d.fx = event.x;
      d.fy = event.y;
    })
    .on('end', function (event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
      d3.select(this).classed('dragging', false);
    });
}

// ── UI ─────────────────────────────────────────────────
function setStatus(msg) {
  statusEl.textContent = msg;
}

function deleteIdea(id) {
  ideas = ideas.filter((d) => d.id !== id);
  tooltip.style.opacity = '0';
  if (ideas.length === 0) {
    emptyState.style.display = '';
    // Clear everything
    nodeGroup.selectAll('g.bubble').remove();
    linkGroup.selectAll('line').remove();
    labelGroup.selectAll('text.cluster-label').remove();
    hullGroup.selectAll('path.group-hull').remove();
    if (simulation) { simulation.stop(); simulation = null; }
    setStatus('Type an idea to begin');
    return;
  }
  setStatus(`${ideas.length} idea${ideas.length > 1 ? 's' : ''}`);
  updateSimulation();
}

async function submitIdea() {
  const text = inputEl.value.trim();
  if (!text) return;

  inputEl.disabled = true;
  submitBtn.disabled = true;
  setStatus('Computing embedding...');

  try {
    const ext = await getExtractor();
    const output = await ext(text, { pooling: 'mean', normalize: true });
    const embedding = Array.from(output.data);

    ideas.push({
      id: Date.now() + Math.random(),
      text,
      embedding,
      group: 0,
      subgroup: '0-0',
      radius: 28 + Math.min(text.length / 8, 18),
      x: width / 2 + (Math.random() - 0.5) * 80,
      y: height / 2 + (Math.random() - 0.5) * 80,
    });

    emptyState.style.display = 'none';
    inputEl.value = '';
    setStatus(`${ideas.length} idea${ideas.length > 1 ? 's' : ''}`);
    updateSimulation();
  } catch (err) {
    console.error(err);
    setStatus('Error: ' + err.message);
  } finally {
    inputEl.disabled = false;
    submitBtn.disabled = false;
    inputEl.focus();
  }
}

submitBtn.addEventListener('click', submitIdea);
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') submitIdea();
});

// ── Threshold Sliders ──────────────────────────────────
const groupSlider = document.getElementById('group-slider');
const subSlider = document.getElementById('sub-slider');
const groupVal = document.getElementById('group-val');
const subVal = document.getElementById('sub-val');

groupSlider.addEventListener('input', () => {
  GROUP_THRESHOLD = parseFloat(groupSlider.value);
  groupVal.textContent = GROUP_THRESHOLD.toFixed(2);
  // Ensure subgroup is always >= group
  if (SUBGROUP_THRESHOLD < GROUP_THRESHOLD) {
    SUBGROUP_THRESHOLD = GROUP_THRESHOLD;
    subSlider.value = SUBGROUP_THRESHOLD;
    subVal.textContent = SUBGROUP_THRESHOLD.toFixed(2);
  }
  if (ideas.length >= 2) updateSimulation();
});

subSlider.addEventListener('input', () => {
  SUBGROUP_THRESHOLD = parseFloat(subSlider.value);
  subVal.textContent = SUBGROUP_THRESHOLD.toFixed(2);
  // Ensure group is always <= subgroup
  if (GROUP_THRESHOLD > SUBGROUP_THRESHOLD) {
    GROUP_THRESHOLD = SUBGROUP_THRESHOLD;
    groupSlider.value = GROUP_THRESHOLD;
    groupVal.textContent = GROUP_THRESHOLD.toFixed(2);
  }
  if (ideas.length >= 2) updateSimulation();
});
