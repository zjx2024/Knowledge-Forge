import {Graph} from '@cosmos.gl/graph';

// Muted categorical hues are deliberately limited.  They are assigned from a
// semantic type/community field, never from an entity ID, so the same category
// remains recognisable without turning the graph into a colour-coded rainbow.
const CATEGORY_PALETTE = [
  '#5f82a3', '#678f98', '#788fa7', '#8d8ea0', '#80978d', '#9a8d86',
];
const FALLBACK_BLUE = '#6e91b2';
const ACCENT_ORANGE = '#df8646';
const BASE_LINK = '#7891a7';
const EMPHASIZED_LINK = '#286da8';

function hash(value) {
  let result = 2166136261;
  for (const character of String(value ?? '')) {
    result ^= character.codePointAt(0);
    result = Math.imul(result, 16777619);
  }
  return result >>> 0;
}

function rgb(value, alpha = 1) {
  const text = String(value || '#2878d0').replace('#', '');
  const normalized = text.length === 3
    ? text.split('').map(character => character + character).join('')
    : text.padEnd(6, '0').slice(0, 6);
  return [
    parseInt(normalized.slice(0, 2), 16) / 255,
    parseInt(normalized.slice(2, 4), 16) / 255,
    parseInt(normalized.slice(4, 6), 16) / 255,
    alpha,
  ];
}

function short(value, limit = 34) {
  const text = String(value ?? '').replace(/\s+/g, ' ').trim();
  return text.length > limit ? `${text.slice(0, Math.max(1, limit - 1))}…` : text;
}

function finite(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function quantile(values, position) {
  if (!values.length) return 0;
  const index = Math.max(0, Math.min(values.length - 1, (values.length - 1) * position));
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  return values[lower] + (values[upper] - values[lower]) * (index - lower);
}

function continuousSizeScale(values, minimum = 16, maximum = 42) {
  const logged = values.map(value => Math.log1p(Math.max(0, finite(value, 1)))).sort((left, right) => left - right);
  const lower = quantile(logged, 0.1);
  const upper = Math.max(lower + 0.01, quantile(logged, 0.9));
  return value => {
    const normalized = Math.max(0, Math.min(1, (Math.log1p(Math.max(0, finite(value, 1))) - lower) / (upper - lower)));
    // This keeps low-degree peripheral nodes legible while reserving the upper
    // part of the 16–42px range for genuinely central nodes.
    return minimum + (maximum - minimum) * Math.pow(normalized, 0.76);
  };
}

function semanticCategory(node) {
  // Entity type is a real logical category.  In a community overview every
  // node has the same type ("社区"), which intentionally yields one calm blue
  // family instead of assigning arbitrary colours to community IDs.
  return String(node.category ?? node.cluster ?? node.type ?? node.community ?? node.group ?? 'entity');
}

function categoryColor(category) {
  if (!category || category === 'undefined' || category === 'null') return FALLBACK_BLUE;
  return CATEGORY_PALETTE[hash(category) % CATEGORY_PALETTE.length];
}

function blueByCentrality(value) {
  // Used when there is no meaningful category distinction in the supplied
  // data (for example, an overview made entirely of communities).  The
  // continuous blue ramp still makes central nodes legible without inventing
  // colour meaning from IDs.
  const ratio = Math.max(0, Math.min(1, finite(value)));
  const peripheral = [0.54, 0.67, 0.78];
  const core = [0.21, 0.43, 0.64];
  return peripheral.map((channel, index) => channel + (core[index] - channel) * ratio).concat(1);
}

function rankNodes(nodes) {
  return nodes.map((node, index) => ({
    index,
    score: finite(node.degree ?? node.size ?? node.member_count, 0),
    title: String(node.title ?? node.label ?? node.id ?? ''),
  })).sort((left, right) =>
    right.score - left.score || left.title.localeCompare(right.title, 'zh-Hans-CN') || left.index - right.index
  );
}

function overviewPosition(index, count, importance) {
  // A centred sunflower distribution is only the deterministic starting point.
  // The bounded topology relaxation below moves connected communities together,
  // so the final overview remains a graph rather than an even radial pattern.
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const normalized = Math.sqrt((index + 0.55) / Math.max(1, count));
  const radius = 150 + normalized * (540 + Math.min(110, Math.log1p(importance) * 22));
  const angle = index * goldenAngle;
  return [Math.cos(angle) * radius, Math.sin(angle) * radius];
}

function seededPosition(node, index, count, cluster, clusterCount, compact = false) {
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const localIndex = Math.max(0, index);
  const localRadius = 72 + Math.sqrt(localIndex + 1) * (compact ? 58 : 44);
  const localAngle = goldenAngle * localIndex + (hash(node.id) % 360) * Math.PI / 1800;
  const clusterAngle = (cluster / Math.max(1, clusterCount)) * Math.PI * 2;
  const clusterRadius = clusterCount > 1
    ? Math.min(compact ? 720 : 580, 220 + clusterCount * (compact ? 11 : 8))
    : 0;
  return [
    Math.cos(clusterAngle) * clusterRadius + Math.cos(localAngle) * localRadius,
    Math.sin(clusterAngle) * clusterRadius + Math.sin(localAngle) * localRadius,
  ];
}

function relaxPositions(positions, edges, nodeIndex, {compact = false, pointSizes} = {}) {
  // A bounded deterministic layout keeps the same scene understandable after
  // refreshes.  It also avoids relying on an unbounded live force simulation
  // to rescue a large, dense graph after it has already been drawn.
  const nodeCount = positions.length / 2;
  if (nodeCount < 2) return positions;
  const x = new Float64Array(nodeCount);
  const y = new Float64Array(nodeCount);
  for (let index = 0; index < nodeCount; index += 1) {
    x[index] = positions[index * 2];
    y[index] = positions[index * 2 + 1];
  }
  const usableEdges = edges.map(edge => [
    nodeIndex.get(String(edge.source)), nodeIndex.get(String(edge.target)),
  ]).filter(([source, target]) => source !== undefined && target !== undefined && source !== target);
  const iterations = nodeCount <= 50 ? 220 : nodeCount <= 120 ? 165 : nodeCount <= 260 ? 90 : 28;
  const repulsion = compact ? 104000 : 148000;
  const idealLength = compact ? 196 : 286;
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const forceX = new Float64Array(nodeCount);
    const forceY = new Float64Array(nodeCount);
    for (let left = 0; left < nodeCount; left += 1) {
      for (let right = left + 1; right < nodeCount; right += 1) {
        let dx = x[left] - x[right];
        let dy = y[left] - y[right];
        let distanceSquared = dx * dx + dy * dy;
        if (distanceSquared < 1) {
          const angle = ((hash(`${left}:${right}`) % 6283) / 1000);
          dx = Math.cos(angle);
          dy = Math.sin(angle);
          distanceSquared = 1;
        }
        const distance = Math.sqrt(distanceSquared);
        const radius = (finite(pointSizes?.[left], 16) + finite(pointSizes?.[right], 16)) / 2;
        const minimumDistance = radius + (compact ? 30 : 40);
        const collision = distance < minimumDistance ? (minimumDistance - distance) * 1.16 : 0;
        const magnitude = repulsion / (distanceSquared + 2600) + collision;
        const fx = (dx / distance) * magnitude;
        const fy = (dy / distance) * magnitude;
        forceX[left] += fx; forceY[left] += fy;
        forceX[right] -= fx; forceY[right] -= fy;
      }
    }
    for (const [source, target] of usableEdges) {
      const dx = x[target] - x[source];
      const dy = y[target] - y[source];
      const distance = Math.max(1, Math.hypot(dx, dy));
      const sourceRadius = finite(pointSizes?.[source], 16) / 2;
      const targetRadius = finite(pointSizes?.[target], 16) / 2;
      const targetLength = idealLength + sourceRadius + targetRadius;
      const magnitude = (distance - targetLength) * 0.029;
      const fx = (dx / distance) * magnitude;
      const fy = (dy / distance) * magnitude;
      forceX[source] += fx; forceY[source] += fy;
      forceX[target] -= fx; forceY[target] -= fy;
    }
    const maximumStep = 20 - (15 * iteration / Math.max(1, iterations - 1));
    for (let index = 0; index < nodeCount; index += 1) {
      // A small central pull prevents disconnected components drifting to a
      // distant corner while preserving their internal topology.
      forceX[index] -= x[index] * 0.0026;
      forceY[index] -= y[index] * 0.0026;
      const magnitude = Math.hypot(forceX[index], forceY[index]);
      const scale = magnitude > maximumStep ? maximumStep / magnitude : 1;
      x[index] += forceX[index] * scale;
      y[index] += forceY[index] * scale;
    }
  }
  for (let index = 0; index < nodeCount; index += 1) {
    positions[index * 2] = x[index];
    positions[index * 2 + 1] = y[index];
  }
  return positions;
}

/**
 * Commercial-safe graph renderer for Knowledge Forge.
 *
 * The class intentionally owns only rendering and interaction. Fetching,
 * evidence display, community drill-down and strategy semantics remain in the
 * application adapter so both graph strategies can share the same renderer.
 */
export class GraphExplorer {
  constructor(container, options = {}) {
    if (!(container instanceof HTMLElement)) throw new TypeError('GraphExplorer requires a container');
    this.container = container;
    this.options = options;
    this.nodes = [];
    this.edges = [];
    this.nodeIndex = new Map();
    this.selectedNode = -1;
    this.selectedEdge = -1;
    this.hoveredNode = -1;
    this.hoveredEdge = -1;
    this.destroyed = false;
    this.labelFrame = 0;
    this.dataRevision = 0;
    this.fitZoomLevel = null;
    this.pointSizes = new Float32Array();
    this.defaultPointColors = new Float32Array();
    this.defaultLinkColors = new Float32Array();
    this.defaultLinkWidths = new Float32Array();
    this.defaultNodes = document.createElement('div');
    this.defaultNodes.className = 'kf-gpu-default-nodes';
    this.defaultEdges = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    this.defaultEdges.setAttribute('class', 'kf-gpu-default-edges');
    this.defaultEdges.setAttribute('width', '100%');
    this.defaultEdges.setAttribute('height', '100%');
    this.defaultEdges.setAttribute('aria-hidden', 'true');
    const defaultDefinitions = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    const defaultMarker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
    defaultMarker.setAttribute('id', 'kf-default-edge-arrow');
    defaultMarker.setAttribute('viewBox', '0 0 8 8');
    defaultMarker.setAttribute('refX', '7');
    defaultMarker.setAttribute('refY', '4');
    defaultMarker.setAttribute('markerWidth', '4');
    defaultMarker.setAttribute('markerHeight', '4');
    defaultMarker.setAttribute('orient', 'auto');
    defaultMarker.setAttribute('markerUnits', 'strokeWidth');
    const defaultArrow = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    defaultArrow.setAttribute('d', 'M 0 0 L 8 4 L 0 8 z');
    defaultArrow.setAttribute('fill', BASE_LINK);
    defaultMarker.append(defaultArrow);
    defaultDefinitions.append(defaultMarker);
    this.defaultEdges.append(defaultDefinitions);
    this.allNodeIndices = [];
    this.lastPointerSelection = null;
    this.mode = 'relations';
    this.labelCandidates = [];
    this.container.classList.add('kf-gpu-graph');
    this.container.replaceChildren();
    this.surface = document.createElement('div');
    this.surface.className = 'kf-gpu-graph-surface';
    this.emphasizedEdges = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    this.emphasizedEdges.setAttribute('class', 'kf-gpu-emphasized-edges');
    this.emphasizedEdges.setAttribute('width', '100%');
    this.emphasizedEdges.setAttribute('height', '100%');
    this.emphasizedEdges.setAttribute('aria-hidden', 'true');
    const definitions = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
    marker.setAttribute('id', 'kf-emphasized-edge-arrow');
    marker.setAttribute('viewBox', '0 0 8 8');
    marker.setAttribute('refX', '7');
    marker.setAttribute('refY', '4');
    marker.setAttribute('markerWidth', '5');
    marker.setAttribute('markerHeight', '5');
    marker.setAttribute('orient', 'auto');
    marker.setAttribute('markerUnits', 'strokeWidth');
    const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    arrow.setAttribute('d', 'M 0 0 L 8 4 L 0 8 z');
    arrow.setAttribute('fill', EMPHASIZED_LINK);
    marker.append(arrow);
    definitions.append(marker);
    this.emphasizedEdges.append(definitions);
    this.nodeRings = document.createElement('div');
    this.nodeRings.className = 'kf-gpu-node-rings';
    this.emphasizedNodes = document.createElement('div');
    this.emphasizedNodes.className = 'kf-gpu-emphasized-nodes';
    this.edgeLabels = document.createElement('div');
    this.edgeLabels.className = 'kf-gpu-graph-edge-labels';
    this.labels = document.createElement('div');
    this.labels.className = 'kf-gpu-graph-labels';
    this.tooltip = document.createElement('div');
    this.tooltip.className = 'kf-gpu-graph-tooltip';
    this.tooltip.hidden = true;
    this.container.append(this.surface, this.defaultEdges, this.defaultNodes, this.emphasizedEdges, this.emphasizedNodes, this.nodeRings, this.edgeLabels, this.labels, this.tooltip);
    this.pointerDown = null;
    this.skipClickAfterPan = false;
    // Keep selection independent from asynchronous GPU picking.  The click
    // itself is blocked below; selection happens on pointer-up, before Cosmos
    // can turn a valid user selection into a background click on a stale pick.
    this.surface.addEventListener('click', event => {
      event.stopImmediatePropagation();
    }, {capture: true});
    // Cosmos binds the wheel to its pan/zoom controller. Intercept it before
    // that controller: plain wheel scrolls the nearest page viewport, while
    // Ctrl + wheel is reserved for graph zoom. Drag panning remains native.
    this.surface.addEventListener('wheel', event => {
      event.preventDefault();
      event.stopImmediatePropagation();
      if (event.ctrlKey) {
        const baseline = this.currentFitZoomLevel();
        const next = this.graph.getZoomLevel() * Math.exp(-finite(event.deltaY) * 0.00145);
        this.graph.setZoomLevel(Math.max(baseline * 0.25, Math.min(baseline * 8, next)), 0, false);
      } else {
        this.scrollDocumentViewport(event.deltaY, event.deltaX);
      }
    }, {passive: false, capture: true});
    this.surface.addEventListener('pointerdown', event => {
      if (event.button !== 0) return;
      this.surface.classList.add('is-panning');
      this.pointerDown = {
        x: event.clientX,
        y: event.clientY,
        moved: false,
        // Ctrl is reserved for modified canvas gestures (including zoom with
        // the wheel).  It must never turn a pan into a background selection.
        preserveSelection: event.ctrlKey,
      };
      // Pointer capture may be unavailable for synthetic browser events.  The
      // cursor state must not depend on it; capture only keeps a real drag
      // active when it leaves the canvas bounds.
      try { this.surface.setPointerCapture?.(event.pointerId); } catch (_) { /* browser supplies capture for real pointers */ }
    });
    this.surface.addEventListener('pointermove', event => {
      if (!this.pointerDown) return;
      if (Math.hypot(event.clientX - this.pointerDown.x, event.clientY - this.pointerDown.y) > 4) {
        this.pointerDown.moved = true;
      }
    });
    const stopPanning = event => {
      const pointerDown = this.pointerDown;
      this.surface.classList.remove('is-panning');
      this.skipClickAfterPan = Boolean(
        pointerDown?.moved || pointerDown?.preserveSelection || event?.ctrlKey
      );
      this.pointerDown = null;
      if (event?.type === 'pointerup' && event.button === 0 && !this.skipClickAfterPan) {
        // The following click event is deliberately suppressed above.  This
        // one deterministic coordinate hit test is the sole selection path.
        this.selectAtClientPoint(event.clientX, event.clientY);
      }
    };
    this.surface.addEventListener('pointerup', stopPanning, {capture: true});
    this.surface.addEventListener('pointercancel', stopPanning, {capture: true});
    this.surface.addEventListener('pointerleave', event => {
      if (event.buttons === 0) stopPanning();
    });
    this.resizeObserver = new ResizeObserver(() => this.renderLabels());
    this.resizeObserver.observe(this.container);
    this.graph = new Graph(this.surface, {
      backgroundColor: '#f6f8fb',
      // The library enforces a 0.001 minimum zoom.  With the old 8192 world
      // and a sparse overview, fitting could land exactly on that floor,
      // collapsing all visual nodes into sub-pixels while HTML labels remained
      // visible.  A 4096 world keeps the same layout resolution but permits a
      // truthful fitted view for both sparse overviews and dense communities.
      spaceSize: 4096,
      randomSeed: String(options.seed || 'knowledge-forge'),
      attribution: '',
      enableSimulation: true,
      // Switching between overview, community and entity views is a data
      // replacement, not an animation.  A transition can keep positions from
      // the previous view alive in the GPU texture after the SVG layer has
      // moved to the new one, which reads as a persistent ghost graph.
      transitionDuration: 0,
      enableSimulationDuringZoom: false,
      enableZoom: true,
      // The primary gesture is viewport panning.  Nodes remain clickable but
      // cannot be accidentally repositioned by a drag gesture.
      enableDrag: false,
      // setData owns fitting after positions are available.  Leaving Cosmos'
      // delayed initial fit enabled races that fit and can restore a stale
      // minimum-scale transform for sparse community overviews.
      fitViewOnInit: false,
      fitViewPadding: 0.16,
      pointDefaultSize: 16,
      pointSizeScale: 1,
      pointOpacity: 0.94,
      pointGreyoutOpacity: 0.28,
      renderHoveredPointRing: true,
      hoveredPointRingColor: ACCENT_ORANGE,
      focusedPointRingColor: ACCENT_ORANGE,
      outlinedPointRingColor: '#ffffff',
      hoveredPointCursor: 'pointer',
      hoveredLinkCursor: 'pointer',
      linkDefaultColor: BASE_LINK,
      linkDefaultWidth: 1.15,
      linkDefaultArrows: true,
      linkArrowsSizeScale: 1.15,
      linkOpacity: 0.32,
      linkGreyoutOpacity: 0.09,
      linkVisibilityDistanceRange: [1150, 1800],
      linkVisibilityMinTransparency: 1,
      hoveredLinkColor: EMPHASIZED_LINK,
      // 1.0–1.3px defaults become about 2.4px when inspected, without
      // turning an active neighbourhood into heavy black strokes.
      hoveredLinkWidthIncrease: 1.35,
      focusedLinkWidthIncrease: 1.35,
      curvedLinks: true,
      curvedLinkSegments: 12,
      curvedLinkWeight: 0.55,
      curvedLinkControlPointDistance: 0.12,
      simulationDecay: 980,
      simulationGravity: 0.09,
      simulationRepulsion: 0.9,
      simulationLinkSpring: 0.64,
      simulationLinkDistance: 96,
      simulationCollision: 1,
      simulationCollisionPadding: 10,
      pointSamplingDistance: 105,
      scalePointsOnZoom: false,
      scaleLinksOnZoom: false,
      onPointClick: index => this.selectNode(index, true),
      onLinkClick: index => this.selectEdge(index, true),
      onBackgroundClick: () => this.clearSelection(true),
      onPointMouseOver: (index, _position, event) => this.hoverNode(index, event),
      onPointMouseOut: () => this.unhoverNode(),
      onLinkMouseOver: index => this.hoverEdge(index),
      onLinkMouseOut: () => this.unhoverEdge(),
      onZoom: () => {
        this.options.onZoomChange?.(this.zoomPercent());
        this.scheduleLabels();
      },
      onZoomEnd: () => {
        this.options.onZoomChange?.(this.zoomPercent());
        this.renderLabels();
      },
      onDrag: () => this.scheduleLabels(),
      onDragEnd: () => this.renderLabels(),
      onSimulationTick: () => this.scheduleLabels(),
      onSimulationEnd: () => this.renderLabels(),
    });
  }

  scrollDocumentViewport(deltaY, deltaX) {
    let viewport = this.container.parentElement;
    while (viewport) {
      const style = window.getComputedStyle(viewport);
      const canScrollY = /(auto|scroll)/.test(style.overflowY)
        && viewport.scrollHeight > viewport.clientHeight + 1;
      const canScrollX = /(auto|scroll)/.test(style.overflowX)
        && viewport.scrollWidth > viewport.clientWidth + 1;
      if (canScrollY || canScrollX) {
        viewport.scrollBy({top: finite(deltaY), left: finite(deltaX), behavior: 'auto'});
        return;
      }
      viewport = viewport.parentElement;
    }
    window.scrollBy({top: finite(deltaY), left: finite(deltaX), behavior: 'auto'});
  }

  async setData(payload = {}) {
    const revision = ++this.dataRevision;
    if (this.labelFrame) {
      cancelAnimationFrame(this.labelFrame);
      this.labelFrame = 0;
    }
    this.clearVisibleTopology();
    this.mode = String(payload.mode || 'relations');
    this.fitZoomLevel = null;
    this.nodes = Array.isArray(payload.nodes) ? payload.nodes : [];
    const rawEdges = Array.isArray(payload.edges) ? payload.edges : [];
    this.nodeIndex = new Map(this.nodes.map((node, index) => [String(node.id), index]));
    this.allNodeIndices = this.nodes.map((_node, index) => index);
    this.edges = rawEdges.filter(edge =>
      this.nodeIndex.has(String(edge.source)) && this.nodeIndex.has(String(edge.target))
    );
    this.selectedNode = -1;
    this.selectedEdge = -1;

    const positions = new Float32Array(this.nodes.length * 2);
    const colors = new Float32Array(this.nodes.length * 4);
    const sizes = new Float32Array(this.nodes.length);
    const overviewLayout = this.mode === 'overview';
    const compactLayout = !overviewLayout;
    const importanceFor = node => finite(
      overviewLayout
        ? node.member_count ?? node.size ?? node.degree
        : node.degree ?? node.size ?? node.member_count,
      1,
    );
    const nodeSize = continuousSizeScale(
      this.nodes.map(importanceFor),
      overviewLayout ? 18 : 16,
      overviewLayout ? 42 : 38,
    );
    const categories = this.nodes.map(semanticCategory);
    const hasMeaningfulCategories = new Set(categories).size > 1;
    const componentParent = this.nodes.map((_node, index) => index);
    const findComponent = index => {
      let root = index;
      while (componentParent[root] !== root) root = componentParent[root];
      while (componentParent[index] !== index) {
        const next = componentParent[index];
        componentParent[index] = root;
        index = next;
      }
      return root;
    };
    const joinComponents = (left, right) => {
      const leftRoot = findComponent(left);
      const rightRoot = findComponent(right);
      if (leftRoot !== rightRoot) componentParent[rightRoot] = leftRoot;
    };
    for (const edge of this.edges) {
      const left = this.nodeIndex.get(String(edge.source));
      const right = this.nodeIndex.get(String(edge.target));
      if (left !== undefined && right !== undefined) joinComponents(left, right);
    }
    const layoutGroups = this.nodes.map((node, index) => overviewLayout
      ? String(node.group ?? node.community ?? node.type ?? 'default')
      : `component:${findComponent(index)}`);
    const clusterIds = new Map();
    for (const group of layoutGroups) {
      if (!clusterIds.has(group)) clusterIds.set(group, clusterIds.size);
    }
    const clusters = [];
    const clusterStrength = new Float32Array(this.nodes.length);
    const ordered = rankNodes(this.nodes);
    const layoutOrder = new Map(ordered.map((item, index) => [item.index, index]));
    const orderWithinGroup = new Map();
    const groupedOrder = new Map();
    for (const item of ordered) {
      const group = layoutGroups[item.index];
      const members = groupedOrder.get(group) || [];
      orderWithinGroup.set(item.index, members.length);
      members.push(item.index);
      groupedOrder.set(group, members);
    }
    for (let index = 0; index < this.nodes.length; index += 1) {
      const node = this.nodes[index];
      const group = layoutGroups[index];
      const cluster = clusterIds.get(group);
      clusters.push(cluster);
      clusterStrength[index] = overviewLayout ? 0.26 : this.nodes.length > 110 ? 0.14 : 0.22;
      if (overviewLayout) {
        const [x, y] = overviewPosition(
          layoutOrder.get(index) ?? index,
          this.nodes.length,
          finite(node.size ?? node.member_count ?? node.degree, 1),
        );
        positions[index * 2] = x;
        positions[index * 2 + 1] = y;
      } else {
        const [x, y] = seededPosition(
          node,
          orderWithinGroup.get(index) ?? index,
          (groupedOrder.get(group) || []).length,
          cluster,
          clusterIds.size,
          compactLayout,
        );
        positions[index * 2] = x;
        positions[index * 2 + 1] = y;
      }
      sizes[index] = nodeSize(importanceFor(node));
    }
    let smallestPoint = Infinity;
    let largestPoint = -Infinity;
    for (const size of sizes) {
      smallestPoint = Math.min(smallestPoint, size);
      largestPoint = Math.max(largestPoint, size);
    }
    largestPoint = Math.max(smallestPoint + 0.01, largestPoint);
    for (let index = 0; index < this.nodes.length; index += 1) {
      // A single category (such as a community overview) uses a continuous
      // blue family. Several real categories retain their semantic hues.
      const centrality = (sizes[index] - smallestPoint) / (largestPoint - smallestPoint);
      const color = hasMeaningfulCategories
        ? rgb(categoryColor(categories[index]))
        : blueByCentrality(centrality);
      colors.set(color, index * 4);
    }

    const links = new Float32Array(this.edges.length * 2);
    const widths = new Float32Array(this.edges.length);
    const linkColors = new Float32Array(this.edges.length * 4);
    for (let index = 0; index < this.edges.length; index += 1) {
      const edge = this.edges[index];
      links[index * 2] = this.nodeIndex.get(String(edge.source));
      links[index * 2 + 1] = this.nodeIndex.get(String(edge.target));
      // Edge weight is available as a weak secondary signal only.  Keeping
      // the normal width in a narrow 1.0–1.3px band preserves the topology
      // instead of making the central portion into a dark mass.
      widths[index] = Math.max(1, Math.min(1.3, 1 + Math.log1p(finite(edge.weight ?? edge.count, 1)) * 0.09));
      // Cosmos retains the canonical link topology for hit testing and zoom,
      // while the DOM/SVG layer below is the single visible source of truth.
      // Keeping this GPU colour transparent prevents one relation from being
      // painted twice by two compositing layers.
      linkColors.set(rgb(BASE_LINK, 0), index * 4);
    }

    relaxPositions(positions, this.edges, this.nodeIndex, {compact: compactLayout, pointSizes: sizes});

    this.graph.setConfigPartial({
      // The scene gets a deterministic bounded layout before it reaches the
      // GPU.  Keeping the runtime simulation off prevents large communities
      // from reshuffling after labels and the detail panel have settled.
      enableSimulation: false,
      simulationGravity: overviewLayout ? 0 : 0.08,
      simulationRepulsion: overviewLayout ? 0 : 0.9,
      simulationLinkDistance: overviewLayout ? 230 : 170,
    });
    this.graph.setPointPositions(positions);
    this.graph.setPointColors(colors);
    this.graph.setPointSizes(sizes);
    this.graph.setPointClusters(clusters);
    this.graph.setPointClusterStrength(clusterStrength);
    this.graph.setLinks(links);
    this.graph.setLinkColors(linkColors);
    this.graph.setLinkWidths(widths);
    this.graph.setLinkArrows(this.edges.map(edge => edge.directed !== false));
    // Snap the new graph state into place.  The renderer still owns layout,
    // picking and zoom; there is deliberately no cross-view position tween.
    this.graph.render(undefined, 0);
    await this.graph.ready;
    if (this.destroyed || revision !== this.dataRevision) return;
    this.pointSizes = sizes;
    this.defaultPointColors = colors;
    this.defaultLinkColors = linkColors;
    this.defaultLinkWidths = widths;
    this.renderDefaultTopology();
    // A persistent ring around every node competes with the relationship
    // structure.  Reserve rings for the item a reader is inspecting.
    this.graph.setConfigPartial({outlinedPointIndices: undefined});
    // Cosmos begins at its minimum zoom before the page layout has settled.
    // A delayed fit here could capture that stale transform and leave an
    // otherwise valid graph collapsed into sub-pixels.  The data is ready at
    // this point, so synchronously fit once and use the resulting scale as the
    // reader's 100% baseline.
    this.fitZoomLevel = this.fitToVisibleGraph(this.mode === 'overview' ? 0.22 : 0.14) || null;
    window.setTimeout(() => {
      if (this.destroyed || revision !== this.dataRevision) return;
      const finalZoom = this.fitToVisibleGraph(this.mode === 'overview' ? 0.22 : 0.14);
      if (finalZoom) {
        this.fitZoomLevel = finalZoom;
        this.renderDefaultTopology();
        this.options.onZoomChange?.(100);
        this.renderLabels();
      }
    }, 240);
    this.scheduleLabels();
  }

  clearVisibleTopology() {
    this.defaultNodes.replaceChildren();
    this.defaultEdges.replaceChildren(this.defaultEdges.firstElementChild);
    this.emphasizedNodes.replaceChildren();
    this.emphasizedEdges.replaceChildren(this.emphasizedEdges.firstElementChild);
    this.nodeRings.replaceChildren();
    this.edgeLabels.replaceChildren();
    this.labels.replaceChildren();
    this.hideTooltip();
  }

  selectNode(index, emit = false) {
    if (!this.nodes[index]) return;
    this.selectedNode = index;
    this.selectedEdge = -1;
    this.hoveredNode = -1;
    this.hoveredEdge = -1;
    const neighbors = this.connectedNodeIndices(index);
    const points = [index, ...neighbors];
    const links = this.connectedLinkIndices(index);
    this.graph.setConfigPartial({
      focusedPointIndex: index,
      focusedLinkIndex: undefined,
      highlightedPointIndices: points,
      outlinedPointIndices: [index],
      highlightedLinkIndices: links,
      linkOpacity: 1,
    });
    this.setEmphasizedLinkColors(links);
    this.renderEmphasizedNodes(points, index);
    this.renderNodeRings([index], index);
    this.renderLabels();
    if (emit) this.options.onNodeSelect?.(this.nodes[index], index);
  }

  selectEdge(index, emit = false) {
    if (!this.edges[index]) return;
    this.selectedNode = -1;
    this.selectedEdge = index;
    this.hoveredNode = -1;
    this.hoveredEdge = -1;
    const points = this.edgePointIndices(index);
    this.graph.setConfigPartial({
      focusedPointIndex: undefined,
      focusedLinkIndex: index,
      highlightedPointIndices: points,
      outlinedPointIndices: points,
      highlightedLinkIndices: [index],
      linkOpacity: 1,
    });
    this.setEmphasizedLinkColors([index]);
    this.renderEmphasizedNodes(points);
    this.renderNodeRings(points);
    this.renderLabels();
    if (emit) this.options.onEdgeSelect?.(this.edges[index], index);
  }

  clearSelection(emit = false) {
    this.selectedNode = -1;
    this.selectedEdge = -1;
    this.hoveredNode = -1;
    this.hoveredEdge = -1;
    this.graph.setConfigPartial({
      focusedPointIndex: undefined,
      focusedLinkIndex: undefined,
      highlightedPointIndices: undefined,
      outlinedPointIndices: undefined,
      highlightedLinkIndices: undefined,
      linkOpacity: 0.32,
    });
    this.restoreLinkColors();
    this.renderEmphasizedNodes();
    this.renderNodeRings();
    this.hideTooltip();
    this.renderLabels();
    if (emit) this.options.onClear?.();
  }

  focusNode(id) {
    const index = this.nodeIndex.get(String(id));
    if (index !== undefined) this.selectNode(index, true);
  }

  fit() {
    this.clearSelection(false);
    this.fitZoomLevel = null;
    this.fitZoomLevel = this.fitToVisibleGraph(0.16) || null;
    window.setTimeout(() => {
      const finalZoom = this.fitToVisibleGraph(0.16);
      if (finalZoom) {
        this.fitZoomLevel = finalZoom;
        this.renderDefaultTopology();
        this.options.onZoomChange?.(100);
        this.renderLabels();
      }
    }, 240);
  }

  zoomBy(factor) {
    const current = this.graph.getZoomLevel();
    const baseline = this.currentFitZoomLevel();
    const next = Math.max(baseline * 0.25, Math.min(baseline * 8, current * finite(factor, 1)));
    this.graph.setZoomLevel(next, 0, false);
    this.options.onZoomChange?.(Math.round((next / baseline) * 100));
  }

  zoomPercent() {
    return Math.round((this.graph.getZoomLevel() / this.currentFitZoomLevel()) * 100);
  }

  currentFitZoomLevel() {
    const known = finite(this.fitZoomLevel, 0);
    if (known > 0) return known;
    return Math.max(0.0000001, finite(this.graph?.getZoomLevel?.(), 1));
  }

  fitToVisibleGraph(padding) {
    if (!this.graph.isReady || !this.nodes.length) return;
    // `fitView()` reaches a correct transform, but d3-zoom can overwrite it
    // with its initial 0.001 transform during the first page-layout cycle.
    // Directly applying the transform computed by the renderer avoids that
    // race and makes the resulting zoom the canonical 100% baseline.
    // The adapter may initiate the graph while the query panel is still
    // hidden.  Defer only until its canvas has a measurable box; otherwise
    // d3 computes a zero-sized fit and clamps it to 0.001.
    if (this.surface.clientWidth < 8 || this.surface.clientHeight < 8) {
      window.setTimeout(() => {
        const deferredZoom = this.fitToVisibleGraph(padding);
        if (deferredZoom) {
          this.fitZoomLevel = deferredZoom;
          this.options.onZoomChange?.(100);
          this.scheduleLabels();
        }
      }, 80);
      return;
    }
    const positions = this.graph.getPointPositions();
    const transform = this.graph.zoomInstance.getTransform(positions, undefined, padding);
    this.graph.zoomInstance.shouldEnableSimulationDuringZoomOverride = false;
    this.graph.canvasD3Selection.call(this.graph.zoomInstance.behavior.transform, transform);
    return transform.k;
  }

  renderDefaultTopology() {
    if (!this.graph.isReady || !this.nodes.length) {
      this.defaultNodes.replaceChildren();
      this.defaultEdges.replaceChildren(this.defaultEdges.firstElementChild);
      return;
    }
    const positions = this.graph.getPointPositions();
    const focusPoints = new Set();
    const focusLinks = new Set();
    // A click establishes persistent focus.  Temporary pointer hover must not
    // replace its orange active node or make the clicked neighbourhood appear
    // to change after the pointer moves away.
    if (this.selectedNode >= 0) {
      focusPoints.add(this.selectedNode);
      for (const index of this.connectedNodeIndices(this.selectedNode)) focusPoints.add(index);
      for (const index of this.connectedLinkIndices(this.selectedNode)) focusLinks.add(index);
    } else if (this.selectedEdge >= 0) {
      for (const index of this.edgePointIndices(this.selectedEdge)) focusPoints.add(index);
      focusLinks.add(this.selectedEdge);
    } else if (this.hoveredNode >= 0) {
      focusPoints.add(this.hoveredNode);
      for (const index of this.connectedNodeIndices(this.hoveredNode)) focusPoints.add(index);
      for (const index of this.connectedLinkIndices(this.hoveredNode)) focusLinks.add(index);
    } else if (this.hoveredEdge >= 0) {
      for (const index of this.edgePointIndices(this.hoveredEdge)) focusPoints.add(index);
      focusLinks.add(this.hoveredEdge);
    }
    const hasVisualFocus = focusPoints.size > 0;
    const definitions = this.defaultEdges.firstElementChild;
    const edges = [];
    for (let index = 0; index < this.edges.length; index += 1) {
      const edge = this.edges[index];
      const source = this.nodeIndex.get(String(edge.source));
      const target = this.nodeIndex.get(String(edge.target));
      if (source === undefined || target === undefined) continue;
      const sourceCenter = this.graph.spaceToScreenPosition([positions[source * 2], positions[source * 2 + 1]]);
      const targetCenter = this.graph.spaceToScreenPosition([positions[target * 2], positions[target * 2 + 1]]);
      const centerDx = targetCenter[0] - sourceCenter[0];
      const centerDy = targetCenter[1] - sourceCenter[1];
      const centerDistance = Math.max(1, Math.hypot(centerDx, centerDy));
      const startRadius = finite(this.pointSizes[source], 16) / 2 + 1;
      const endRadius = finite(this.pointSizes[target], 16) / 2 + (edge.directed !== false ? 4 : 1);
      const start = [sourceCenter[0] + (centerDx / centerDistance) * startRadius, sourceCenter[1] + (centerDy / centerDistance) * startRadius];
      const end = [targetCenter[0] - (centerDx / centerDistance) * endRadius, targetCenter[1] - (centerDy / centerDistance) * endRadius];
      const dx = end[0] - start[0];
      const dy = end[1] - start[1];
      const distance = Math.max(1, Math.hypot(dx, dy));
      const control = [
        (start[0] + end[0]) / 2 + (dy / distance) * distance * 0.12,
        (start[1] + end[1]) / 2 - (dx / distance) * distance * 0.12,
      ];
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', `M ${start[0]} ${start[1]} Q ${control[0]} ${control[1]} ${end[0]} ${end[1]}`);
      path.setAttribute('class', `kf-gpu-default-edge${hasVisualFocus && !focusLinks.has(index) ? ' is-muted' : ''}`);
      if (edge.directed !== false) path.setAttribute('marker-end', 'url(#kf-default-edge-arrow)');
      edges.push(path);
    }
    this.defaultEdges.replaceChildren(definitions, ...edges);
    const nodes = [];
    for (let index = 0; index < this.nodes.length; index += 1) {
      const point = this.graph.spaceToScreenPosition([positions[index * 2], positions[index * 2 + 1]]);
      const node = document.createElement('i');
      node.className = `kf-gpu-default-node${hasVisualFocus && !focusPoints.has(index) ? ' is-muted' : ''}`;
      const size = finite(this.pointSizes[index], 16);
      const offset = index * 4;
      node.style.width = `${Math.round(size)}px`;
      node.style.height = `${Math.round(size)}px`;
      node.style.backgroundColor = `rgb(${Math.round(this.defaultPointColors[offset] * 255)}, ${Math.round(this.defaultPointColors[offset + 1] * 255)}, ${Math.round(this.defaultPointColors[offset + 2] * 255)})`;
      node.style.transform = `translate(${Math.round(point[0] - size / 2)}px,${Math.round(point[1] - size / 2)}px)`;
      nodes.push(node);
    }
    this.defaultNodes.replaceChildren(...nodes);
  }

  selectAtClientPoint(clientX, clientY) {
    if (!this.graph.isReady || !this.nodes.length) return false;
    const canvasRect = this.surface.getBoundingClientRect();
    const pointer = [clientX - canvasRect.left, clientY - canvasRect.top];
    const positions = this.graph.getPointPositions();
    let closestNode = -1;
    let closestNodeDistance = Infinity;
    for (let index = 0; index < this.nodes.length; index += 1) {
      const point = this.graph.spaceToScreenPosition([positions[index * 2], positions[index * 2 + 1]]);
      const distance = Math.hypot(pointer[0] - point[0], pointer[1] - point[1]);
      const hitRadius = Math.max(11, finite(this.pointSizes[index], 16) / 2 + 3);
      if (distance <= hitRadius && distance < closestNodeDistance) {
        closestNode = index;
        closestNodeDistance = distance;
      }
    }
    let closestEdge = -1;
    let closestEdgeDistance = Infinity;
    for (let index = 0; index < this.edges.length; index += 1) {
      const edge = this.edges[index];
      const source = this.nodeIndex.get(String(edge.source));
      const target = this.nodeIndex.get(String(edge.target));
      if (source === undefined || target === undefined) continue;
      const start = this.graph.spaceToScreenPosition([positions[source * 2], positions[source * 2 + 1]]);
      const end = this.graph.spaceToScreenPosition([positions[target * 2], positions[target * 2 + 1]]);
      const distance = this.distanceToCurvedLink(pointer, start, end);
      if (distance <= 8 && distance < closestEdgeDistance) {
        closestEdge = index;
        closestEdgeDistance = distance;
      }
    }
    // A point inside a node's hit area is always a node click.  Comparing its
    // distance from the node centre with an edge's stroke distance made clicks
    // near the circumference intermittently select the edge instead, which in
    // turn removed the orange selected-node state.
    if (closestNode >= 0) {
      this.selectNode(closestNode, true);
      this.lastPointerSelection = {kind: 'node', index: closestNode};
      return true;
    }
    if (closestEdge >= 0) {
      this.selectEdge(closestEdge, true);
      this.lastPointerSelection = {kind: 'edge', index: closestEdge};
      return true;
    }
    this.lastPointerSelection = {kind: 'background'};
    this.clearSelection(true);
    return false;
  }

  distanceToCurvedLink(point, start, end) {
    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const distance = Math.max(1, Math.hypot(dx, dy));
    const control = [
      // graph coordinates use an upward Y axis, while DOM screen coordinates
      // use a downward Y axis.  Transform the renderer's perpendicular into
      // screen space before checking the visible curve.
      (start[0] + end[0]) / 2 + (dy / distance) * distance * 0.12,
      (start[1] + end[1]) / 2 - (dx / distance) * distance * 0.12,
    ];
    const curvePoint = t => {
      const inverse = 1 - t;
      const weight = 0.55;
      const divisor = inverse * inverse + 2 * inverse * t * weight + t * t;
      return [
        (inverse * inverse * start[0] + 2 * inverse * t * weight * control[0] + t * t * end[0]) / divisor,
        (inverse * inverse * start[1] + 2 * inverse * t * weight * control[1] + t * t * end[1]) / divisor,
      ];
    };
    let nearest = Infinity;
    let previous = start;
    for (let part = 1; part <= 12; part += 1) {
      const current = curvePoint(part / 12);
      nearest = Math.min(nearest, this.distanceToSegment(point, previous, current));
      previous = current;
    }
    return nearest;
  }

  distanceToSegment(point, start, end) {
    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const lengthSquared = dx * dx + dy * dy;
    if (lengthSquared < 0.0001) return Math.hypot(point[0] - start[0], point[1] - start[1]);
    const ratio = Math.max(0, Math.min(1, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / lengthSquared));
    return Math.hypot(point[0] - (start[0] + ratio * dx), point[1] - (start[1] + ratio * dy));
  }

  connectedLinkIndices(index) {
    const id = String(this.nodes[index]?.id);
    if (!id) return [];
    return this.edges.reduce((indices, edge, edgeIndex) => {
      if (String(edge.source) === id || String(edge.target) === id) indices.push(edgeIndex);
      return indices;
    }, []);
  }

  connectedNodeIndices(index) {
    const id = String(this.nodes[index]?.id);
    if (!id) return [];
    const related = new Set();
    for (const edge of this.edges) {
      if (String(edge.source) === id) related.add(this.nodeIndex.get(String(edge.target)));
      if (String(edge.target) === id) related.add(this.nodeIndex.get(String(edge.source)));
    }
    related.delete(undefined);
    return [...related];
  }

  edgePointIndices(index) {
    const edge = this.edges[index];
    if (!edge) return [];
    return [this.nodeIndex.get(String(edge.source)), this.nodeIndex.get(String(edge.target))]
      .filter(pointIndex => pointIndex !== undefined);
  }

  hoverNode(index, event) {
    this.hoveredNode = index;
    const node = this.nodes[index];
    if (!node) return;
    if (this.selectedNode >= 0 || this.selectedEdge >= 0) {
      this.restorePersistentFocus();
      this.showTooltip(short(node.title || node.label || node.id, 80), event);
      this.renderLabels();
      return;
    }
    const neighbours = this.connectedNodeIndices(index);
    const related = [index, ...neighbours];
    const links = this.connectedLinkIndices(index);
    this.graph.setConfigPartial({
      focusedPointIndex: this.selectedNode >= 0 ? this.selectedNode : undefined,
      focusedLinkIndex: this.selectedEdge >= 0 ? this.selectedEdge : undefined,
      highlightedPointIndices: related,
      outlinedPointIndices: [index],
      highlightedLinkIndices: links,
      linkOpacity: 1,
    });
    this.setEmphasizedLinkColors(links);
    this.renderEmphasizedNodes(related, index);
    this.renderNodeRings(related, index);
    this.showTooltip(short(node.title || node.label || node.id, 80), event);
    this.renderLabels();
  }

  unhoverNode() {
    this.hoveredNode = -1;
    this.restorePersistentFocus();
    this.hideTooltip();
    this.renderLabels();
  }

  hoverEdge(index) {
    const edge = this.edges[index];
    if (!edge) return;
    this.hoveredEdge = index;
    if (this.selectedNode >= 0 || this.selectedEdge >= 0) {
      this.restorePersistentFocus();
      const source = this.nodes[this.nodeIndex.get(String(edge.source))];
      const target = this.nodes[this.nodeIndex.get(String(edge.target))];
      this.tooltip.textContent = `${short(source?.title || edge.source, 34)} → ${short(target?.title || edge.target, 34)}：${short(edge.label || edge.predicate_label || edge.description || '关系', 70)}`;
      this.tooltip.hidden = false;
      this.tooltip.style.left = '18px';
      this.tooltip.style.top = '18px';
      this.renderLabels();
      return;
    }
    const points = this.edgePointIndices(index);
    this.graph.setConfigPartial({
      focusedPointIndex: this.selectedNode >= 0 ? this.selectedNode : undefined,
      focusedLinkIndex: this.selectedEdge >= 0 ? this.selectedEdge : undefined,
      highlightedPointIndices: points,
      outlinedPointIndices: points,
      highlightedLinkIndices: [index],
      linkOpacity: 1,
    });
    this.setEmphasizedLinkColors([index]);
    this.renderNodeRings(points);
    const source = this.nodes[this.nodeIndex.get(String(edge.source))];
    const target = this.nodes[this.nodeIndex.get(String(edge.target))];
    this.tooltip.textContent = `${short(source?.title || edge.source, 34)} → ${short(target?.title || edge.target, 34)}：${short(edge.label || edge.predicate_label || edge.description || '关系', 70)}`;
    this.tooltip.hidden = false;
    this.tooltip.style.left = '18px';
    this.tooltip.style.top = '18px';
    this.renderLabels();
  }

  unhoverEdge() {
    this.hoveredEdge = -1;
    this.restorePersistentFocus();
    this.hideTooltip();
    this.renderLabels();
  }

  restorePersistentFocus() {
    if (this.selectedNode >= 0) {
      const neighbours = this.connectedNodeIndices(this.selectedNode);
      const links = this.connectedLinkIndices(this.selectedNode);
      this.graph.setConfigPartial({
        focusedPointIndex: this.selectedNode,
        focusedLinkIndex: undefined,
        highlightedPointIndices: [this.selectedNode, ...neighbours],
        outlinedPointIndices: [this.selectedNode],
        highlightedLinkIndices: links,
        linkOpacity: 1,
      });
      this.setEmphasizedLinkColors(links);
      this.renderEmphasizedNodes([this.selectedNode, ...neighbours], this.selectedNode);
      this.renderNodeRings([this.selectedNode, ...neighbours], this.selectedNode);
    } else if (this.selectedEdge >= 0) {
      const points = this.edgePointIndices(this.selectedEdge);
      this.graph.setConfigPartial({
        focusedPointIndex: undefined,
        focusedLinkIndex: this.selectedEdge,
        highlightedPointIndices: points,
        outlinedPointIndices: points,
        highlightedLinkIndices: [this.selectedEdge],
        linkOpacity: 1,
      });
      this.setEmphasizedLinkColors([this.selectedEdge]);
      this.renderEmphasizedNodes(points);
      this.renderNodeRings(points);
    } else {
      this.graph.setConfigPartial({
        focusedPointIndex: undefined,
        focusedLinkIndex: undefined,
        highlightedPointIndices: undefined,
        outlinedPointIndices: undefined,
        highlightedLinkIndices: undefined,
        linkOpacity: 0.32,
      });
      this.restoreLinkColors();
      this.renderEmphasizedNodes();
      this.nodeRings.replaceChildren();
    }
  }

  setEmphasizedLinkColors(indices) {
    if (!this.defaultLinkColors.length) return;
    const emphasized = new Float32Array(this.defaultLinkColors);
    const widths = new Float32Array(this.defaultLinkWidths);
    // The highlighted line is rendered only by `renderEmphasizedEdges`.
    // Keep the GPU copy transparent so hover/selection also cannot produce
    // a second, slightly offset stroke.
    const color = rgb(EMPHASIZED_LINK, 0);
    for (const index of indices) {
      if (index < 0 || index >= this.edges.length) continue;
      emphasized.set(color, index * 4);
      // A highlighted relation is intentionally more than a colour change.
      // Cosmos only increases width for the one GPU-picked link; node hover
      // can involve several direct links, so give every direct relation a
      // clear 2.3px stroke here as well.
      widths[index] = Math.max(2.3, widths[index]);
    }
    this.graph.setLinkColors(emphasized);
    this.graph.setLinkWidths(widths);
    this.renderEmphasizedEdges(indices);
  }

  restoreLinkColors() {
    if (this.defaultLinkColors.length) this.graph.setLinkColors(this.defaultLinkColors);
    if (this.defaultLinkWidths.length) this.graph.setLinkWidths(this.defaultLinkWidths);
    this.renderEmphasizedEdges();
  }

  renderEmphasizedNodes(indices = [], activeIndex = -1) {
    if (!this.graph.isReady || !indices.length || !this.defaultPointColors.length) {
      this.emphasizedNodes.replaceChildren();
      return;
    }
    const positions = this.graph.getPointPositions();
    const nodes = [];
    for (const index of indices) {
      if (!this.nodes[index]) continue;
      const point = this.graph.spaceToScreenPosition([positions[index * 2], positions[index * 2 + 1]]);
      const node = document.createElement('i');
      node.className = `kf-gpu-emphasized-node${index === activeIndex ? ' is-active' : ''}`;
      const size = finite(this.pointSizes[index], 16);
      const colorOffset = index * 4;
      const color = index === activeIndex
        ? ACCENT_ORANGE
        : `rgb(${Math.round(this.defaultPointColors[colorOffset] * 255)}, ${Math.round(this.defaultPointColors[colorOffset + 1] * 255)}, ${Math.round(this.defaultPointColors[colorOffset + 2] * 255)})`;
      node.style.width = `${Math.round(size)}px`;
      node.style.height = `${Math.round(size)}px`;
      node.style.backgroundColor = color;
      node.style.transform = `translate(${Math.round(point[0] - size / 2)}px,${Math.round(point[1] - size / 2)}px)`;
      nodes.push(node);
    }
    this.emphasizedNodes.replaceChildren(...nodes);
  }

  renderEmphasizedEdges(indices = []) {
    if (!this.graph.isReady || !indices.length) {
      this.emphasizedEdges.replaceChildren(this.emphasizedEdges.firstElementChild);
      return;
    }
    const definitions = this.emphasizedEdges.firstElementChild;
    const positions = this.graph.getPointPositions();
    const paths = [];
    for (const index of indices) {
      const edge = this.edges[index];
      if (!edge) continue;
      const source = this.nodeIndex.get(String(edge.source));
      const target = this.nodeIndex.get(String(edge.target));
      if (source === undefined || target === undefined) continue;
      const sourceCenter = this.graph.spaceToScreenPosition([positions[source * 2], positions[source * 2 + 1]]);
      const targetCenter = this.graph.spaceToScreenPosition([positions[target * 2], positions[target * 2 + 1]]);
      const dx = targetCenter[0] - sourceCenter[0];
      const dy = targetCenter[1] - sourceCenter[1];
      const distance = Math.max(1, Math.hypot(dx, dy));
      // End the overlay at the visible node boundary.  The relationship must
      // be clear without painting a dark stripe across either node.
      const startRadius = finite(this.pointSizes[source], 16) / 2 + 2;
      const endRadius = finite(this.pointSizes[target], 16) / 2 + 4;
      const start = [sourceCenter[0] + (dx / distance) * startRadius, sourceCenter[1] + (dy / distance) * startRadius];
      const end = [targetCenter[0] - (dx / distance) * endRadius, targetCenter[1] - (dy / distance) * endRadius];
      const visibleDx = end[0] - start[0];
      const visibleDy = end[1] - start[1];
      const visibleDistance = Math.max(1, Math.hypot(visibleDx, visibleDy));
      const control = [
        (start[0] + end[0]) / 2 + (visibleDy / visibleDistance) * visibleDistance * 0.12,
        (start[1] + end[1]) / 2 - (visibleDx / visibleDistance) * visibleDistance * 0.12,
      ];
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', `M ${start[0]} ${start[1]} Q ${control[0]} ${control[1]} ${end[0]} ${end[1]}`);
      path.setAttribute('class', 'kf-gpu-emphasized-edge');
      if (edge.directed !== false) path.setAttribute('marker-end', 'url(#kf-emphasized-edge-arrow)');
      paths.push(path);
    }
    this.emphasizedEdges.replaceChildren(definitions, ...paths);
  }

  renderNodeRings(indices = [], activeIndex = -1) {
    if (!this.graph.isReady || !this.pointSizes.length) return;
    const positions = this.graph.getPointPositions();
    const rings = [];
    for (const index of indices) {
      const point = this.graph.spaceToScreenPosition([positions[index * 2], positions[index * 2 + 1]]);
      const ring = document.createElement('i');
      ring.className = `kf-gpu-node-ring is-related${index === activeIndex ? ' is-active' : ''}`;
      const size = finite(this.pointSizes[index], 16) + (index === activeIndex ? 8 : 4);
      ring.style.width = `${Math.round(size)}px`;
      ring.style.height = `${Math.round(size)}px`;
      ring.style.transform = `translate(${Math.round(point[0] - size / 2)}px,${Math.round(point[1] - size / 2)}px)`;
      rings.push(ring);
    }
    this.nodeRings.replaceChildren(...rings);
  }

  showTooltip(text, event) {
    this.tooltip.textContent = text;
    this.tooltip.hidden = false;
    const bounds = this.container.getBoundingClientRect();
    const clientX = finite(event?.sourceEvent?.clientX ?? event?.clientX, bounds.left + 20);
    const clientY = finite(event?.sourceEvent?.clientY ?? event?.clientY, bounds.top + 20);
    this.tooltip.style.left = `${Math.min(bounds.width - 250, Math.max(8, clientX - bounds.left + 12))}px`;
    this.tooltip.style.top = `${Math.min(bounds.height - 48, Math.max(8, clientY - bounds.top + 12))}px`;
  }

  hideTooltip() {
    this.tooltip.hidden = true;
  }

  scheduleLabels() {
    if (this.destroyed || this.labelFrame) return;
    this.labelFrame = requestAnimationFrame(() => {
      this.labelFrame = 0;
      this.renderLabels();
      if (this.selectedNode >= 0) {
        const links = this.connectedLinkIndices(this.selectedNode);
        this.renderEmphasizedEdges(links);
        this.renderEmphasizedNodes([this.selectedNode, ...this.connectedNodeIndices(this.selectedNode)], this.selectedNode);
        this.renderNodeRings([this.selectedNode, ...this.connectedNodeIndices(this.selectedNode)], this.selectedNode);
      } else if (this.selectedEdge >= 0) {
        this.renderEmphasizedEdges([this.selectedEdge]);
        this.renderEmphasizedNodes(this.edgePointIndices(this.selectedEdge));
        this.renderNodeRings(this.edgePointIndices(this.selectedEdge));
      } else if (this.hoveredNode >= 0) {
        const links = this.connectedLinkIndices(this.hoveredNode);
        this.renderEmphasizedEdges(links);
        this.renderEmphasizedNodes([this.hoveredNode, ...this.connectedNodeIndices(this.hoveredNode)], this.hoveredNode);
        this.renderNodeRings([this.hoveredNode, ...this.connectedNodeIndices(this.hoveredNode)], this.hoveredNode);
      } else if (this.hoveredEdge >= 0) {
        this.renderEmphasizedEdges([this.hoveredEdge]);
        this.renderEmphasizedNodes(this.edgePointIndices(this.hoveredEdge));
        this.renderNodeRings(this.edgePointIndices(this.hoveredEdge));
      } else {
        this.renderEmphasizedEdges();
        this.renderEmphasizedNodes();
        this.nodeRings.replaceChildren();
      }
    });
  }

  renderLabels() {
    if (this.destroyed || !this.nodes.length || !this.graph.isReady) return;
    this.renderDefaultTopology();
    const zoom = this.graph.getZoomLevel();
    // Cosmos zoom levels vary with the dimensions and density of a graph.  A
    // relative scale is therefore the only meaningful way to decide when a
    // reader has deliberately zoomed in far enough to reveal peripheral names.
    const relativeZoom = zoom / this.currentFitZoomLevel();
    const selected = new Set();
    const hoveredRelated = new Set();
    if (this.selectedNode >= 0) {
      selected.add(this.selectedNode);
      for (const index of this.connectedNodeIndices(this.selectedNode)) selected.add(index);
    }
    if (this.selectedEdge >= 0) {
      for (const index of this.edgePointIndices(this.selectedEdge)) selected.add(index);
    }
    if (this.hoveredNode >= 0) {
      // Hover is a temporary local reading mode: reveal the hovered entity and
      // every directly connected entity, so a user can read a relationship
      // neighbourhood without committing to a click or a drill-down.
      hoveredRelated.add(this.hoveredNode);
      for (const index of this.connectedNodeIndices(this.hoveredNode)) {
        hoveredRelated.add(index);
      }
    }
    let indices = [];
    if (hoveredRelated.size) {
      indices = [...hoveredRelated];
    } else if (selected.size) {
      indices = [...selected].slice(0, 32);
    } else if (this.mode === 'overview') {
      indices = this.nodes
        .map((node, index) => ({index, score: finite(node.size ?? node.member_count ?? node.degree)}))
        .sort((left, right) => right.score - left.score)
        .slice(0, this.nodes.length <= 24 || relativeZoom >= 1.35 ? 26 : 16)
        .map(item => item.index);
    } else if (this.nodes.length <= 46 || relativeZoom >= 1.4) {
      // Once a reader has deliberately zoomed in, every node in the current
      // view deserves a chance to show its name.  The collision pass below
      // still prevents labels from drawing over one another, but degree must
      // not silently hide peripheral facts at high magnification.
      const candidates = relativeZoom >= 2.2
        ? this.nodes.map((_node, index) => index)
        : this.graph.getSampledPoints().indices;
      indices = candidates
        .sort((left, right) => finite(this.nodes[right]?.degree ?? this.nodes[right]?.size) - finite(this.nodes[left]?.degree ?? this.nodes[left]?.size))
        .slice(0, relativeZoom >= 2.2 ? Math.min(this.nodes.length, 180) : relativeZoom >= 1.8 ? 34 : 18);
    } else {
      indices = this.nodes
        .map((node, index) => ({index, score: finite(node.degree ?? node.size ?? node.member_count)}))
        .sort((left, right) => right.score - left.score)
        .slice(0, 10)
        .map(item => item.index);
    }
    const positions = this.graph.getPointPositions();
    const activeEdge = this.selectedEdge >= 0 ? this.selectedEdge : this.hoveredEdge;
    const labelEdges = activeEdge >= 0
      ? [[this.edges[activeEdge], activeEdge]]
      : this.edges.length <= 40 && this.mode !== 'overview' && relativeZoom >= 1
        ? this.edges.map((edge, index) => [edge, index])
        : [];
    const labelEntries = [];
    const occupiedLabels = [];
    const rankedLabelEdges = [...labelEdges].sort(([left], [right]) =>
      finite(right.weight ?? right.count, 1) - finite(left.weight ?? left.count, 1)
      || String(left.id ?? '').localeCompare(String(right.id ?? ''))
    );
    for (const [edge, index] of rankedLabelEdges) {
      const sourceIndex = this.nodeIndex.get(String(edge.source));
      const targetIndex = this.nodeIndex.get(String(edge.target));
      if (sourceIndex === undefined || targetIndex === undefined) continue;
      const pointA = this.graph.spaceToScreenPosition([
        positions[sourceIndex * 2], positions[sourceIndex * 2 + 1],
      ]);
      const pointB = this.graph.spaceToScreenPosition([
        positions[targetIndex * 2], positions[targetIndex * 2 + 1],
      ]);
      const text = short(edge.label || edge.predicate_label || edge.predicate || '关系', 22);
      const dx = pointB[0] - pointA[0];
      const dy = pointB[1] - pointA[1];
      const length = Math.max(1, Math.hypot(dx, dy));
      const direction = hash(edge.id ?? `${edge.source}:${edge.target}`) % 2 ? 1 : -1;
      const offset = Math.min(18, Math.max(8, length * 0.07)) * direction;
      const x = (pointA[0] + pointB[0]) / 2 - (dy / length) * offset;
      const y = (pointA[1] + pointB[1]) / 2 + (dx / length) * offset;
      const width = Math.min(130, Math.max(44, 14 + text.length * 6.3));
      const bounds = {left: x - width / 2, top: y - 11, right: x + width / 2, bottom: y + 11};
      if (this.intersectsCanvasControls(bounds)) continue;
      const overlaps = occupiedLabels.some(other =>
        bounds.left < other.right && bounds.right > other.left
        && bounds.top < other.bottom && bounds.bottom > other.top
      );
      // Never suppress a relationship that the user is actively inspecting;
      // otherwise retain a non-overlapping label and let hover reveal the rest.
      if (overlaps && activeEdge !== index) continue;
      occupiedLabels.push(bounds);
      const label = document.createElement('span');
      label.className = `kf-gpu-edge-label${activeEdge === index ? ' is-selected' : ''}`;
      label.textContent = text;
      label.style.transform = `translate(${Math.round(x)}px,${Math.round(y)}px)`;
      labelEntries.push(label);
    }
    this.edgeLabels.replaceChildren(...labelEntries);
    const ranked = rankNodes(this.nodes);
    const defaultLabelLimit = this.mode === 'overview'
      ? (this.nodes.length <= 24 || relativeZoom >= 2 ? 24 : relativeZoom >= 1.35 ? 16 : 7)
      : (relativeZoom >= 2.2 ? Math.min(this.nodes.length, 180) : relativeZoom >= 1.8 ? 26 : relativeZoom >= 1.4 ? 14 : 8);
    const defaultLabels = new Set(ranked.slice(0, defaultLabelLimit).map(item => item.index));
    const labelIndices = hoveredRelated.size || selected.size
      ? indices
      : indices.filter(index => defaultLabels.has(index));
    const occupiedNodeLabels = [];
    const nodeLabelEntries = [];
    for (const index of labelIndices) {
      const node = this.nodes[index];
      const point = this.graph.spaceToScreenPosition([positions[index * 2], positions[index * 2 + 1]]);
      const title = short(node.title || node.label || node.id, this.mode === 'overview' ? 24 : relativeZoom >= 2.2 ? 34 : 20);
      const fontSize = hoveredRelated.has(index) ? 12 : 11;
      const width = Math.min(196, Math.max(34, title.length * (fontSize * 0.78) + 10));
      const height = fontSize + 8;
      const x = Math.round(point[0] - width / 2);
      const y = Math.round(point[1] + finite(this.pointSizes[index], 16) / 2 + 6);
      const bounds = {left: x, top: y, right: x + width, bottom: y + height};
      // Controls belong to the interaction layer, not the graph.  A selected
      // node must not be allowed to put its label beneath them either: doing
      // so produces a half-visible name and makes the control look broken.
      if (this.intersectsCanvasControls(bounds)) continue;
      const overlaps = occupiedNodeLabels.some(other =>
        bounds.left < other.right && bounds.right > other.left
        && bounds.top < other.bottom && bounds.bottom > other.top
      );
      if (overlaps && !hoveredRelated.has(index) && !selected.has(index)) continue;
      occupiedNodeLabels.push(bounds);
      const label = document.createElement('span');
      label.className = `kf-gpu-label${selected.has(index) ? ' is-selected' : ''}${hoveredRelated.has(index) ? ' is-related' : ''}`;
      label.textContent = title;
      label.style.fontSize = `${fontSize}px`;
      label.style.maxWidth = `${width}px`;
      label.style.transform = `translate(${x}px,${y}px)`;
      nodeLabelEntries.push(label);
    }
    this.labels.replaceChildren(...nodeLabelEntries);
  }

  intersectsCanvasControls(bounds) {
    const canvasBounds = this.container.getBoundingClientRect();
    // Only the currently visible mutually-exclusive scope control occupies
    // the top-left corner.  Account for both the overview and community
    // variants so labels never appear underneath either control.
    const labels = ['.graph-canvas-scope:not([hidden])', '.graph-canvas-nav'];
    return labels.some(selector => {
      const control = this.container.closest('.embedded-graph-canvas')?.querySelector(selector);
      if (!control || control.hidden) return false;
      const rect = control.getBoundingClientRect();
      const reserved = {
        left: rect.left - canvasBounds.left - 8,
        top: rect.top - canvasBounds.top - 8,
        right: rect.right - canvasBounds.left + 8,
        bottom: rect.bottom - canvasBounds.top + 8,
      };
      return bounds.left < reserved.right && bounds.right > reserved.left
        && bounds.top < reserved.bottom && bounds.bottom > reserved.top;
    });
  }

  destroy() {
    this.destroyed = true;
    if (this.labelFrame) cancelAnimationFrame(this.labelFrame);
    this.resizeObserver.disconnect();
    this.graph.destroy();
    this.container.replaceChildren();
  }
}

export const license = Object.freeze({
  renderer: '@cosmos.gl/graph',
  version: '3.4.0',
  license: 'MIT',
});
