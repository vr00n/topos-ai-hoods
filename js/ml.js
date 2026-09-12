// Small, dependency-free numerics: standardisation, PCA (Jacobi eigen-solver),
// k-means++ with restarts, Pearson correlation. Everything runs in the browser
// so the user can change K and the feature set interactively.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Column-wise z-score. Returns {Z, mean, std, keep} where keep = indices with non-zero variance. */
export function standardize(X) {
  const n = X.length, d = X[0].length;
  const mean = new Array(d).fill(0), std = new Array(d).fill(0);
  for (const row of X) for (let j = 0; j < d; j++) mean[j] += row[j];
  for (let j = 0; j < d; j++) mean[j] /= n;
  for (const row of X) for (let j = 0; j < d; j++) { const v = row[j] - mean[j]; std[j] += v * v; }
  for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j] / Math.max(1, n - 1));
  const keep = [];
  for (let j = 0; j < d; j++) if (std[j] > 1e-9) keep.push(j);
  const Z = X.map(row => keep.map(j => (row[j] - mean[j]) / std[j]));
  return { Z, mean, std, keep };
}

function covariance(Z) {
  const n = Z.length, d = Z[0].length;
  const C = Array.from({ length: d }, () => new Float64Array(d));
  for (const row of Z) {
    for (let i = 0; i < d; i++) {
      const ri = row[i];
      if (ri === 0) continue;
      const Ci = C[i];
      for (let j = i; j < d; j++) Ci[j] += ri * row[j];
    }
  }
  for (let i = 0; i < d; i++) for (let j = i; j < d; j++) { C[i][j] /= (n - 1); C[j][i] = C[i][j]; }
  return C;
}

/** Cyclic Jacobi eigen-decomposition of a symmetric matrix. Returns {values, vectors} (vectors as columns). */
function jacobiEigen(A) {
  const d = A.length;
  const a = A.map(r => Float64Array.from(r));
  const V = Array.from({ length: d }, (_, i) => { const r = new Float64Array(d); r[i] = 1; return r; });
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let i = 0; i < d; i++) for (let j = i + 1; j < d; j++) off += a[i][j] * a[i][j];
    if (off < 1e-18) break;
    for (let p = 0; p < d; p++) {
      for (let q = p + 1; q < d; q++) {
        if (Math.abs(a[p][q]) < 1e-14) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < d; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < d; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < d; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const idx = Array.from({ length: d }, (_, i) => i).sort((i, j) => a[j][j] - a[i][i]);
  return {
    values: idx.map(i => Math.max(0, a[i][i])),
    vectors: idx.map(i => Array.from({ length: d }, (_, k) => V[k][i])), // vectors[c][k]
  };
}

/**
 * PCA on standardised data. Keeps the smallest number of components explaining
 * >= varianceTarget of total variance (min 2).
 */
export function pca(Z, varianceTarget = 0.86) {
  const { values, vectors } = jacobiEigen(covariance(Z));
  const total = values.reduce((s, v) => s + v, 0) || 1;
  const ratios = values.map(v => v / total);
  let k = 0, cum = 0;
  while (k < ratios.length && (cum < varianceTarget || k < 2)) { cum += ratios[k]; k++; }
  // fix sign convention: make the largest-magnitude loading of each component positive
  const comps = vectors.slice(0, k).map(vec => {
    let m = 0, mi = 0;
    vec.forEach((v, i) => { if (Math.abs(v) > m) { m = Math.abs(v); mi = i; } });
    return vec[mi] < 0 ? vec.map(v => -v) : vec;
  });
  const scores = Z.map(row => comps.map(vec => { let s = 0; for (let i = 0; i < row.length; i++) s += row[i] * vec[i]; return s; }));
  return { scores, components: comps, ratios, explained: cum, nComponents: k };
}

function sqdist(a, b) { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return s; }

/** k-means++ with several restarts; deterministic for a given seed. */
export function kmeans(X, k, { restarts = 12, maxIter = 200, seed = 42 } = {}) {
  const n = X.length;
  if (n === 0) return { labels: [], centroids: [], inertia: 0 };
  k = Math.min(k, n);
  let best = null;
  for (let r = 0; r < restarts; r++) {
    const rnd = mulberry32(seed + r * 7919);
    // k-means++ init
    const centroids = [X[Math.floor(rnd() * n)].slice()];
    const dist = new Float64Array(n).fill(Infinity);
    while (centroids.length < k) {
      const c = centroids[centroids.length - 1];
      let sum = 0;
      for (let i = 0; i < n; i++) { const d = sqdist(X[i], c); if (d < dist[i]) dist[i] = d; sum += dist[i]; }
      let t = rnd() * sum, pick = n - 1;
      for (let i = 0; i < n; i++) { t -= dist[i]; if (t <= 0) { pick = i; break; } }
      centroids.push(X[pick].slice());
    }
    const labels = new Int32Array(n).fill(-1);
    let inertia = 0;
    for (let it = 0; it < maxIter; it++) {
      let changed = 0; inertia = 0;
      for (let i = 0; i < n; i++) {
        let bi = 0, bd = Infinity;
        for (let c = 0; c < k; c++) { const d = sqdist(X[i], centroids[c]); if (d < bd) { bd = d; bi = c; } }
        if (labels[i] !== bi) { labels[i] = bi; changed++; }
        inertia += bd;
      }
      if (changed === 0 && it > 0) break;
      const sums = centroids.map(c => new Float64Array(c.length)), cnt = new Int32Array(k);
      for (let i = 0; i < n; i++) { const l = labels[i]; cnt[l]++; const s = sums[l], x = X[i]; for (let j = 0; j < x.length; j++) s[j] += x[j]; }
      for (let c = 0; c < k; c++) {
        if (cnt[c] === 0) { centroids[c] = X[Math.floor(rnd() * n)].slice(); continue; }
        for (let j = 0; j < centroids[c].length; j++) centroids[c][j] = sums[c][j] / cnt[c];
      }
    }
    if (!best || inertia < best.inertia) best = { labels: Array.from(labels), centroids: centroids.map(c => c.slice()), inertia };
  }
  return best;
}

export function pearson(x, y) {
  const n = x.length; let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
  mx /= n; my /= n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { const a = x[i] - mx, b = y[i] - my; sxy += a * b; sxx += a * a; syy += b * b; }
  return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : 0;
}

export function quantile(sorted, q) {
  if (!sorted.length) return 0;
  const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}
