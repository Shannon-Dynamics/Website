/**
 * Small dense linear algebra, sized for state estimation.
 *
 * The Rust side uses `nalgebra`'s stack-allocated `SMatrix<f64, N, M>`; here we
 * use plain nested arrays. Dimensions are small (2–9 for most of this book), so
 * clarity beats cleverness — these routines are meant to be *read* alongside
 * the equations they implement.
 */

export type Vec = number[];
export type Mat = number[][];

export const zeros = (n: number): Vec => new Array(n).fill(0);

export const zerosMat = (r: number, c: number = r): Mat =>
  Array.from({ length: r }, () => new Array(c).fill(0));

export const eye = (n: number, scale = 1): Mat =>
  Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? scale : 0)),
  );

export const diag = (d: number[]): Mat =>
  Array.from({ length: d.length }, (_, i) =>
    Array.from({ length: d.length }, (_, j) => (i === j ? d[i] : 0)),
  );

export const add = (a: Vec, b: Vec): Vec => a.map((x, i) => x + b[i]);
export const sub = (a: Vec, b: Vec): Vec => a.map((x, i) => x - b[i]);
export const scale = (a: Vec, s: number): Vec => a.map((x) => x * s);
export const dot = (a: Vec, b: Vec): number => a.reduce((s, x, i) => s + x * b[i], 0);
export const norm = (a: Vec): number => Math.sqrt(dot(a, a));

export const matAdd = (a: Mat, b: Mat): Mat => a.map((r, i) => r.map((x, j) => x + b[i][j]));
export const matSub = (a: Mat, b: Mat): Mat => a.map((r, i) => r.map((x, j) => x - b[i][j]));
export const matScale = (a: Mat, s: number): Mat => a.map((r) => r.map((x) => x * s));

/** Matrix product A (r×k) · B (k×c). */
export function matMul(a: Mat, b: Mat): Mat {
  const r = a.length;
  const k = b.length;
  const c = b[0].length;
  const out = zerosMat(r, c);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) {
      let s = 0;
      for (let t = 0; t < k; t++) s += a[i][t] * b[t][j];
      out[i][j] = s;
    }
  }
  return out;
}

/** Matrix–vector product. */
export function matVec(a: Mat, v: Vec): Vec {
  return a.map((row) => row.reduce((s, x, j) => s + x * v[j], 0));
}

export function transpose(a: Mat): Mat {
  return a[0].map((_, j) => a.map((row) => row[j]));
}

/** Outer product v wᵀ. */
export function outer(v: Vec, w: Vec): Mat {
  return v.map((x) => w.map((y) => x * y));
}

/** Gauss–Jordan inverse with partial pivoting. Throws on singular input. */
export function inv(a: Mat): Mat {
  const n = a.length;
  const m = a.map((row, i) => [...row, ...eye(n)[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) piv = r;
    }
    if (Math.abs(m[piv][col]) < 1e-12) throw new Error('inv: matrix is singular');
    [m[col], m[piv]] = [m[piv], m[col]];
    const d = m[col][col];
    for (let j = 0; j < 2 * n; j++) m[col][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = m[r][col];
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) m[r][j] -= f * m[col][j];
    }
  }
  return m.map((row) => row.slice(n));
}

/** Lower-triangular Cholesky factor L with A = L Lᵀ. */
export function cholesky(a: Mat): Mat {
  const n = a.length;
  const l = zerosMat(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = 0;
      for (let k = 0; k < j; k++) s += l[i][k] * l[j][k];
      if (i === j) {
        l[i][j] = Math.sqrt(Math.max(a[i][i] - s, 1e-12));
      } else {
        l[i][j] = (a[i][j] - s) / l[j][j];
      }
    }
  }
  return l;
}

/** Solve A x = b by Gaussian elimination with partial pivoting. */
export function solve(a: Mat, b: Vec): Vec {
  const n = a.length;
  const m = a.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) piv = r;
    }
    if (Math.abs(m[piv][col]) < 1e-12) {
      // Tikhonov nudge keeps interactive demos alive on degenerate input.
      m[piv][col] += 1e-9;
    }
    [m[col], m[piv]] = [m[piv], m[col]];
    for (let r = col + 1; r < n; r++) {
      const f = m[r][col] / m[col][col];
      for (let j = col; j <= n; j++) m[r][j] -= f * m[col][j];
    }
  }
  const x = zeros(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = m[i][n];
    for (let j = i + 1; j < n; j++) s -= m[i][j] * x[j];
    x[i] = s / m[i][i];
  }
  return x;
}

/** Force symmetry — the standard numerical hygiene step after a filter update. */
export const symmetrize = (a: Mat): Mat =>
  a.map((row, i) => row.map((x, j) => (x + a[j][i]) / 2));

/**
 * Eigen-decomposition of a symmetric 2×2 matrix, returned as the geometry of
 * its confidence ellipse. This is what every covariance visualization draws.
 */
export function ellipse2(cov: Mat, nSigma = 2) {
  const [a, b] = [cov[0][0], cov[0][1]];
  const d = cov[1][1];
  const tr = a + d;
  const det = a * d - b * b;
  const disc = Math.sqrt(Math.max(tr * tr / 4 - det, 0));
  const l1 = tr / 2 + disc;
  const l2 = tr / 2 - disc;
  const angle = Math.abs(b) < 1e-12 ? (a >= d ? 0 : Math.PI / 2) : Math.atan2(l1 - a, b);
  return {
    rx: nSigma * Math.sqrt(Math.max(l1, 0)),
    ry: nSigma * Math.sqrt(Math.max(l2, 0)),
    angle,
  };
}
