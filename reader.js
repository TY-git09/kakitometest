// 受領証の読み取り。すべてこの端末のブラウザの中で動く。
// 画像も文字も、どこにも送信しない(通信するコードがこのファイルには無い)。
//
// 流れ:
//   1. 写真から受領証の外枠を見つけ、真上から見た形に補正する
//   2. 決まった位置で 6 つの枠に切り分ける
//   3. 枠ごとに文字を読む(同じ枠から出た番号と宛先は必ず同じ郵便のもの)

/* global ort */

// 補正後の外枠の大きさ(px)。サンプルの受領証の縦横比 876:1134 に合わせている。
const FRAME_W = 1752, FRAME_H = 2268;
const MARGIN = 150;               // 外枠の外側に残す余白(左下の発行日時を読むため)

// 外枠の中での位置(割合)。実物の様式が違う場合はここを調整する。
const HEADER_BOTTOM = 0.2328;     // ご依頼人欄の下端
const ROW_LINES_DEFAULT = [0.2328, 0.4877, 0.7416, 1.0];
const MID_X = 0.5;                // 左右の列の境目
const HEADER_DIVIDER_X = 0.349;   // ご依頼人欄の右端の縦線
const ADDRESS_RIGHT = 0.70;       // 枠の中で宛先画像が占める右端
const ADDRESS_TOP = 0.16;         // 枠の中で宛先画像が始まる上端
const NUMBER_BAND_BOTTOM = 0.20;  // 「お問い合わせ番号」の行がある帯の下端
const SERVICE_WORDS = ["現金書留", "一般書留", "簡易書留", "特定記録", "書留"];

let cv = null, det = null, rec = null, charset = null;

export function checkDigitOk(number) {
  // 先頭10桁を7で割った余りが末尾1桁になる
  return /^\d{11}$/.test(number) && Number(number.slice(0, 10)) % 7 === Number(number[10]);
}

const nfkc = (s) => s.normalize("NFKC");

export async function init(onProgress = () => {}) {
  if (rec) return;
  onProgress("画像処理の部品を読み込み中");
  let mod = window.cv;
  if (mod && typeof mod.then === "function") mod = await mod;
  if (!mod.Mat) await new Promise((resolve) => { mod.onRuntimeInitialized = resolve; });
  cv = mod;
  ort.env.wasm.wasmPaths = new URL("./", document.baseURI).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  onProgress("文字の一覧を読み込み中");
  charset = await (await fetch("charset.json")).json();
  onProgress("行を見つけるモデルを読み込み中");
  det = await ort.InferenceSession.create("det.onnx", { executionProviders: ["wasm"] });
  onProgress("文字を読むモデルを読み込み中(初回は時間がかかります)");
  // モデルは大きいので2つのファイルに分けて置いてある。読み込んでからつなぎ直す。
  const parts = await Promise.all(["rec.onnx.part0", "rec.onnx.part1"].map(
    async (url) => new Uint8Array(await (await fetch(url)).arrayBuffer())));
  const model = new Uint8Array(parts[0].length + parts[1].length);
  model.set(parts[0], 0);
  model.set(parts[1], parts[0].length);
  rec = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
}

// ---------------------------------------------------------------------------
// OpenCV の Mat は自分で解放する必要があるので、まとめて解放する仕組みを使う
// ---------------------------------------------------------------------------
class Scope {
  constructor() { this.items = []; }
  add(m) { this.items.push(m); return m; }
  mat() { return this.add(new cv.Mat()); }
  free() { for (const m of this.items) { try { m.delete(); } catch (e) { /* 解放済み */ } } this.items = []; }
}

function quadMat(scope, pts) {
  return scope.add(cv.matFromArray(4, 1, cv.CV_32FC2, pts.flat()));
}

function warp(scope, img, quad, w, h, margin) {
  const dst = [[margin, margin], [margin + w, margin], [margin + w, margin + h], [margin, margin + h]];
  const m = scope.add(cv.getPerspectiveTransform(quadMat(scope, quad), quadMat(scope, dst)));
  const out = scope.mat();
  cv.warpPerspective(img, out, m, new cv.Size(w + 2 * margin, h + 2 * margin),
    cv.INTER_LINEAR, cv.BORDER_REPLICATE, new cv.Scalar());
  return out;
}

function binarize(scope, gray) {
  const out = scope.mat();
  cv.adaptiveThreshold(gray, out, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 12);
  return out;
}

function toGray(scope, bgr) {
  const out = scope.mat();
  cv.cvtColor(bgr, out, cv.COLOR_BGR2GRAY);
  return out;
}

// ---------------------------------------------------------------------------
// 外枠の検出と補正
// ---------------------------------------------------------------------------
function orderCorners(pts) {
  // 左上・右上・右下・左下の順に並べる
  const sum = pts.map((p) => p[0] + p[1]), diff = pts.map((p) => p[1] - p[0]);
  const at = (arr, fn) => pts[arr.indexOf(fn(...arr))];
  return [at(sum, Math.min), at(diff, Math.min), at(sum, Math.max), at(diff, Math.max)];
}

function lineScore(bw, axis, pos, start, end, tol = 5) {
  // 指定した位置に罫線がどのくらい通っているか(0〜1)
  const h = bw.rows, w = bw.cols, d = bw.data;
  let hit = 0, total = 0;
  if (axis === "v") {
    const x = Math.floor(pos * w), x0 = Math.max(0, x - tol), x1 = Math.min(w - 1, x + tol);
    for (let y = Math.floor(start * h); y < Math.floor(end * h); y++) {
      total++;
      for (let xx = x0; xx <= x1; xx++) if (d[y * w + xx]) { hit++; break; }
    }
  } else {
    const y = Math.floor(pos * h), y0 = Math.max(0, y - tol), y1 = Math.min(h - 1, y + tol);
    for (let x = Math.floor(start * w); x < Math.floor(end * w); x++) {
      total++;
      for (let yy = y0; yy <= y1; yy++) if (d[yy * w + x]) { hit++; break; }
    }
  }
  return total ? hit / total : 0;
}

function layoutScore(img, quad) {
  // この四角形が受領証の外枠らしいかを、中の罫線の位置で採点する
  const s = new Scope();
  try {
    const small = warp(s, img, quad, FRAME_W / 4, FRAME_H / 4, 0);
    const bw = binarize(s, toGray(s, small));
    const mid = lineScore(bw, "v", MID_X, HEADER_BOTTOM + 0.02, 0.98);
    const header = lineScore(bw, "h", HEADER_BOTTOM, 0.02, 0.98);
    const divider = lineScore(bw, "v", HEADER_DIVIDER_X, 0.02, HEADER_BOTTOM - 0.02);
    const wrong = lineScore(bw, "v", MID_X, 0.02, HEADER_BOTTOM - 0.03); // 上下逆ならここに線が出る
    return mid + header + divider - wrong;
  } finally { s.free(); }
}

function fitLine(points) {
  // 主成分で直線を当てはめる。戻り値は [方向x, 方向y, 通る点x, 通る点y]
  let mx = 0, my = 0;
  for (const p of points) { mx += p[0]; my += p[1]; }
  mx /= points.length; my /= points.length;
  let sxx = 0, sxy = 0, syy = 0;
  for (const p of points) {
    const dx = p[0] - mx, dy = p[1] - my;
    sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
  }
  const theta = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  return [Math.cos(theta), Math.sin(theta), mx, my];
}

function refineQuad(points, quad) {
  // 輪郭の点に直線を当てはめ直して4隅を求める。
  // 右上の「頁」欄のように外枠から少しはみ出した部分に、隅が引っ張られるのを防ぐ。
  const lines = [];
  for (let i = 0; i < 4; i++) {
    const a = quad[i], b = quad[(i + 1) % 4];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (len < 1) return quad;
    const dx = (b[0] - a[0]) / len, dy = (b[1] - a[1]) / len;
    let side = points.filter((p) => {
      const rx = p[0] - a[0], ry = p[1] - a[1];
      const along = rx * dx + ry * dy, dist = Math.abs(rx * dy - ry * dx);
      return along > 0.05 * len && along < 0.95 * len && dist < 0.06 * len;
    });
    if (side.length < 20) return quad;
    let line = fitLine(side);
    for (const tol of [16, 8, 4]) {
      const [vx, vy, x0, y0] = line;
      const keep = side.filter((p) => Math.abs((p[0] - x0) * vy - (p[1] - y0) * vx) < tol);
      if (keep.length < 20) break;
      side = keep;
      line = fitLine(side);
    }
    lines.push(line);
  }
  const corners = [];
  for (let i = 0; i < 4; i++) {
    const [vx1, vy1, x1, y1] = lines[(i + 3) % 4], [vx2, vy2, x2, y2] = lines[i];
    const detm = vx1 * -vy2 - -vx2 * vy1;
    if (Math.abs(detm) < 1e-6) return quad;
    const t = ((x2 - x1) * -vy2 - -vx2 * (y2 - y1)) / detm;
    corners.push([x1 + t * vx1, y1 + t * vy1]);
  }
  return corners;
}

function candidateQuads(img) {
  const s = new Scope();
  try {
    const longer = Math.max(img.rows, img.cols);
    const scale = longer > 1400 ? 1400 / longer : 1;
    let small = img;
    if (scale < 1) {
      small = s.mat();
      cv.resize(img, small, new cv.Size(Math.round(img.cols * scale), Math.round(img.rows * scale)), 0, 0, cv.INTER_AREA);
    }
    const gray = toGray(s, small);
    cv.GaussianBlur(gray, gray, new cv.Size(3, 3), 0);
    const bw = binarize(s, gray);
    const kernel = s.add(cv.Mat.ones(3, 3, cv.CV_8U));
    cv.dilate(bw, bw, kernel);
    const contours = s.add(new cv.MatVector()), hierarchy = s.mat();
    cv.findContours(bw, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_NONE);
    const areaMin = 0.15 * small.rows * small.cols;
    const big = [];
    for (let i = 0; i < contours.size(); i++) {
      const c = contours.get(i);
      const area = cv.contourArea(c);
      if (area >= areaMin) big.push({ c, area }); else c.delete();
    }
    big.sort((a, b) => b.area - a.area);
    const quads = [];
    for (const { c } of big.slice(0, 25)) {
      const approx = new cv.Mat();
      cv.approxPolyDP(c, approx, 0.02 * cv.arcLength(c, true), true);
      if (approx.rows === 4 && cv.isContourConvex(approx)) {
        const a = approx.data32S;
        const rough = orderCorners([[a[0], a[1]], [a[2], a[3]], [a[4], a[5]], [a[6], a[7]]]);
        const d = c.data32S, pts = [];
        for (let k = 0; k < d.length; k += 2) pts.push([d[k], d[k + 1]]);
        quads.push(refineQuad(pts, rough).map((p) => [p[0] / scale, p[1] / scale]));
      }
      approx.delete();
    }
    for (const { c } of big) c.delete();
    return quads;
  } finally { s.free(); }
}

function findFrame(img) {
  // 受領証の外枠の4隅を返す。向き(90度・180度の回転)も合わせる。
  let best = null, bestScore = 1.6; // 3本の罫線のうち、おおむね半分以上が合っていること
  for (const quad of candidateQuads(img)) {
    for (let rot = 0; rot < 4; rot++) {
      const q = quad.slice(rot).concat(quad.slice(0, rot));
      const score = layoutScore(img, q);
      if (score > bestScore) { best = q; bestScore = score; }
    }
  }
  return best;
}

function rowLines(frameGray) {
  // 枠の中の横線の位置を実測する。うまく取れなければ既定の割合を使う。
  const s = new Scope();
  try {
    const h = frameGray.rows, w = frameGray.cols;
    const bw = binarize(s, frameGray);
    const kernel = s.add(cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(Math.floor(w / 3), 1)));
    const lines = s.mat();
    cv.morphologyEx(bw, lines, cv.MORPH_OPEN, kernel);
    const d = lines.data, ys = [];
    for (let y = 0; y < h; y++) {
      let n = 0;
      for (let x = 0; x < w; x++) if (d[y * w + x]) n++;
      if (n > 0.85 * w) ys.push(y);
    }
    if (!ys.length) return ROW_LINES_DEFAULT;
    const groups = [];
    let start = ys[0], prev = ys[0];
    for (const y of ys.slice(1)) {
      if (y - prev > 12) { groups.push((start + prev) / 2); start = y; }
      prev = y;
    }
    groups.push((start + prev) / 2);
    const fr = groups.map((g) => g / h).filter((f) => f > 0.15 && f < 0.985);
    if (!fr.length || !(fr[0] > 0.15 && fr[0] < 0.35)) return ROW_LINES_DEFAULT;
    fr.push(1.0);
    if (fr.length < 3) return ROW_LINES_DEFAULT;
    for (let i = 1; i < fr.length; i++) if (fr[i] - fr[i - 1] < 0.12) return ROW_LINES_DEFAULT;
    return fr;
  } finally { s.free(); }
}

// ---------------------------------------------------------------------------
// 文字認識(行の検出 + 1行ごとの認識)
// ---------------------------------------------------------------------------
const DET_MEAN = [0.485, 0.456, 0.406], DET_STD = [0.229, 0.224, 0.225];

function miniBox(rect) {
  // 回転した長方形の4隅を、左上・右上・右下・左下の順で返す
  const a = (rect.angle * Math.PI) / 180;
  const ux = [(Math.cos(a) * rect.size.width) / 2, (Math.sin(a) * rect.size.width) / 2];
  const uy = [(-Math.sin(a) * rect.size.height) / 2, (Math.cos(a) * rect.size.height) / 2];
  const c = rect.center;
  const pts = [
    [c.x - ux[0] - uy[0], c.y - ux[1] - uy[1]], [c.x + ux[0] - uy[0], c.y + ux[1] - uy[1]],
    [c.x + ux[0] + uy[0], c.y + ux[1] + uy[1]], [c.x - ux[0] + uy[0], c.y - ux[1] + uy[1]],
  ].sort((p, q) => p[0] - q[0]);
  const [l1, l2] = pts[1][1] > pts[0][1] ? [pts[0], pts[1]] : [pts[1], pts[0]]; // 左側の上・下
  const [r1, r2] = pts[3][1] > pts[2][1] ? [pts[2], pts[3]] : [pts[3], pts[2]]; // 右側の上・下
  return [l1, r1, r2, l2];
}

function boxScore(pred, w, h, box) {
  // 四角形の内側の、文字らしさの平均
  const xs = box.map((p) => p[0]), ys = box.map((p) => p[1]);
  const clip = (v, hi) => Math.min(Math.max(v, 0), hi);
  const x0 = clip(Math.floor(Math.min(...xs)), w - 1), x1 = clip(Math.ceil(Math.max(...xs)), w - 1);
  const y0 = clip(Math.floor(Math.min(...ys)), h - 1), y1 = clip(Math.ceil(Math.max(...ys)), h - 1);
  let sum = 0, n = 0;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      let inside = true;
      for (let i = 0; i < 4; i++) {
        const a = box[i], b = box[(i + 1) % 4];
        if ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) < -0.5) { inside = false; break; }
      }
      if (inside) { sum += pred[y * w + x]; n++; }
    }
  }
  return n ? sum / n : 0;
}

async function detect(img) {
  // 画像の中の文字の行を見つけ、行ごとの四角形(4隅)を返す
  const s = new Scope();
  try {
    const h = img.rows, w = img.cols;
    const ratio = Math.min(h, w) < 736 ? 736 / Math.min(h, w) : 1;
    const rh = Math.max(Math.round((h * ratio) / 32) * 32, 32);
    const rw = Math.max(Math.round((w * ratio) / 32) * 32, 32);
    const resized = s.mat();
    cv.resize(img, resized, new cv.Size(rw, rh), 0, 0, cv.INTER_LINEAR);
    const px = resized.data, plane = rh * rw, input = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      for (let c = 0; c < 3; c++) input[c * plane + i] = (px[i * 3 + c] / 255 - DET_MEAN[c]) / DET_STD[c];
    }
    const feeds = { [det.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, rh, rw]) };
    const pred = (await det.run(feeds))[det.outputNames[0]].data;

    const bitmap = s.add(new cv.Mat(rh, rw, cv.CV_8UC1));
    const bd = bitmap.data;
    for (let i = 0; i < plane; i++) bd[i] = pred[i] > 0.3 ? 255 : 0;
    const kernel = s.add(cv.Mat.ones(2, 2, cv.CV_8U));
    cv.dilate(bitmap, bitmap, kernel);
    const contours = s.add(new cv.MatVector()), hierarchy = s.mat();
    cv.findContours(bitmap, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

    const boxes = [];
    for (let i = 0; i < Math.min(contours.size(), 1000); i++) {
      const c = contours.get(i);
      const rect = cv.minAreaRect(c);
      c.delete();
      if (Math.min(rect.size.width, rect.size.height) < 3) continue;
      if (boxScore(pred, rw, rh, miniBox(rect)) < 0.5) continue;
      // 文字の芯だけが検出されるので、外側へ広げる
      const bw = rect.size.width, bh = rect.size.height;
      const grow = (bw * bh * 1.6) / (2 * (bw + bh));
      const wide = { center: rect.center, angle: rect.angle, size: { width: bw + 2 * grow, height: bh + 2 * grow } };
      if (Math.min(wide.size.width, wide.size.height) < 5) continue;
      const pts = miniBox(wide).map((p) => [
        Math.min(Math.max(Math.round((p[0] / rw) * w), 0), w - 1),
        Math.min(Math.max(Math.round((p[1] / rh) * h), 0), h - 1),
      ]);
      // 左2点・右2点をそれぞれ上下で並べ直す
      const sorted = pts.slice().sort((p, q) => p[0] - q[0]);
      const [tl, bl] = sorted.slice(0, 2).sort((p, q) => p[1] - q[1]);
      const [tr, br] = sorted.slice(2).sort((p, q) => p[1] - q[1]);
      const box = [tl, tr, br, bl];
      if (Math.hypot(tl[0] - tr[0], tl[1] - tr[1]) <= 3 || Math.hypot(tl[0] - bl[0], tl[1] - bl[1]) <= 3) continue;
      boxes.push(box);
    }
    return boxes;
  } finally { s.free(); }
}

function cropLine(scope, img, box, pad = 4) {
  // 斜めの行を水平な帯に直して切り出す
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const w = Math.floor(Math.max(dist(box[0], box[1]), dist(box[2], box[3]))) + 2 * pad;
  const h = Math.floor(Math.max(dist(box[0], box[3]), dist(box[1], box[2]))) + 2 * pad;
  const dst = [[pad, pad], [w - pad, pad], [w - pad, h - pad], [pad, h - pad]];
  const m = scope.add(cv.getPerspectiveTransform(quadMat(scope, box), quadMat(scope, dst)));
  const out = scope.mat();
  cv.warpPerspective(img, out, m, new cv.Size(w, h), cv.INTER_LINEAR, cv.BORDER_REPLICATE, new cv.Scalar());
  return out;
}

async function recognize(line) {
  // 1行分の画像を文字にする
  if (!line || line.rows === 0 || line.cols === 0) return "";
  const s = new Scope();
  try {
    const [, , ih, iw] = [1, 3, 24, 384];
    const resized = s.mat();
    cv.resize(line, resized, new cv.Size(iw, ih), 0, 0, cv.INTER_LINEAR);
    const px = resized.data, plane = ih * iw, input = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      for (let c = 0; c < 3; c++) input[c * plane + i] = px[i * 3 + (2 - c)] / 127.5 - 1.0; // BGR → RGB
    }
    const out = (await rec.run({ [rec.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, ih, iw]) }))[rec.outputNames[0]];
    const [, steps, classes] = out.dims, data = out.data;
    let text = "";
    for (let t = 0; t < steps; t++) {
      let best = 0, bestV = -Infinity;
      for (let k = 0; k < classes; k++) {
        const v = data[t * classes + k];
        if (v > bestV) { bestV = v; best = k; }
      }
      if (best === 0) break;
      text += charset[best - 1];
    }
    return text;
  } finally { s.free(); }
}

async function readBox(img, box, pad = 4) {
  const s = new Scope();
  try { return await recognize(cropLine(s, img, box, pad)); } finally { s.free(); }
}

// ---------------------------------------------------------------------------
// 枠ごとの読み取り
// ---------------------------------------------------------------------------
function isNoise(text) {
  // 汚れや点を文字と読んだだけの短い断片か
  const core = text.replace(/[^\p{L}\p{N}]/gu, "");
  return core.length <= 2 && !/[぀-ヿ一-鿿]/.test(core);
}

async function findNumber(cell, band) {
  // 番号を取り出す。検算が合わなければ、切り出し方を変えて読み直す。
  let fallback = "";
  const tryText = (text) => {
    for (const found of nfkc(text).replace(/ /g, "").match(/\d{11}/g) || []) {
      if (checkDigitOk(found)) return found;
      fallback = fallback || found;
    }
    return null;
  };
  for (const { box, text } of band) {
    let hit = tryText(text);
    if (hit) return { number: hit, ok: true };
    if ((nfkc(text).match(/\d/g) || []).length >= 6) {
      for (const pad of [8, 1]) {
        hit = tryText(await readBox(cell, box, pad));
        if (hit) return { number: hit, ok: true };
      }
    }
  }
  return { number: fallback, ok: false };
}

async function readCell(cellSrc, slot) {
  const s = new Scope();
  try {
    const cell = s.mat();
    cv.resize(cellSrc, cell, new cv.Size(Math.round(cellSrc.cols * 1.5), Math.round(cellSrc.rows * 1.5)), 0, 0, cv.INTER_CUBIC);
    const h = cell.rows, w = cell.cols;
    const band = [], address = [];
    let service = "";
    for (const box of await detect(cell)) {
      const cx = box.reduce((a, p) => a + p[0], 0) / 4 / w, cy = box.reduce((a, p) => a + p[1], 0) / 4 / h;
      if (cx > ADDRESS_RIGHT && cy > 0.5) continue; // 「申出損害要償額」と金額は使わないので読まない
      const text = nfkc(await readBox(cell, box)).trim();
      if (cy < NUMBER_BAND_BOTTOM) {
        band.push({ box, text });
      } else if (cx > ADDRESS_RIGHT) {
        for (const word of SERVICE_WORDS) if (!service && text.includes(word)) service = word;
      } else if (cy > ADDRESS_TOP && text && !isNoise(text)) {
        address.push({ y: cy, text });
      }
    }
    const { number, ok } = await findNumber(cell, band);
    address.sort((a, b) => a.y - b.y);
    const texts = address.map((a) => a.text);
    if (!number) return { item: null, unreadable: texts.length > 0 };
    let postal = "";
    for (const t of texts) {
      const m = t.match(/(\d{3})-?(\d{4})/);
      if (m && t.replace(/\D/g, "").length <= 8) { postal = `${m[1]}-${m[2]}`; break; }
    }
    return {
      item: {
        slot, number, checkOk: ok, service, postal,
        recipient: texts.length ? texts[texts.length - 1] : "",
        address: texts.join(" "),
      },
      unreadable: false,
    };
  } finally { s.free(); }
}

async function readIssueDate(warped) {
  // 左下の「2020/10/17 9:12:39」のような発行日時から日付を取る
  const s = new Scope();
  try {
    const y0 = MARGIN + FRAME_H + 8, x0 = MARGIN - 20;
    const roi = s.add(warped.roi(new cv.Rect(x0, y0, Math.floor(FRAME_W * 0.7) + 20, warped.rows - y0)));
    const strip = s.mat();
    cv.resize(roi, strip, new cv.Size(Math.round(roi.cols * 1.5), Math.round(roi.rows * 1.5)), 0, 0, cv.INTER_CUBIC);
    for (const box of await detect(strip)) {
      const m = nfkc(await readBox(strip, box)).match(/(20\d{2})[/\-.年](\d{1,2})[/\-.月](\d{1,2})/);
      if (!m) continue;
      const [y, mo, d] = [Number(m[1]), Number(m[2]), Number(m[3])];
      const dt = new Date(y, mo - 1, d);
      if (dt.getFullYear() === y && dt.getMonth() === mo - 1 && dt.getDate() === d) {
        return `${y}-${String(mo).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
      }
    }
    return null;
  } finally { s.free(); }
}

/**
 * 受領証の写真を読む。
 * @param {ImageData} imageData 写真の画素データ
 * @returns {Promise<{ok:boolean, reason?:string, issued?:string|null, items?:object[], unreadableSlots?:number[]}>}
 */
export async function readReceipt(imageData, onProgress = () => {}) {
  await init(onProgress);
  const s = new Scope();
  try {
    const rgba = s.add(cv.matFromImageData(imageData));
    const img = s.mat();
    cv.cvtColor(rgba, img, cv.COLOR_RGBA2BGR);
    onProgress("受領証の枠を探しています");
    const quad = findFrame(img);
    if (!quad) {
      return { ok: false, reason: "受領証の枠を見つけられませんでした。紙全体が写るように、明るい場所で撮り直してください。" };
    }
    const warped = warp(s, img, quad, FRAME_W, FRAME_H, MARGIN);
    const frame = s.add(warped.roi(new cv.Rect(MARGIN, MARGIN, FRAME_W, FRAME_H)));
    const rows = rowLines(toGray(s, frame));
    onProgress("発行日を読んでいます");
    const issued = await readIssueDate(warped);

    const items = [], unreadableSlots = [];
    const nRows = rows.length - 1, inset = 8;
    for (let col = 0; col < 2; col++) {
      const x0 = Math.floor(col * MID_X * FRAME_W), x1 = Math.floor((col + 1) * MID_X * FRAME_W);
      for (let row = 0; row < nRows; row++) {
        const y0 = Math.floor(rows[row] * FRAME_H), y1 = Math.floor(rows[row + 1] * FRAME_H);
        const slot = col * nRows + row + 1;
        onProgress(`${slot}番の枠を読んでいます(全${nRows * 2}枠)`);
        const cell = frame.roi(new cv.Rect(x0 + inset, y0 + inset, x1 - x0 - 2 * inset, y1 - y0 - 2 * inset));
        try {
          const { item, unreadable } = await readCell(cell, slot);
          if (item) items.push(item); else if (unreadable) unreadableSlots.push(slot);
        } finally { cell.delete(); }
      }
    }
    if (!items.length && !unreadableSlots.length) {
      return { ok: false, reason: "お問い合わせ番号を1件も読めませんでした。ピントと明るさを確かめて撮り直してください。" };
    }
    return { ok: true, issued, items, unreadableSlots };
  } finally { s.free(); }
}
