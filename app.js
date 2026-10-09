// 画面と記録の保存。記録はこの端末のブラウザ内(IndexedDB)にだけ保存する。
// このファイルにも、外部へデータを送る処理は無い。

/* global XLSX */
import { init, readReceipt } from "./reader.js";

const TRACK_URL = "https://trackings.post.japanpost.jp/services/srv/search/direct";
const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// 追跡リンク
// ---------------------------------------------------------------------------
export function trackingUrl(numbers) {
  // 日本郵便の個別番号検索の結果ページが直接開く。含まれるのは番号だけ。
  const q = numbers.slice(0, 10).map((n, i) => `reqCodeNo${i + 1}=${n}`).join("&");
  return `${TRACK_URL}?${q}&searchKind=S002&locale=ja`;
}

// ---------------------------------------------------------------------------
// 記録の保存(IndexedDB)。郵便1通につき1件、お問い合わせ番号で重複を防ぐ。
// ---------------------------------------------------------------------------
let dbPromise = null;
function db() {
  dbPromise ||= new Promise((resolve, reject) => {
    const req = indexedDB.open("kakitome", 1);
    req.onupgradeneeded = () => req.result.createObjectStore("items", { keyPath: "number" });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

async function store(mode, fn) {
  const d = await db();
  return new Promise((resolve, reject) => {
    const tx = d.transaction("items", mode);
    const result = fn(tx.objectStore("items"));
    tx.oncomplete = () => resolve(result && "result" in result ? result.result : undefined);
    tx.onerror = tx.onabort = () => reject(tx.error);
  });
}

export const allItems = () => store("readonly", (s) => s.getAll());
const putItem = (item) => store("readwrite", (s) => s.put(item));
const deleteItem = (number) => store("readwrite", (s) => s.delete(number));
const clearItems = () => store("readwrite", (s) => s.clear());

const today = () => {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
};
const daysAgo = (n) => {
  const d = new Date();
  d.setDate(d.getDate() - n);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
};

function retentionDays() {
  try {
    const v = localStorage.getItem("retentionDays");
    return v === null ? 100 : Math.max(0, Number(v) || 0);
  } catch (e) { return 100; }
}

async function purgeExpired() {
  const days = retentionDays();
  if (!days) return 0;
  const limit = daysAgo(days);
  const old = (await allItems()).filter((it) => it.date < limit);
  for (const it of old) await deleteItem(it.number);
  return old.length;
}

async function addReceipt(receipt) {
  const date = receipt.issued || today();
  const days = retentionDays();
  if (days && date < daysAgo(days)) return { expired: true, date, added: [], duplicates: [] };
  const known = new Set((await allItems()).map((it) => it.number));
  const added = [], duplicates = [];
  for (const it of receipt.items) {
    if (known.has(it.number)) { duplicates.push(it); continue; }
    known.add(it.number);
    const row = {
      number: it.number, date, recipient: it.recipient, address: it.address, postal: it.postal,
      checkOk: it.checkOk, service: it.service, created: new Date().toISOString(),
    };
    await putItem(row);
    added.push(row);
  }
  return { expired: false, date, added, duplicates };
}

// ---------------------------------------------------------------------------
// 検索
// ---------------------------------------------------------------------------
function normalize(text) {
  // 表記のゆれをそろえる(全角半角、空白、ハイフンの種類)
  return String(text || "").normalize("NFKC").toLowerCase()
    .replace(/[\s　]/g, "").replace(/[‐‑‒–—―ー−-]/g, "-");
}

function similarity(query, text) {
  // 誤読が少し混ざっていても当たるよう、最も似ている部分との一致度(0〜1)を返す
  const n = query.length;
  if (n < 3 || text.length < n) return 0;
  let best = 0;
  for (let i = 0; i + n <= text.length && best < 1; i++) {
    const win = text.slice(i, i + n);
    let prev = new Array(n + 1).fill(0);
    for (let a = 1; a <= n; a++) {
      const cur = [0];
      for (let b = 1; b <= n; b++) cur[b] = query[a - 1] === win[b - 1] ? prev[b - 1] + 1 : Math.max(prev[b], cur[b - 1]);
      prev = cur;
    }
    best = Math.max(best, prev[n] / n);
  }
  return best;
}

export function searchItems(items, query, limit = 30) {
  const q = normalize(query).replace(/(さん|様|さま|殿|御中|の件)$/, "");
  if (!q) return [];
  const digits = q.replace(/\D/g, "");
  const scored = [];
  for (const it of items) {
    const hay = normalize(`${it.recipient}${it.address}${it.postal}`);
    let score;
    if (hay.includes(q) || (digits.length >= 4 && it.number.includes(digits)) ||
        (digits.length === 7 && hay.replace(/-/g, "").includes(digits))) score = 1;
    else score = similarity(q, hay);
    if (score >= 0.67) scored.push({ score, it });
  }
  const exact = scored.filter((s) => s.score === 1);
  const chosen = exact.length ? exact : scored; // 完全に当たる行があれば、あいまい一致は出さない
  chosen.sort((a, b) => b.score - a.score || (a.it.date < b.it.date ? 1 : -1));
  return chosen.slice(0, limit).map((s) => s.it);
}

// ---------------------------------------------------------------------------
// 写真の取り込み。ファイルには保存せず、読み取りが終わったら捨てる。
// ---------------------------------------------------------------------------
async function fileToImageData(file) {
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    const scale = Math.min(1, 2600 / Math.max(img.naturalWidth, img.naturalHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(img.naturalWidth * scale);
    canvas.height = Math.round(img.naturalHeight * scale);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    canvas.width = canvas.height = 0; // 画素データを早めに手放す
    return data;
  } finally {
    URL.revokeObjectURL(url);
  }
}

// ---------------------------------------------------------------------------
// 画面
// ---------------------------------------------------------------------------
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v; else if (k.startsWith("on")) node.addEventListener(k.slice(2), v); else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

function setStatus(text, kind = "") {
  const box = $("status");
  box.hidden = !text;
  box.className = kind === "error" ? "error" : "";
  box.replaceChildren(text || "");
  if (kind === "busy") box.append(el("div", { class: "bar" }, el("i")));
}

function formatDate(iso) {
  const [y, m, d] = iso.split("-").map(Number);
  return `${y}年${m}月${d}日`;
}

function itemCard(it) {
  const card = el("div", { class: `item${it.checkOk ? "" : " warn"}` });
  card.append(
    el("div", { class: "name" }, it.recipient || "(宛名を読めませんでした)"),
    el("div", { class: "meta" }, `${formatDate(it.date)} 提出${it.service ? "・" + it.service : ""}`),
    el("div", { class: "addr" }, it.address),
  );
  const num = el("div", { class: "num" }, it.number);
  if (!it.checkOk) num.append(el("span", { class: "flag" }, "番号要確認"));
  card.append(num);
  card.append(el("div", { class: "actions" },
    el("a", { class: "track", href: trackingUrl([it.number]), target: "_blank", rel: "noopener noreferrer" }, "配達状況を開く"),
    el("button", { class: "ghost", onclick: () => editName(it) }, "宛名を直す"),
    el("button", { class: "ghost danger", onclick: () => removeItem(it) }, "消す"),
  ));
  return card;
}

async function editName(it) {
  const name = prompt("宛名を修正", it.recipient);
  if (name === null) return;
  // 宛先全文の中の宛名も入れ替えて、検索に当たるようにする
  const address = it.recipient && it.address.includes(it.recipient)
    ? it.address.replace(it.recipient, name.trim()) : `${it.address} ${name.trim()}`.trim();
  await putItem({ ...it, recipient: name.trim(), address });
  render();
}

async function removeItem(it) {
  if (!confirm(`${it.recipient || it.number} を消しますか?`)) return;
  await deleteItem(it.number);
  render();
}

async function render() {
  const items = await allItems();
  $("count").textContent = `${items.length}件`;
  const query = $("search").value.trim();
  let rows, title;
  if (query) {
    rows = searchItems(items, query);
    title = rows.length ? `${rows.length}件 見つかりました` : "";
  } else {
    rows = items.slice().sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : a.created < b.created ? 1 : -1)).slice(0, 30);
    title = rows.length ? "最近の登録" : "";
  }
  $("listTitle").textContent = title;
  const list = $("list");
  list.replaceChildren(...rows.map(itemCard));
  if (!rows.length) {
    list.append(el("div", { class: "empty" }, query
      ? "見つかりませんでした。名前の一部・町名・郵便番号でも試してください。それでもなければ紙の受領証を確認してください。"
      : "まだ登録がありません。受領証を撮ると、ここに並びます。"));
  }
}

function showRegistered(receipt, result) {
  const box = $("registered");
  box.replaceChildren();
  if (result.expired) {
    setStatus(`発行日が ${formatDate(result.date)} で、記録を残す日数を過ぎているため登録しませんでした。`, "error");
    return;
  }
  const notes = [];
  if (result.added.length) notes.push(`${formatDate(result.date)} 提出分として ${result.added.length}件 登録しました。`);
  if (result.duplicates.length) notes.push(`登録済みのため飛ばした郵便: ${result.duplicates.length}件`);
  if (receipt.unreadableSlots.length) notes.push(`${receipt.unreadableSlots.join("・")}番の枠は番号を読めなかったので登録していません。撮り直してください。`);
  if (!receipt.issued && result.added.length) notes.push("発行日を読めなかったので、提出日は今日の日付にしています。");
  setStatus(notes.join(" "), receipt.unreadableSlots.length ? "error" : "");
  const numbers = (result.added.length ? result.added : result.duplicates).map((it) => it.number);
  if (numbers.length) {
    box.append(el("a", { class: "track all", href: trackingUrl(numbers), target: "_blank", rel: "noopener noreferrer" },
      `この受領証 ${numbers.length}件の配達状況を開く`));
  }
}

let busy = false;
async function handleFile(file) {
  if (!file || busy) return;
  busy = true;
  $("shoot").disabled = true;
  $("registered").replaceChildren();
  try {
    setStatus("写真を開いています", "busy");
    let imageData = await fileToImageData(file);
    const receipt = await readReceipt(imageData, (msg) => setStatus(msg, "busy"));
    imageData = null; // 写真はここで捨てる
    if (!receipt.ok) { setStatus(receipt.reason, "error"); return; }
    const result = await addReceipt(receipt);
    $("search").value = "";
    showRegistered(receipt, result);
    await render();
  } catch (e) {
    console.error(e);
    setStatus("読み取り中にエラーが起きました。もう一度撮ってください。", "error");
  } finally {
    busy = false;
    $("shoot").disabled = false;
    $("camera").value = "";
    $("library").value = "";
  }
}

async function exportXlsx() {
  const items = (await allItems()).sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  const header = ["提出日", "宛名", "宛先(読み取り全文)", "郵便番号", "お問い合わせ番号", "検算", "種別", "追跡リンク"];
  const sheet = XLSX.utils.aoa_to_sheet([header, ...items.map((it) => [
    it.date, it.recipient, it.address, it.postal, it.number, it.checkOk ? "OK" : "要確認", it.service, "追跡を開く",
  ])]);
  items.forEach((it, i) => {
    sheet[XLSX.utils.encode_cell({ r: i + 1, c: 4 })].t = "s"; // 番号を数値にしない
    sheet[XLSX.utils.encode_cell({ r: i + 1, c: 7 })].l = { Target: trackingUrl([it.number]) };
  });
  sheet["!cols"] = [12, 26, 58, 11, 17, 8, 10, 12].map((wch) => ({ wch }));
  const book = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(book, sheet, "記録");
  const bytes = XLSX.write(book, { bookType: "xlsx", type: "array" });
  const file = new File([bytes], `書留の記録_${today()}.xlsx`, {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });
  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    try { await navigator.share({ files: [file] }); } catch (e) { /* 取り消した場合 */ }
    return;
  }
  const url = URL.createObjectURL(file);
  el("a", { href: url, download: file.name }).click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

async function start() {
  $("shoot").addEventListener("click", () => $("camera").click());
  $("pick").addEventListener("click", () => $("library").click());
  $("camera").addEventListener("change", (e) => handleFile(e.target.files[0]));
  $("library").addEventListener("change", (e) => handleFile(e.target.files[0]));
  $("search").addEventListener("input", render);
  $("export").addEventListener("click", exportXlsx);
  $("retention").value = retentionDays();
  $("retention").addEventListener("change", async (e) => {
    try { localStorage.setItem("retentionDays", String(Math.max(0, Number(e.target.value) || 0))); } catch (err) { /* 保存できない環境 */ }
    await purgeExpired();
    render();
  });
  $("wipe").addEventListener("click", async () => {
    if (!confirm("記録をすべて消します。元に戻せません。よろしいですか?")) return;
    await clearItems();
    $("registered").replaceChildren();
    setStatus("");
    render();
  });
  $("about").textContent = "読み取りも検索もこのiPhoneの中で行い、写真は保存しません。外部に送られるのは、配達状況を開いたときのお問い合わせ番号だけです。";

  if (navigator.storage && navigator.storage.persist) navigator.storage.persist().catch(() => {});
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
  await purgeExpired();
  await render();
  // 最初の撮影で待たせないよう、裏でモデルを読み込んでおく
  init().catch((e) => { console.error(e); setStatus("読み取りの部品を読み込めませんでした。通信できる場所で開き直してください。", "error"); });
}

window.kakitome = { handleFile, allItems, searchItems, render }; // 動作確認用
start();
