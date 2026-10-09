// 一度開いたあとは、電波のない場所でも使えるように部品を端末に保存しておく。
// 保存するのはプログラムとモデルだけで、台帳や写真は扱わない。
const VERSION = "v2";
const FILES = [
  "./", "index.html", "app.js", "reader.js", "manifest.webmanifest", "icon-180.png", "icon-512.png",
  "opencv.js", "ort.wasm.min.js", "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm", "xlsx.mini.min.js",
  "det.onnx", "rec.onnx.part0", "rec.onnx.part1", "charset.json",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(VERSION).then((cache) => cache.addAll(FILES)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET" || url.origin !== self.location.origin) return;
  event.respondWith(
    caches.match(event.request, { ignoreSearch: true }).then((hit) => hit || fetch(event.request)),
  );
});
