import Foundation
import CoreGraphics
import ScreenCaptureKit

// Нативный детектор на Swift (замена detect.py). Снимок экрана берётся В ПАМЯТЬ
// через CGDisplayCreateImage — без screencapture, без PNG, без диска. Тот же
// алгоритм (сетка/медиана/кубики/преобладание/5-я точка) и тот же HTTP-протокол
// (/scan, /points long-poll, /health), поэтому overlay и button не меняются.

// ---- параметры (env, дефолты как в detect.py) ----
func envInt(_ k: String, _ d: Int) -> Int {
    if let v = ProcessInfo.processInfo.environment[k], let i = Int(v) { return i }
    return d
}
let PORT = UInt16(envInt("SG_PORT", 8133))
let STEP_DEF = envInt("SG_STEP", 60)
let NDOWN_DEF = envInt("SG_NDOWN", 18)
let PREDOM_DEF = envInt("SG_PREDOM", 3)
let PATCH = envInt("SG_PATCH", 6)
let START_X = envInt("SG_START_X", 20)
let START_Y = envInt("SG_START_Y", 20)

// ---- кадр в памяти (RGBA8, row 0 = верх экрана) ----
final class Frame {
    let w: Int, h: Int, bpr: Int
    let buf: UnsafeMutablePointer<UInt8>
    init(w: Int, h: Int, bpr: Int, buf: UnsafeMutablePointer<UInt8>) {
        self.w = w; self.h = h; self.bpr = bpr; self.buf = buf
    }
    deinit { buf.deallocate() }
    @inline(__always) func rgb(_ x: Int, _ y: Int) -> (Int, Int, Int) {
        let o = y * bpr + x * 4
        return (Int(buf[o]), Int(buf[o + 1]), Int(buf[o + 2]))   // R,G,B (A в o+3)
    }
}

// Снимок главного дисплея В ПАМЯТЬ через ScreenCaptureKit (CGDisplayCreateImage
// удалён в macOS 15). Фильтр/конфиг кэшируем (SCShareableContent перечисляет
// все окна — дорого), пере-снимаем только сам кадр. Async мостим в sync семафором.
@available(macOS 14.0, *)
final class Capturer {
    static let shared = Capturer()
    private var filter: SCContentFilter?
    private var cfg: SCStreamConfiguration?

    private func prepare() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        let did = CGMainDisplayID()
        guard let disp = content.displays.first(where: { $0.displayID == did }) ?? content.displays.first else { return }
        let f = SCContentFilter(display: disp, excludingWindows: [])
        let c = SCStreamConfiguration()
        if let mode = CGDisplayCopyDisplayMode(did) {       // полное пиксельное разрешение (Retina)
            c.width = mode.pixelWidth; c.height = mode.pixelHeight
        } else {
            c.width = disp.width; c.height = disp.height
        }
        c.showsCursor = false
        filter = f; cfg = c
    }

    func capture() -> CGImage? {
        let sem = DispatchSemaphore(value: 0)
        var out: CGImage?
        Task {
            defer { sem.signal() }
            do {
                if filter == nil || cfg == nil { try await prepare() }
                if let f = filter, let c = cfg {
                    out = try await SCScreenshotManager.captureImage(contentFilter: f, configuration: c)
                }
            } catch {
                FileHandle.standardError.write("[err] capture: \(error)\n".data(using: .utf8)!)
            }
        }
        sem.wait()
        return out
    }
}

func grabScreen() -> Frame? {
    guard #available(macOS 14.0, *), let img = Capturer.shared.capture() else { return nil }
    let w = img.width, h = img.height, bpr = w * 4
    let buf = UnsafeMutablePointer<UInt8>.allocate(capacity: bpr * h)
    let cs = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(data: buf, width: w, height: h, bitsPerComponent: 8,
                              bytesPerRow: bpr, space: cs,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
        buf.deallocate(); return nil
    }
    // CGImage из ScreenCaptureKit уже top-left — рисуем БЕЗ флипа, тогда row 0 =
    // верх экрана (как cv2/screencapture). Флип зеркалил y -> цвета не совпадали с отрисовкой.
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    return Frame(w: w, h: h, bpr: bpr, buf: buf)
}

// ---- цвет ----
func medianColor(_ f: Frame, _ x: Double, _ y: Double) -> (Int, Int, Int) {
    let hp = PATCH / 2
    // клампим точку в пределы кадра — иначе патч за экраном пуст и median падает
    let xi = min(max(0, Int(x.rounded())), f.w - 1)
    let yi = min(max(0, Int(y.rounded())), f.h - 1)
    let x0 = max(0, xi - hp), x1 = min(f.w - 1, xi + hp)
    let y0 = max(0, yi - hp), y1 = min(f.h - 1, yi + hp)
    var rs = [Int](), gs = [Int](), bs = [Int]()
    var yy = y0
    while yy <= y1 {
        var xx = x0
        while xx <= x1 {
            let (r, g, b) = f.rgb(xx, yy)
            rs.append(r); gs.append(g); bs.append(b)
            xx += 1
        }
        yy += 1
    }
    func med(_ a: [Int]) -> Int {
        if a.isEmpty { return 0 }
        let s = a.sorted(); let n = s.count
        return n % 2 == 1 ? s[n / 2] : Int((Double(s[n / 2 - 1] + s[n / 2]) / 2.0).rounded())
    }
    return (med(rs), med(gs), med(bs))
}

func hexOf(_ r: Int, _ g: Int, _ b: Int) -> String { String(format: "%02x%02x%02x", r, g, b) }

// ---- преобладание ----
func predominant(_ cols: [String], _ minCount: Int) -> String? {
    var cnt = [String: Int]()
    for c in cols { cnt[c, default: 0] += 1 }
    var top = "", n = 0
    for (c, v) in cnt where v > n { n = v; top = c }
    if n < minCount { return nil }
    var atMax = 0
    for (_, v) in cnt where v == n { atMax += 1 }
    return atMax > 1 ? nil : top          // ничья за максимум -> нет преобладания
}

// ---- случайный яркий цвет на ключ (стабильный: FNV-хеш ключа -> оттенок) ----
func keyColor(_ key: String) -> String {
    var h: UInt64 = 1469598103934665603
    for byte in key.utf8 { h = (h ^ UInt64(byte)) &* 1099511628211 }
    let hue = Double(h % 100000) / 100000.0
    let i = Int(hue * 6.0), f = hue * 6.0 - Double(Int(hue * 6.0))
    let s = 0.85, v = 1.0
    let p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s)
    var r = 0.0, g = 0.0, b = 0.0
    switch i % 6 {
    case 0: r = v; g = t; b = p
    case 1: r = q; g = v; b = p
    case 2: r = p; g = v; b = t
    case 3: r = p; g = q; b = v
    case 4: r = t; g = p; b = v
    default: r = v; g = p; b = q
    }
    return hexOf(Int(r * 255), Int(g * 255), Int(b * 255))
}

@inline(__always) func r4(_ v: Double) -> Double { (v * 10000).rounded() / 10000 }

// ---- детект (порт detect.py) ----
func detectFrame(_ f: Frame, step: Int, predom: Int, ndown ndownIn: Int, useMid: Bool, full: Bool) -> [String: Any] {
    let W = Double(f.w), H = Double(f.h)
    let sx = Double(START_X), sy = Double(START_Y)
    var xs = [Double](); var x = sx
    while x < W { xs.append(x); x += Double(step) }
    var ndown = ndownIn
    if full {                                   // «до конца экрана»: сколько точек по высоте влезает
        var c = 0; var yy = sy
        while yy < H { c += 1; yy += Double(step) }
        ndown = max(2, c)
    }
    let rows = (0..<ndown).map { sy + Double($0) * Double(step) }

    var points = [[Double]](), colors = [String](), kinds = [String](), numbers = [Int]()
    for xi in xs {                                   // столбец за столбцом, сверху вниз
        for yj in rows {
            let (r, g, b) = medianColor(f, xi, yj)
            points.append([r4(xi / W), r4(yj / H)])
            colors.append(hexOf(r, g, b))
        }
    }
    kinds = Array(repeating: "base", count: points.count)
    if !kinds.isEmpty { kinds[0] = "seed" }
    numbers = Array(0..<points.count)

    let ncols = xs.count
    let nrows = max(0, ndown - 1), ncubes = max(0, ncols - 1)
    var nextNum = points.count
    var blocks = [String: [[Double]]]()
    @inline(__always) func corners(_ k: Int, _ i: Int) -> [Int] {
        [i * ndown + k, i * ndown + k + 1, (i + 1) * ndown + k, (i + 1) * ndown + k + 1]
    }

    // ЭТАП 1: цвет каждого кубика (k,i) -> в сетку grid (ключ = позиция квадрата,
    // для быстрого поиска соседей на этапе 2). nil = не закрашен.
    var grid = [[String?]](repeating: [String?](repeating: nil, count: ncubes), count: nrows)
    for k in 0..<nrows {
        let cyk = sy + (Double(k) + 0.5) * Double(step)
        for i in 0..<ncubes {
            let idx = corners(k, i)
            var cornerCols = idx.map { colors[$0] }
            var blockPts = idx.map { points[$0] }
            var key = predominant(cornerCols, predom)
            if key == nil && useMid {                 // нет преобладания у 4 углов -> 5-я точка в центре (опц.)
                let mcx = (xs[i] + xs[i + 1]) / 2.0
                let (r, g, b) = medianColor(f, mcx, cyk)
                let mcol = hexOf(r, g, b)
                let mpt = [r4(mcx / W), r4(cyk / H)]
                points.append(mpt); colors.append(mcol)
                kinds.append("mid"); numbers.append(nextNum); nextNum += 1
                cornerCols.append(mcol); blockPts.append(mpt)
                key = predominant(cornerCols, predom)
            }
            if let kk = key {
                grid[k][i] = kk
                blocks[kk, default: []].append(contentsOf: blockPts)
            }
        }
    }

    // ЭТАП 2: пустой кубик закрашиваем, если БЛИЖАЙШИЕ закрашенные соседи слева и
    // справа в том же ряду — одного цвета (это тот же блок с провалом на тексте).
    // Соседей ищем по grid этапа 1 (без цепной заливки) -> один проход.
    var filled = grid
    for k in 0..<nrows {
        for i in 0..<ncubes where grid[k][i] == nil {
            var l = i - 1; while l >= 0 && grid[k][l] == nil { l -= 1 }
            var r = i + 1; while r < ncubes && grid[k][r] == nil { r += 1 }
            if l >= 0, r < ncubes, let lc = grid[k][l], let rc = grid[k][r], lc == rc {
                filled[k][i] = lc
                blocks[lc, default: []].append(contentsOf: corners(k, i).map { points[$0] })
            }
        }
    }

    // отдаём все закрашенные кубики (этап 1 + этап 2)
    var cubes = [[Double]](), cubeFills = [String]()
    for k in 0..<nrows {
        let yt = r4((sy + Double(k) * Double(step)) / H)
        let yb = r4((sy + Double(k + 1) * Double(step)) / H)
        for i in 0..<ncubes {
            guard let c = filled[k][i] else { continue }
            cubes.append([r4(xs[i] / W), yt, r4(xs[i + 1] / W), yb])
            cubeFills.append(keyColor(c))
        }
    }
    return ["points": points, "colors": colors, "kinds": kinds, "numbers": numbers,
            "lines": [], "cubes": cubes, "cube_fills": cubeFills, "blocks": blocks,
            "count": points.count]
}

// ---- состояние + long-poll ----
let cond = NSCondition()
var latest: [String: Any] = ["points": [], "colors": [], "kinds": [], "numbers": [],
                             "lines": [], "cubes": [], "cube_fills": [], "blocks": [String: Any](),
                             "count": 0, "v": 0, "ms": 0.0, "show_numbers": false]
var ver = 0

func doScan(step: Int, predom: Int, ndown: Int, useMid: Bool, full: Bool, showNumbers: Bool) -> [String: Any]? {
    let t0 = Date()
    guard let f = grabScreen() else { return nil }
    var res = detectFrame(f, step: step, predom: predom, ndown: ndown, useMid: useMid, full: full)
    let ms = Date().timeIntervalSince(t0) * 1000
    cond.lock()
    ver += 1
    res["v"] = ver
    res["ms"] = (ms * 10).rounded() / 10
    res["show_numbers"] = showNumbers
    latest = res
    cond.broadcast()
    cond.unlock()
    let nc = (res["cubes"] as? [[Double]])?.count ?? 0
    FileHandle.standardError.write("[scan] \(res["count"] ?? 0) точек, \(nc) кубиков за \(Int(ms))мс\n".data(using: .utf8)!)
    return res
}

// ---- крошечный HTTP-сервер (сокеты + поток на соединение, как в Python) ----
func writeAll(_ fd: Int32, _ data: Data) {
    data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
        guard var p = raw.baseAddress else { return }
        var rem = data.count
        while rem > 0 {
            let n = write(fd, p, rem)
            if n <= 0 { break }
            p = p.advanced(by: n); rem -= n
        }
    }
}

func sendJSON(_ fd: Int32, _ code: Int, _ obj: [String: Any]) {
    let body = (try? JSONSerialization.data(withJSONObject: obj)) ?? Data("{}".utf8)
    let head = "HTTP/1.1 \(code) OK\r\nContent-Type: application/json\r\nContent-Length: \(body.count)\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"
    var out = Data(head.utf8); out.append(body)
    writeAll(fd, out)
    close(fd)
}

func parseQuery(_ q: String) -> [String: String] {
    var d = [String: String]()
    for pair in q.split(separator: "&") {
        let kv = pair.split(separator: "=", maxSplits: 1)
        if kv.count == 2 { d[String(kv[0])] = String(kv[1]) }
    }
    return d
}

func handleConn(_ fd: Int32) {
    var buf = [UInt8](repeating: 0, count: 8192)
    let n = read(fd, &buf, buf.count)
    if n <= 0 { close(fd); return }
    let req = String(decoding: buf[0..<n], as: UTF8.self)
    guard let line = req.split(separator: "\r\n").first else { close(fd); return }
    let parts = line.split(separator: " ")
    guard parts.count >= 2 else { close(fd); return }
    let target = String(parts[1])
    let comps = target.split(separator: "?", maxSplits: 1)
    let path = String(comps[0])
    let qs = comps.count > 1 ? parseQuery(String(comps[1])) : [:]

    switch path {
    case "/health":
        cond.lock(); let c = latest["count"] ?? 0; cond.unlock()
        sendJSON(fd, 200, ["ok": true, "count": c])
    case "/scan":
        let step = Int(qs["step"] ?? "") ?? STEP_DEF
        let predom = Int(qs["predom"] ?? "") ?? PREDOM_DEF
        let ndownRaw = Int(qs["ndown"] ?? "") ?? NDOWN_DEF
        let ndown = ndownRaw >= 2 ? ndownRaw : NDOWN_DEF
        let useMid = (qs["mid"] ?? "0") == "1"        // 5-я точка опциональна (по умолчанию выкл)
        let full = (qs["full"] ?? "1") == "1"         // «до конца экрана» (по умолчанию вкл)
        let labels = (qs["labels"] ?? "0") == "1"     // показывать номера точек (по умолчанию выкл)
        if let r = doScan(step: step, predom: predom, ndown: ndown, useMid: useMid, full: full, showNumbers: labels) {
            sendJSON(fd, 200, ["ok": true, "count": r["count"]!, "v": r["v"]!,
                               "numbers": r["numbers"]!, "colors": r["colors"]!])
        } else {
            sendJSON(fd, 500, ["error": "capture failed"])
        }
    case "/points", "/":
        let since = Int(qs["since"] ?? "0") ?? 0
        cond.lock()
        if ver <= since { _ = cond.wait(until: Date().addingTimeInterval(25)) }
        let snap = latest
        cond.unlock()
        sendJSON(fd, 200, snap)
    default:
        sendJSON(fd, 404, ["error": "use /scan, /points или /health"])
    }
}

func startServer() {
    signal(SIGPIPE, SIG_IGN)                          // не падать при записи в закрытый сокет
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    var yes: Int32 = 1
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
    var addr = sockaddr_in()
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = PORT.bigEndian
    addr.sin_addr.s_addr = inet_addr("127.0.0.1")
    let bindRes = withUnsafePointer(to: &addr) { p in
        p.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
        }
    }
    guard bindRes == 0 else {
        FileHandle.standardError.write("bind :\(PORT) failed\n".data(using: .utf8)!)
        exit(1)
    }
    listen(fd, 32)
    FileHandle.standardError.write("[server] слушаю http://127.0.0.1:\(PORT) — детект ручной, по GET /scan\n".data(using: .utf8)!)
    while true {
        let cfd = accept(fd, nil, nil)
        if cfd < 0 { continue }
        Thread.detachNewThread { handleConn(cfd) }
    }
}

startServer()
