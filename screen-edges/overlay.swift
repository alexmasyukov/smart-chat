import AppKit
import Foundation

// =============================================================================
// screen-edges overlay — прозрачный click-through оверлей поверх рабочего стола.
// Опрашивает HTTP-детектор (detect.py, /edges) и рисует поверх экрана зелёные
// линии по найденным граням: H (горизонтальные «полки») — ярко-зелёные,
// V (вертикальные «стены») — жёлто-зелёные. Это визуализация того, по чему
// потом будет ходить/лезть кот; сам оверлей мышь НЕ перехватывает.
//
//   EDGES_URL=http://127.0.0.1:8130/edges POLL=1.0 ./overlay
// =============================================================================

struct Seg {
    let x1, y1, x2, y2: CGFloat   // нормализованные 0..1, y сверху вниз
    let horizontal: Bool
}

// MARK: - Клиент детектора (поллинг /edges)

final class EdgesClient {
    private let url: URL
    private let poll: TimeInterval
    var onSegments: (([Seg]) -> Void)?

    init() {
        let base = ProcessInfo.processInfo.environment["EDGES_URL"]
            ?? "http://127.0.0.1:8130/edges"
        url = URL(string: base)!
        poll = TimeInterval(ProcessInfo.processInfo.environment["POLL"] ?? "1.0") ?? 1.0
    }

    func start() {
        Timer.scheduledTimer(withTimeInterval: poll, repeats: true) { [weak self] _ in
            self?.fetch()
        }
        fetch()
    }

    private func fetch() {
        var req = URLRequest(url: url)
        req.timeoutInterval = 3
        URLSession.shared.dataTask(with: req) { [weak self] data, _, _ in
            guard let data,
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let arr = obj["segments"] as? [[String: Any]] else { return }
            let segs: [Seg] = arr.compactMap { s in
                guard let x1 = s["x1"] as? Double, let y1 = s["y1"] as? Double,
                      let x2 = s["x2"] as? Double, let y2 = s["y2"] as? Double
                else { return nil }
                let o = (s["o"] as? String) ?? "H"
                return Seg(x1: CGFloat(x1), y1: CGFloat(y1),
                           x2: CGFloat(x2), y2: CGFloat(y2), horizontal: o == "H")
            }
            DispatchQueue.main.async { self?.onSegments?(segs) }
        }.resume()
    }
}

// MARK: - Вид: рисует линии по нормализованным координатам

final class LinesView: NSView {
    private var segs: [Seg] = []

    override var isFlipped: Bool { true }   // (0,0) сверху-слева, как в кадре

    func update(_ s: [Seg]) {
        segs = s
        needsDisplay = true
    }

    override func draw(_ dirtyRect: NSRect) {
        // Стираем прошлый кадр до прозрачности: линии, пропавшие после нового
        // детекта, не должны оставаться призраками на экране.
        NSColor.clear.setFill()
        dirtyRect.fill(using: .copy)

        let w = bounds.width, h = bounds.height
        for s in segs {
            let path = NSBezierPath()
            path.lineWidth = 2
            path.lineCapStyle = .round
            path.move(to: CGPoint(x: s.x1 * w, y: s.y1 * h))
            path.line(to: CGPoint(x: s.x2 * w, y: s.y2 * h))
            if s.horizontal {
                NSColor(srgbRed: 0.15, green: 1.0, blue: 0.20, alpha: 0.95).setStroke()
            } else {
                NSColor(srgbRed: 0.75, green: 1.0, blue: 0.15, alpha: 0.95).setStroke()
            }
            path.stroke()
        }
    }
}

// MARK: - App delegate

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    let client = EdgesClient()
    var view: LinesView!

    func applicationDidFinishLaunching(_ notification: Notification) {
        let screen = NSScreen.main?.frame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)

        window = NSWindow(contentRect: screen, styleMask: [.borderless],
                          backing: .buffered, defer: false)
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = false
        window.level = .floating                 // поверх обычных окон, но безопасно
        window.ignoresMouseEvents = true         // click-through: не мешает работе
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]

        view = LinesView(frame: NSRect(origin: .zero, size: screen.size))
        // Layer-backing: без него прозрачное окно накапливает отрисовку и старые
        // линии не стираются. С ним AppKit чистит слой перед каждой перерисовкой.
        view.wantsLayer = true
        window.contentView = view

        client.onSegments = { [weak self] segs in self?.view.update(segs) }
        client.start()

        window.orderFrontRegardless()            // показать, но фокус не забирать
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

// MARK: - Bootstrap

let app = NSApplication.shared
app.setActivationPolicy(.accessory)   // без иконки в доке, не крадёт фокус у CLI
let delegate = AppDelegate()
app.delegate = delegate
app.run()
