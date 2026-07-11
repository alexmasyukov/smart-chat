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
            let ts = (obj["ts"] as? Double) ?? 0
            FileHandle.standardError.write(
                "[overlay] frame ts=\(ts) segs=\(segs.count)\n".data(using: .utf8)!)
            DispatchQueue.main.async { self?.onSegments?(segs) }
        }.resume()
    }
}

// MARK: - Вид: рисует линии через CAShapeLayer (полная замена содержимого)

// Каждый детект целиком заменяет `path` у слоёв — это атомарно перерисовывает
// всё с нуля, поэтому старые линии физически не могут остаться призраками
// (в отличие от drawRect на прозрачном окне, где буфер накапливался).
final class LinesView: NSView {
    private let hLayer = CAShapeLayer()   // горизонтали — ярко-зелёные
    private let vLayer = CAShapeLayer()   // вертикали — жёлто-зелёные

    override init(frame: NSRect) {
        super.init(frame: frame)
        let host = CALayer()
        host.isGeometryFlipped = true      // (0,0) сверху-слева, как в кадре детектора
        layer = host                       // layer-hosting: сперва layer, потом wantsLayer
        wantsLayer = true

        hLayer.strokeColor = NSColor(srgbRed: 0.15, green: 1.0, blue: 0.20, alpha: 0.95).cgColor
        vLayer.strokeColor = NSColor(srgbRed: 0.75, green: 1.0, blue: 0.15, alpha: 0.95).cgColor
        for lay in [hLayer, vLayer] {
            lay.fillColor = NSColor.clear.cgColor
            lay.lineWidth = 2
            lay.lineCap = .round
            lay.frame = bounds
            host.addSublayer(lay)
        }
    }

    required init?(coder: NSCoder) { fatalError() }

    func update(_ segs: [Seg]) {
        let w = bounds.width, h = bounds.height
        let hp = CGMutablePath()
        let vp = CGMutablePath()
        for s in segs {
            let p1 = CGPoint(x: s.x1 * w, y: s.y1 * h)
            let p2 = CGPoint(x: s.x2 * w, y: s.y2 * h)
            let path = s.horizontal ? hp : vp
            path.move(to: p1)
            path.addLine(to: p2)
        }
        CATransaction.begin()
        CATransaction.setDisableActions(true)   // без плавных переходов — мгновенно
        hLayer.frame = bounds
        vLayer.frame = bounds
        hLayer.path = hp                         // присваивание = полная замена кадра
        vLayer.path = vp
        CATransaction.commit()
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
        // КРИТИЧНО: прячем окно от screencapture. Иначе детектор снимает экран
        // вместе с нашими зелёными линиями, переобнаруживает их как яркие грани и
        // рисует снова — петля, из-за которой линии «не очищаются» никогда.
        window.sharingType = .none
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]

        view = LinesView(frame: NSRect(origin: .zero, size: screen.size))
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
