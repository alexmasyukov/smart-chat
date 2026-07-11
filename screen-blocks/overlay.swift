import AppKit
import Foundation

// Прозрачный click-through оверлей: рисует зелёные ТОЧКИ на границах блоков.
// Опрашивает detect.py (/points). Уроки прошлого проекта учтены:
//   • sharingType = .none  — окно невидимо для screencapture (иначе петля);
//   • без isGeometryFlipped — y инвертируем сами (иначе сдвиг сетки).

final class PointsClient {
    private let base: String
    private var lastV = 0
    var onPoints: (([[Double]]) -> Void)?

    init() {
        base = ProcessInfo.processInfo.environment["POINTS_URL"]
            ?? "http://127.0.0.1:8131/points"
    }

    func start() { fetch() }

    // Long-poll: висим на запросе до нового детекта, рисуем и сразу запрашиваем
    // снова. Перерисовка происходит ровно в момент готовности детекта.
    private func fetch() {
        guard let url = URL(string: "\(base)?since=\(lastV)") else { return }
        var req = URLRequest(url: url)
        req.timeoutInterval = 30                      // > серверного ожидания (25с)
        URLSession.shared.dataTask(with: req) { [weak self] data, _, err in
            guard let self else { return }
            var delay = 0.0
            if let data,
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let v = obj["v"] as? Int,
               let pts = obj["points"] as? [[Double]] {
                self.lastV = v
                FileHandle.standardError.write("[overlay] v=\(v) points=\(pts.count)\n"
                    .data(using: .utf8)!)
                DispatchQueue.main.async { self.onPoints?(pts) }
            } else {
                delay = 0.5                            // сервер не поднят — притормозим
            }
            DispatchQueue.global().asyncAfter(deadline: .now() + delay) { self.fetch() }
        }.resume()
    }
}

final class PointsView: NSView {
    private let dots = CAShapeLayer()

    override init(frame: NSRect) {
        super.init(frame: frame)
        let host = CALayer()
        layer = host
        wantsLayer = true
        dots.fillColor = NSColor(srgbRed: 0.15, green: 1.0, blue: 0.20, alpha: 0.95).cgColor
        dots.strokeColor = NSColor.clear.cgColor
        host.addSublayer(dots)
    }

    required init?(coder: NSCoder) { fatalError() }

    func update(_ points: [[Double]]) {
        let w = bounds.width, h = bounds.height
        let r: CGFloat = 3
        let path = CGMutablePath()
        for p in points where p.count >= 2 {
            let x = CGFloat(p[0]) * w
            let y = (1 - CGFloat(p[1])) * h          // y детектора сверху вниз -> слой снизу вверх
            path.addEllipse(in: CGRect(x: x - r, y: y - r, width: 2 * r, height: 2 * r))
        }
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        dots.frame = bounds
        dots.path = path
        CATransaction.commit()
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    let client = PointsClient()
    var view: PointsView!

    func applicationDidFinishLaunching(_ notification: Notification) {
        let screen = NSScreen.main?.frame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
        window = NSWindow(contentRect: screen, styleMask: [.borderless],
                          backing: .buffered, defer: false)
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = false
        window.level = .floating
        window.ignoresMouseEvents = true
        window.sharingType = .none                    // не попадать в screencapture
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        view = PointsView(frame: NSRect(origin: .zero, size: screen.size))
        window.contentView = view
        client.onPoints = { [weak self] pts in self?.view.update(pts) }
        client.start()
        window.orderFrontRegardless()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
