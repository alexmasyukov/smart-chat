import AppKit
import Foundation

// Прозрачный click-through оверлей: рисует зелёные ТОЧКИ на границах блоков.
// Опрашивает detect.py (/points). Уроки прошлого проекта учтены:
//   • sharingType = .none  — окно невидимо для screencapture (иначе петля);
//   • без isGeometryFlipped — y инвертируем сами (иначе сдвиг сетки).

final class PointsClient {
    private let base: String
    private var lastV = 0
    var onData: (([[Double]], [String], [Int], [Double]) -> Void)?

    init() {
        base = ProcessInfo.processInfo.environment["POINTS_URL"]
            ?? "http://127.0.0.1:8132/points"
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
                let kinds = (obj["kinds"] as? [String]) ?? Array(repeating: "base", count: pts.count)
                let numbers = (obj["numbers"] as? [Int]) ?? Array(0..<pts.count)
                let lines = (obj["lines"] as? [Double]) ?? []
                FileHandle.standardError.write("[overlay] v=\(v) points=\(pts.count) lines=\(lines.count)\n"
                    .data(using: .utf8)!)
                DispatchQueue.main.async { self.onData?(pts, kinds, numbers, lines) }
            } else {
                delay = 0.5                            // сервер не поднят — притормозим
            }
            DispatchQueue.global().asyncAfter(deadline: .now() + delay) { self.fetch() }
        }.resume()
    }
}

final class PointsView: NSView {
    private let dots = CAShapeLayer()          // base-точки, зелёные
    private let probeDots = CAShapeLayer()     // probe-точки (несовпадение), оранжевые
    private let lines = CAShapeLayer()         // уточнённая граница блока, 1px
    private let labels = CALayer()          // хост для подписей номеров точек

    override init(frame: NSRect) {
        super.init(frame: frame)
        let host = CALayer()
        layer = host
        wantsLayer = true
        dots.fillColor = NSColor(srgbRed: 0.15, green: 1.0, blue: 0.20, alpha: 0.95).cgColor
        dots.strokeColor = NSColor.clear.cgColor
        probeDots.fillColor = NSColor(srgbRed: 1.0, green: 0.55, blue: 0.0, alpha: 0.95).cgColor
        probeDots.strokeColor = NSColor.clear.cgColor
        lines.fillColor = NSColor.clear.cgColor
        lines.strokeColor = NSColor(srgbRed: 0.15, green: 1.0, blue: 0.20, alpha: 0.95).cgColor
        lines.lineWidth = 1
        host.addSublayer(lines)
        host.addSublayer(dots)
        host.addSublayer(probeDots)
        host.addSublayer(labels)
    }

    required init?(coder: NSCoder) { fatalError() }

    func update(_ points: [[Double]], _ kinds: [String], _ numbers: [Int], _ lineXs: [Double]) {
        let w = bounds.width, h = bounds.height
        let r: CGFloat = 3
        let scale = self.window?.backingScaleFactor ?? 2.0
        let dotsPath = CGMutablePath()
        let probePath = CGMutablePath()
        labels.sublayers?.forEach { $0.removeFromSuperlayer() }
        for (i, p) in points.enumerated() where p.count >= 2 {
            let x = CGFloat(p[0]) * w
            let y = (1 - CGFloat(p[1])) * h          // y детектора сверху вниз -> слой снизу вверх
            let isProbe = i < kinds.count && kinds[i] == "probe"
            let path = isProbe ? probePath : dotsPath
            path.addEllipse(in: CGRect(x: x - r, y: y - r, width: 2 * r, height: 2 * r))
            let num = i < numbers.count ? numbers[i] : i

            // Номер точки — маленький белый текст высотой ~2 точки, НАД ней.
            let label = CATextLayer()
            label.string = "\(num)"
            label.fontSize = 4 * r                     // высота текста ~2 диаметра точки
            label.foregroundColor = NSColor.white.cgColor
            label.alignmentMode = .center
            label.contentsScale = scale
            let lw: CGFloat = 30
            label.frame = CGRect(x: x - lw / 2, y: y + r + 2, width: lw, height: 4 * r + 2)
            labels.addSublayer(label)
        }
        let linesPath = CGMutablePath()
        for nx in lineXs {
            let x = CGFloat(nx) * w
            linesPath.move(to: CGPoint(x: x, y: 0))
            linesPath.addLine(to: CGPoint(x: x, y: h))
        }
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        dots.frame = bounds
        dots.path = dotsPath
        probeDots.frame = bounds
        probeDots.path = probePath
        lines.frame = bounds
        lines.path = linesPath
        labels.frame = bounds
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
        client.onData = { [weak self] pts, kinds, nums, lns in self?.view.update(pts, kinds, nums, lns) }
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
