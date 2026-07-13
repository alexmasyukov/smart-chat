import AppKit
import QuartzCore

// =============================================================================
// fpstest — диагностика реальной частоты кадров на ОБЫЧНОМ окне (с рамкой,
// непрозрачном). Показывает: максимум дисплея, измеренный CADisplayLink fps,
// и три способа анимации рядом для сравнения плавности:
//   • красный маркер — двигаю сам по CADisplayLink (ручной рендер-цикл)
//   • зелёный маркер — render-server CABasicAnimation (без участия процесса)
//   • конический градиент — наш «siri»-перелив (CABasicAnimation)
// Собрать/запустить:
//   swiftc -O fpstest.swift -o /tmp/fpstest -framework AppKit -framework QuartzCore && /tmp/fpstest
// =============================================================================

final class TestView: NSView {
    private let redDot = CALayer()      // двигается по CADisplayLink
    private let greenDot = CALayer()    // двигается по CABasicAnimation
    private let glow = CAGradientLayer()
    private let label = CATextLayer()
    private var link: CADisplayLink?

    // измерение fps
    private var lastTs: CFTimeInterval = 0
    private var acc: CFTimeInterval = 0
    private var cnt = 0
    private var fps = 0.0
    private var minDt = 999.0
    private var maxDt = 0.0

    override init(frame: NSRect) {
        super.init(frame: frame)
        let host = CALayer()
        layer = host
        wantsLayer = true
        host.backgroundColor = NSColor(white: 0.08, alpha: 1).cgColor

        // Конический радужный градиент в центре (наш эффект), вращается render-server.
        let side: CGFloat = 240
        glow.frame = CGRect(x: frame.midX - side/2, y: frame.midY - side/2, width: side, height: side)
        glow.type = .conic
        glow.cornerRadius = side/2
        glow.masksToBounds = true
        glow.colors = (0..<24).map { i -> CGColor in
            let f = Double(i)/23; let tri = f < 0.5 ? f*2 : (1-f)*2
            return NSColor(hue: CGFloat((170+tri*160)/360), saturation: 0.85, brightness: 1, alpha: 1).cgColor
        }
        glow.startPoint = CGPoint(x: 0.5, y: 0.5)
        glow.endPoint = CGPoint(x: 0.5, y: 0)
        host.addSublayer(glow)
        spin(glow, duration: 6)

        // Маркеры.
        redDot.frame = CGRect(x: 0, y: frame.height*0.72, width: 26, height: 26)
        redDot.cornerRadius = 13
        redDot.backgroundColor = NSColor.systemRed.cgColor
        host.addSublayer(redDot)

        greenDot.frame = CGRect(x: 0, y: frame.height*0.20, width: 26, height: 26)
        greenDot.cornerRadius = 13
        greenDot.backgroundColor = NSColor.systemGreen.cgColor
        host.addSublayer(greenDot)
        // Зелёный — чистая render-server анимация (autoreverse слева-направо).
        let a = CABasicAnimation(keyPath: "position.x")
        a.fromValue = 13
        a.toValue = frame.width - 13
        a.duration = 1.4
        a.autoreverses = true
        a.repeatCount = .infinity
        a.timingFunction = CAMediaTimingFunction(name: .linear)
        greenDot.add(a, forKey: "move")

        // Текст fps.
        label.frame = CGRect(x: 20, y: frame.height - 90, width: frame.width - 40, height: 60)
        label.fontSize = 34
        label.foregroundColor = NSColor.white.cgColor
        label.string = "измерение…"
        label.contentsScale = 2
        host.addSublayer(label)
    }

    required init?(coder: NSCoder) { fatalError() }

    private func spin(_ l: CALayer, duration: CFTimeInterval) {
        let a = CABasicAnimation(keyPath: "transform.rotation.z")
        a.fromValue = 0; a.toValue = CGFloat.pi*2
        a.duration = duration; a.repeatCount = .infinity; a.isRemovedOnCompletion = false
        l.add(a, forKey: "spin")
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        guard window != nil, link == nil else { return }
        let l = displayLink(target: self, selector: #selector(tick(_:)))
        l.preferredFrameRateRange = CAFrameRateRange(minimum: 100, maximum: 120, preferred: 120)
        l.add(to: .main, forMode: .common)
        link = l
        let maxHz = window?.screen?.maximumFramesPerSecond ?? 0
        FileHandle.standardError.write("[fpstest] дисплей max=\(maxHz)Гц\n".data(using: .utf8)!)
    }

    @objc private func tick(_ link: CADisplayLink) {
        let ts = link.timestamp
        if lastTs > 0 {
            let dt = ts - lastTs
            acc += dt; cnt += 1
            minDt = min(minDt, dt); maxDt = max(maxDt, dt)
        }
        lastTs = ts

        // Двигаю красный маркер сам, по реальному времени кадра (ручной рендер).
        let period = 1.4 * 2
        let phase = ts.truncatingRemainder(dividingBy: period) / period   // 0..1
        let tri = phase < 0.5 ? phase*2 : (1-phase)*2                       // 0..1..0
        let x = 13 + CGFloat(tri) * (bounds.width - 26)
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        redDot.position = CGPoint(x: x, y: redDot.position.y)
        CATransaction.commit()

        if cnt >= 20 {
            fps = Double(cnt) / acc
            let maxHz = window?.screen?.maximumFramesPerSecond ?? 0
            let jitter = (maxDt - minDt) * 1000
            label.string = String(format: "дисплей %d Гц   CADisplayLink %.0f fps   джиттер %.1f мс",
                                   maxHz, fps, jitter)
            FileHandle.standardError.write(String(format: "[fpstest] fps=%.1f min=%.2fms max=%.2fms\n",
                                                  fps, minDt*1000, maxDt*1000).data(using: .utf8)!)
            acc = 0; cnt = 0; minDt = 999; maxDt = 0
        }
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
final class D: NSObject, NSApplicationDelegate {
    var w: NSWindow!
    func applicationDidFinishLaunching(_ n: Notification) {
        let f = CGRect(x: 200, y: 200, width: 900, height: 520)
        w = NSWindow(contentRect: f, styleMask: [.titled, .closable, .miniaturizable],
                     backing: .buffered, defer: false)
        w.title = "FPS test — обычное окно"
        w.contentView = TestView(frame: NSRect(origin: .zero, size: f.size))
        w.center()
        w.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}
let d = D(); app.delegate = d; app.run()
