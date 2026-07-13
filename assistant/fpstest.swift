import AppKit
import QuartzCore
import AVFoundation

// Микрофон: сглаженный уровень громкости 0..1 (резкий рост, плавный спад).
final class MicLevel {
    private let engine = AVAudioEngine()
    var level: Float = 0
    func start() {
        let input = engine.inputNode
        let fmt = input.inputFormat(forBus: 0)
        guard fmt.channelCount > 0 else { return }
        input.installTap(onBus: 0, bufferSize: 1024, format: fmt) { [weak self] buf, _ in
            guard let self, let ch = buf.floatChannelData?[0] else { return }
            let n = Int(buf.frameLength); if n == 0 { return }
            var sum: Float = 0
            for i in 0..<n { let s = ch[i]; sum += s * s }
            let db = 20 * log10(max((sum / Float(n)).squareRoot(), 1e-7))
            var lvl = (db + 52) / 40; lvl = min(max(lvl, 0), 1)
            let cur = self.level
            // Резкий рост, ОЧЕНЬ быстрый спад — на паузах речи кружок мгновенно
            // опадает, создавая живую пульсацию «говорения».
            self.level = lvl > cur ? cur + (lvl - cur) * 0.9 : cur + (lvl - cur) * 0.7
        }
        try? engine.start()
        FileHandle.standardError.write("[fpstest] микрофон запущен\n".data(using: .utf8)!)
    }
}

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
    private let blurHolder = CALayer()          // размытый кружок рядом
    private let blurGrad = CAGradientLayer()
    private let blurMask = CAGradientLayer()     // радиальная маска (ссылка для ресайза)
    private let blurBaseR: CGFloat = 120         // базовый радиус кружка
    private var blurCenter = CGPoint.zero
    private let label = CATextLayer()
    let mic = MicLevel()
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
        // Прозрачный фон — чтобы в OVERLAY-режиме окно было по-настоящему прозрачным.
        host.backgroundColor = NSColor.clear.cgColor

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

        // Размытый кружок РЯДОМ (справа) — тот же конический градиент, но края
        // растворяются радиальной альфа-маской (мягкое размытие) вместо жёсткого
        // cornerRadius-обрезания у левого. Для сравнения края.
        blurCenter = CGPoint(x: frame.midX + 300, y: frame.midY)
        blurHolder.anchorPoint = CGPoint(x: 0.5, y: 0.5)
        blurHolder.bounds = CGRect(x: 0, y: 0, width: blurBaseR * 2, height: blurBaseR * 2)
        blurHolder.position = blurCenter
        blurGrad.frame = blurHolder.bounds
        blurGrad.type = .conic
        blurGrad.colors = glow.colors
        blurGrad.startPoint = CGPoint(x: 0.5, y: 0.5)
        blurGrad.endPoint = CGPoint(x: 0.5, y: 0)
        blurHolder.addSublayer(blurGrad)
        blurMask.frame = blurHolder.bounds
        blurMask.type = .radial
        blurMask.colors = [
            NSColor(white: 1, alpha: 1).cgColor,
            NSColor(white: 1, alpha: 1).cgColor,
            NSColor(white: 1, alpha: 0).cgColor,
        ]
        blurMask.locations = [0.0, 0.3, 1.0]
        blurMask.startPoint = CGPoint(x: 0.5, y: 0.5)
        blurMask.endPoint = CGPoint(x: 1.0, y: 1.0)
        blurHolder.mask = blurMask
        host.addSublayer(blurHolder)
        spin(blurGrad, duration: 6)

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
        // Правый (размытый) кружок реагирует на голос: раздувается и ярчает.
        // Лёгкая gamma>1 гасит тихий фон и подчёркивает пики → чёткая пульсация.
        let g = powf(mic.level, 1.1)
        let s = CGFloat(0.7 + g * 1.3)

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        redDot.position = CGPoint(x: x, y: redDot.position.y)
        // ТЕСТ: не transform, а РЕАЛЬНЫЙ ресайз слоя — CA перерастеризует конический
        // градиент и радиальную маску в новый размер каждый кадр (перерисовка).
        let r = blurBaseR * s
        let b = CGRect(x: 0, y: 0, width: r * 2, height: r * 2)
        blurHolder.bounds = b
        blurHolder.position = blurCenter
        blurGrad.frame = b
        blurMask.frame = b
        blurHolder.opacity = Float(0.55 + g * 0.45)
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
    var view: TestView!
    func applicationDidFinishLaunching(_ n: Notification) {
        let overlay = ProcessInfo.processInfo.environment["OVERLAY"] == "1"
        if overlay {
            // Прозрачный borderless оверлей вверху, поверх всех окон.
            let scr = NSScreen.main!.frame
            let f = CGRect(x: scr.midX - 450, y: scr.maxY - 520, width: 900, height: 520)
            view = TestView(frame: NSRect(origin: .zero, size: f.size))
            w = NSWindow(contentRect: f, styleMask: [.borderless], backing: .buffered, defer: false)
            w.isOpaque = false
            w.backgroundColor = .clear
            w.hasShadow = false
            w.level = NSWindow.Level(rawValue: Int(CGShieldingWindowLevel()))
            w.ignoresMouseEvents = true
            w.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
            w.contentView = view
            w.orderFrontRegardless()
        } else {
            let f = CGRect(x: 200, y: 200, width: 900, height: 520)
            view = TestView(frame: NSRect(origin: .zero, size: f.size))
            w = NSWindow(contentRect: f, styleMask: [.titled, .closable, .miniaturizable],
                         backing: .buffered, defer: false)
            w.title = "FPS test — обычное окно"
            w.contentView = view
            w.center()
            w.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
        }
        // Микрофон — правый (размытый) кружок реагирует на голос.
        AVCaptureDevice.requestAccess(for: .audio) { [weak self] ok in
            DispatchQueue.main.async { if ok { self?.view.mic.start() } }
        }
    }
}
let d = D(); app.delegate = d; app.run()
