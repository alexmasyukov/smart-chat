import AppKit
import QuartzCore
import AVFoundation

// =============================================================================
// fpstest — прозрачная форма (OVERLAY=1 — вверху поверх всех окон) с несколькими
// РАЗНЫМИ видами голос-реактивной анимации для сравнения, все на CADisplayLink
// (vsync-locked 120 Гц). Сверху — реальный счётчик fps/джиттера.
//   • слева   — размытый орб (scale + яркость)
//   • центр   — эквалайзер-бары (высота по голосу)
//   • справа  — пульсирующие кольца (разбегаются по голосу)
// Собрать: Fpstest.app (нужен микрофон). OVERLAY=1 — прозрачный оверлей.
// =============================================================================

// Микрофон: сглаженный уровень громкости 0..1 (резкий рост, быстрый спад).
final class MicLevel {
    private let engine = AVAudioEngine()
    var level: Float = 0
    func start() {
        let input = engine.inputNode
        let fmt = input.inputFormat(forBus: 0)
        guard fmt.channelCount > 0 else { return }
        // Меньший буфер — уровень обновляется чаще (~94/с вместо ~47/с).
        input.installTap(onBus: 0, bufferSize: 512, format: fmt) { [weak self] buf, _ in
            guard let self, let ch = buf.floatChannelData?[0] else { return }
            let n = Int(buf.frameLength); if n == 0 { return }
            var sum: Float = 0
            for i in 0..<n { let s = ch[i]; sum += s * s }
            let db = 20 * log10(max((sum / Float(n)).squareRoot(), 1e-7))
            var lvl = (db + 52) / 40; lvl = min(max(lvl, 0), 1)
            let cur = self.level
            // Резкий рост, быстрый спад — живая пульсация «говорения».
            self.level = lvl > cur ? cur + (lvl - cur) * 0.9 : cur + (lvl - cur) * 0.7
        }
        try? engine.start()
        FileHandle.standardError.write("[fpstest] микрофон запущен\n".data(using: .utf8)!)
    }
}

final class TestView: NSView {
    let mic = MicLevel()
    private var link: CADisplayLink?
    private let label = CATextLayer()
    private var t: CGFloat = 0
    private var dispLevel: Float = 0   // уровень, сглаженный НА КАЖДОМ кадре (120 Гц)

    // Палитра (2-й коммит): холодные тона + тёплый оранжевый акцент.
    private static let palette: [CGColor] = [
        NSColor(srgbRed: 0.20, green: 0.85, blue: 1.00, alpha: 1).cgColor,
        NSColor(srgbRed: 0.30, green: 0.45, blue: 1.00, alpha: 1).cgColor,
        NSColor(srgbRed: 0.65, green: 0.30, blue: 1.00, alpha: 1).cgColor,
        NSColor(srgbRed: 1.00, green: 0.30, blue: 0.70, alpha: 1).cgColor,
        NSColor(srgbRed: 1.00, green: 0.55, blue: 0.30, alpha: 1).cgColor,
        NSColor(srgbRed: 0.20, green: 0.85, blue: 1.00, alpha: 1).cgColor,
    ]

    // — эталон плавности: чёткий кружок, крутится с постоянной скоростью (не
    //   реагирует на голос). По нему видно на глаз, дёргается ли картинка. —
    private let ref = CAGradientLayer()

    // — вид 1: размытый орб (ЗАПЕЧЁННАЯ текстура, без маски в рантайме) —
    private let orb = CALayer()
    private let orbBaseR: CGFloat = 95
    private var orbCenter = CGPoint.zero
    private var orbAngle: CGFloat = 0

    // — вид 2: эквалайзер-бары —
    private var bars: [CALayer] = []
    private var barBaseY: CGFloat = 0
    private let barMaxH: CGFloat = 190

    // — вид 3: пульсирующие кольца (ЗАПЕЧЁННАЯ текстура кольца, без CAShapeLayer) —
    private var rings: [CALayer] = []
    private var ringCenter = CGPoint.zero

    // измерение fps
    private var lastTs: CFTimeInterval = 0
    private var acc: CFTimeInterval = 0
    private var cnt = 0
    private var minDt = 999.0
    private var maxDt = 0.0

    override init(frame: NSRect) {
        super.init(frame: frame)
        let host = CALayer()
        layer = host
        wantsLayer = true
        host.backgroundColor = NSColor.clear.cgColor

        buildRef(host: host, center: CGPoint(x: 90, y: frame.height - 150), r: 44)
        buildOrb(host: host, center: CGPoint(x: frame.width * 0.20, y: frame.midY))
        buildBars(host: host, center: CGPoint(x: frame.midX, y: frame.midY))
        buildRings(host: host, center: CGPoint(x: frame.width * 0.80, y: frame.midY))

        label.frame = CGRect(x: 20, y: frame.height - 70, width: frame.width - 40, height: 40)
        label.fontSize = 26
        label.foregroundColor = NSColor.white.cgColor
        label.string = "измерение…"
        label.contentsScale = 2
        host.addSublayer(label)
    }

    required init?(coder: NSCoder) { fatalError() }

    // MARK: эталон плавности — чёткий крутящийся кружок

    private func buildRef(host: CALayer, center: CGPoint, r: CGFloat) {
        ref.bounds = CGRect(x: 0, y: 0, width: r * 2, height: r * 2)
        ref.position = center
        ref.cornerRadius = r
        ref.masksToBounds = true
        ref.type = .conic
        ref.colors = (0..<24).map { i -> CGColor in
            let f = Double(i) / 23; let tri = f < 0.5 ? f * 2 : (1 - f) * 2
            return NSColor(hue: CGFloat((170 + tri * 160) / 360), saturation: 0.85,
                           brightness: 1, alpha: 1).cgColor
        }
        ref.startPoint = CGPoint(x: 0.5, y: 0.5)
        ref.endPoint = CGPoint(x: 0.5, y: 0)
        host.addSublayer(ref)
        spin(ref, duration: 6)
    }

    // MARK: вид 1 — размытый орб (радиальная маска = мягкие края)

    private func buildOrb(host: CALayer, center: CGPoint) {
        orbCenter = center
        let scale = window?.backingScaleFactor ?? 2
        orb.anchorPoint = CGPoint(x: 0.5, y: 0.5)
        orb.bounds = CGRect(x: 0, y: 0, width: orbBaseR * 2, height: orbBaseR * 2)
        orb.position = center
        // Радужное свечение с мягкими краями запекается ОДИН раз в текстуру. В
        // рантайме нет маски → нет offscreen-прохода, масштаб/поворот дёшевы.
        orb.contents = bakeOrb(diameter: orbBaseR * 2, scale: scale, colors: TestView.palette)
        orb.contentsScale = scale
        host.addSublayer(orb)
    }

    /// Запекает конический радужный градиент с радиальным затуханием альфы к краям
    /// в один CGImage. conic рисуется через render(in:), затем destinationIn с
    /// радиальным градиентом делает мягкие края.
    private func bakeOrb(diameter d: CGFloat, scale: CGFloat, colors: [CGColor]) -> CGImage? {
        let px = Int(d * scale)
        let cs = CGColorSpaceCreateDeviceRGB()
        let bi = CGImageAlphaInfo.premultipliedLast.rawValue
        // 1) конический градиент в bitmap.
        guard let c1 = CGContext(data: nil, width: px, height: px, bitsPerComponent: 8,
                                 bytesPerRow: 0, space: cs, bitmapInfo: bi) else { return nil }
        let conic = CAGradientLayer()
        conic.frame = CGRect(x: 0, y: 0, width: CGFloat(px), height: CGFloat(px))
        conic.type = .conic
        conic.colors = colors
        conic.startPoint = CGPoint(x: 0.5, y: 0.5)
        conic.endPoint = CGPoint(x: 0.5, y: 0.0)
        conic.render(in: c1)
        guard let conicImg = c1.makeImage() else { return nil }
        // 2) умножаем на радиальную альфу (мягкие края) через destinationIn.
        guard let c2 = CGContext(data: nil, width: px, height: px, bitsPerComponent: 8,
                                 bytesPerRow: 0, space: cs, bitmapInfo: bi) else { return nil }
        c2.draw(conicImg, in: CGRect(x: 0, y: 0, width: px, height: px))
        c2.setBlendMode(.destinationIn)
        let rad = CGGradient(colorsSpace: cs,
                             colors: [NSColor(white: 1, alpha: 1).cgColor,
                                      NSColor(white: 1, alpha: 1).cgColor,
                                      NSColor(white: 1, alpha: 0).cgColor] as CFArray,
                             locations: [0.0, 0.35, 1.0])!
        let mid = CGPoint(x: px / 2, y: px / 2)
        // .drawsAfterEndLocation — за радиусом продолжается clear (alpha 0), иначе
        // углы квадратной текстуры остаются непрозрачными (торчащие «огранки»).
        c2.drawRadialGradient(rad, startCenter: mid, startRadius: 0,
                              endCenter: mid, endRadius: CGFloat(px) / 2,
                              options: [.drawsAfterEndLocation])
        return c2.makeImage()
    }

    // MARK: вид 2 — эквалайзер: столбики, высота по голосу

    private func buildBars(host: CALayer, center: CGPoint) {
        let count = 7
        let w: CGFloat = 16, gap: CGFloat = 12
        let total = CGFloat(count) * w + CGFloat(count - 1) * gap
        barBaseY = center.y - barMaxH / 2
        var x = center.x - total / 2 + w / 2
        for i in 0..<count {
            let bar = CALayer()
            bar.anchorPoint = CGPoint(x: 0.5, y: 0)     // растёт вверх от базовой линии
            bar.bounds = CGRect(x: 0, y: 0, width: w, height: 20)
            bar.position = CGPoint(x: x, y: barBaseY)
            bar.cornerRadius = w / 2
            bar.backgroundColor = TestView.palette[i % TestView.palette.count]
            host.addSublayer(bar)
            bars.append(bar)
            x += w + gap
        }
    }

    // MARK: вид 3 — пульсирующие кольца, разбегаются по голосу

    private func buildRings(host: CALayer, center: CGPoint) {
        ringCenter = center
        let scale = window?.backingScaleFactor ?? 2
        // Кольцо-обводку запекаем ОДИН раз; 4 слоя разного размера дают концентрику,
        // а голос масштабирует готовую текстуру (без CAShapeLayer / offscreen).
        let tex = bakeRing(diameter: 200, scale: scale, colors: TestView.palette)
        for i in 0..<4 {
            let ring = CALayer()
            let d = 120 + CGFloat(i) * 44
            ring.anchorPoint = CGPoint(x: 0.5, y: 0.5)
            ring.bounds = CGRect(x: 0, y: 0, width: d, height: d)
            ring.position = center
            ring.contents = tex
            ring.contentsScale = scale
            ring.opacity = 0.2
            host.addSublayer(ring)
            rings.append(ring)
        }
    }

    /// Запекает радужное кольцо-обводку (conic + кольцевая альфа) в один CGImage.
    private func bakeRing(diameter d: CGFloat, scale: CGFloat, colors: [CGColor]) -> CGImage? {
        let px = Int(d * scale)
        let cs = CGColorSpaceCreateDeviceRGB()
        let bi = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let c1 = CGContext(data: nil, width: px, height: px, bitsPerComponent: 8,
                                 bytesPerRow: 0, space: cs, bitmapInfo: bi) else { return nil }
        let conic = CAGradientLayer()
        conic.frame = CGRect(x: 0, y: 0, width: CGFloat(px), height: CGFloat(px))
        conic.type = .conic
        conic.colors = colors
        conic.startPoint = CGPoint(x: 0.5, y: 0.5)
        conic.endPoint = CGPoint(x: 0.5, y: 0.0)
        conic.render(in: c1)
        guard let conicImg = c1.makeImage() else { return nil }
        guard let c2 = CGContext(data: nil, width: px, height: px, bitsPerComponent: 8,
                                 bytesPerRow: 0, space: cs, bitmapInfo: bi) else { return nil }
        c2.draw(conicImg, in: CGRect(x: 0, y: 0, width: px, height: px))
        c2.setBlendMode(.destinationIn)
        // Кольцевая альфа: прозрачно в центре и снаружи, непрозрачно у края (обводка).
        let rad = CGGradient(colorsSpace: cs,
                             colors: [NSColor(white: 1, alpha: 0).cgColor,
                                      NSColor(white: 1, alpha: 0).cgColor,
                                      NSColor(white: 1, alpha: 1).cgColor,
                                      NSColor(white: 1, alpha: 0).cgColor] as CFArray,
                             locations: [0.0, 0.68, 0.86, 1.0])!
        let mid = CGPoint(x: px / 2, y: px / 2)
        c2.drawRadialGradient(rad, startCenter: mid, startRadius: 0,
                              endCenter: mid, endRadius: CGFloat(px) / 2,
                              options: [.drawsAfterEndLocation])
        return c2.makeImage()
    }

    private func spin(_ l: CALayer, duration: CFTimeInterval) {
        let a = CABasicAnimation(keyPath: "transform.rotation.z")
        a.fromValue = 0; a.toValue = CGFloat.pi * 2
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
        t += CGFloat(link.duration > 0 ? link.duration : 1.0 / 120.0)

        // Уровень из микрофона обновляется ~94/с, а рисуем 120/с. Чтобы рост не был
        // ступенчатым, интерполируем отображаемый уровень КАЖДЫЙ кадр: быстрый рост
        // (эффект говорения слышен), плавный спад.
        let target = mic.level
        let k: Float = target > dispLevel ? 0.28 : 0.16
        dispLevel += (target - dispLevel) * k
        let g = powf(dispLevel, 1.1)

        CATransaction.begin()
        CATransaction.setDisableActions(true)

        // Вид 1: орб — вращаем и масштабируем ГОТОВУЮ текстуру (без маски).
        orbAngle += .pi * 2 * CGFloat(link.duration > 0 ? link.duration : 1.0 / 120.0) / 6
        let s = CGFloat(0.7 + CGFloat(g) * 1.3)
        var m = CATransform3DMakeScale(s, s, 1)
        m = CATransform3DRotate(m, orbAngle, 0, 0, 1)
        orb.transform = m
        orb.opacity = Float(0.55 + g * 0.45)

        // Вид 2: бары — высота по голосу, каждый со своей «дрожью» для живости.
        let n = bars.count
        for (i, bar) in bars.enumerated() {
            let bell = 0.5 + 0.5 * sin(.pi * CGFloat(i) / CGFloat(n - 1))   // выше в центре
            let wobble = 0.65 + 0.35 * abs(sin(t * (6 + CGFloat(i) * 0.7)))
            let h = 14 + CGFloat(g) * barMaxH * bell * wobble
            bar.bounds = CGRect(x: 0, y: 0, width: bar.bounds.width, height: h)
            bar.position = CGPoint(x: bar.position.x, y: barBaseY)
        }

        // Вид 3: кольца — разбегаются наружу и ярчают.
        for (i, ring) in rings.enumerated() {
            let gf = CGFloat(g)
            let sc = 1.0 + gf * (0.5 + CGFloat(i) * 0.6)
            ring.transform = CATransform3DMakeScale(sc, sc, 1)
            let op: CGFloat = 0.12 + gf * (0.9 - CGFloat(i) * 0.16)
            ring.opacity = Float(op)
        }
        CATransaction.commit()

        if cnt >= 20 {
            let fps = Double(cnt) / acc
            let maxHz = window?.screen?.maximumFramesPerSecond ?? 0
            label.string = String(format: "дисплей %d Гц   CADisplayLink %.0f fps   джиттер %.1f мс",
                                   maxHz, fps, (maxDt - minDt) * 1000)
            FileHandle.standardError.write(String(format: "[fpstest] fps=%.1f max=%.2fms\n",
                                                  fps, maxDt * 1000).data(using: .utf8)!)
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
        AVCaptureDevice.requestAccess(for: .audio) { [weak self] ok in
            DispatchQueue.main.async { if ok { self?.view.mic.start() } }
        }
    }
}
let d = D(); app.delegate = d; app.run()
