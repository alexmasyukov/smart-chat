import AppKit
import AVFoundation
import QuartzCore

// =============================================================================
// tuner — форма с ползунками для ЖИВОЙ настройки обёртки выреза (того же эффекта,
// что в main.swift). Слева-снизу панель со всеми параметрами (у каждого ползунка
// видно точное значение), справа-вверху — сам эффект вокруг выреза, меняется
// сразу. Внизу панели — копируемый дамп всех значений: настрой как нравится,
// выдели, скопируй и передай.
//
// Собрать: Tuner.app (нужен микрофон). Основной эффект (main.swift) НЕ трогает.
// =============================================================================

// MARK: - Параметры (дефолты = текущий вид main.swift)

struct Params {
    var scaleX: CGFloat = 1.0        // ширина обёртки
    var scaleY: CGFloat = 1.0        // высота обёртки
    var outlineWidth: CGFloat = 64   // толщина ореола (макс ширина стека обводок)
    var blur: CGFloat = 5            // размытие маски
    var corner: CGFloat = 0.7        // скругление углов выреза (доля)
    var falloff: CGFloat = 1.5       // спад яркости к внешнему краю
    var saturation: CGFloat = 1.4    // насыщенность цветов
    var brightness: CGFloat = 1.0    // яркость цветов
    var rotSec: CGFloat = 9          // период вращения перелива, сек
    var voiceScale: CGFloat = 0.7    // размах на голос
    var idleOpacity: Float = 0.85    // прозрачность в покое
    var voiceOpacity: Float = 0.15   // прирост прозрачности на голос
    var attack: Float = 0.28         // сглаживание роста (per-frame)
    var release: Float = 0.16        // сглаживание спада (per-frame)
    var gamma: Float = 1.1           // контраст реакции на голос
    var sensitivity: Float = 1.0     // чувствительность микрофона

    func dump() -> String {
        String(format:
        "scaleX=%.2f scaleY=%.2f outline=%.0f blur=%.0f corner=%.2f falloff=%.2f " +
        "sat=%.2f bright=%.2f rotSec=%.1f voiceScale=%.2f idleOp=%.2f voiceOp=%.2f " +
        "attack=%.2f release=%.2f gamma=%.2f sens=%.2f",
        scaleX, scaleY, outlineWidth, blur, corner, falloff, saturation, brightness,
        rotSec, voiceScale, idleOpacity, voiceOpacity, attack, release, gamma, sensitivity)
    }
}

// MARK: - Микрофон

final class MicLevel {
    private let engine = AVAudioEngine()
    var level: Float = 0
    var attack: Float = 0.9
    var release: Float = 0.7
    var sensitivity: Float = 1.0
    func start() {
        let input = engine.inputNode
        let fmt = input.inputFormat(forBus: 0)
        guard fmt.channelCount > 0 else { return }
        input.installTap(onBus: 0, bufferSize: 512, format: fmt) { [weak self] buf, _ in
            guard let self, let ch = buf.floatChannelData?[0] else { return }
            let n = Int(buf.frameLength); if n == 0 { return }
            var sum: Float = 0
            for i in 0..<n { let s = ch[i]; sum += s * s }
            let db = 20 * log10(max((sum / Float(n)).squareRoot(), 1e-7))
            var lvl = (db + 52) / 40 * self.sensitivity
            lvl = min(max(lvl, 0), 1)
            let cur = self.level
            self.level = lvl > cur ? cur + (lvl - cur) * self.attack
                                   : cur + (lvl - cur) * self.release
        }
        try? engine.start()
    }
}

// MARK: - Оверлей эффекта (обёртка выреза), управляется Params

final class GlowOverlay: NSView {
    let mic = MicLevel()
    var p = Params()
    private var notchLocal: CGRect
    private let orb = CALayer()
    private let conic = CAGradientLayer()
    private var maskLayer = CALayer()
    private var center = CGPoint.zero
    private var link: CADisplayLink?
    private var angle: CGFloat = 0
    private var dispLevel: Float = 0

    init(frame: NSRect, notchLocal: CGRect) {
        self.notchLocal = notchLocal
        super.init(frame: frame)
        let host = CALayer()
        layer = host; wantsLayer = true
        host.backgroundColor = NSColor.clear.cgColor
        center = CGPoint(x: notchLocal.midX, y: notchLocal.midY)

        let side = max(bounds.width, bounds.height) * 1.8
        conic.frame = CGRect(x: center.x - side/2, y: center.y - side/2, width: side, height: side)
        conic.type = .conic
        conic.startPoint = CGPoint(x: 0.5, y: 0.5)
        conic.endPoint = CGPoint(x: 0.5, y: 0.0)

        orb.bounds = CGRect(origin: .zero, size: bounds.size)
        orb.anchorPoint = CGPoint(x: center.x / bounds.width, y: center.y / bounds.height)
        orb.position = center
        orb.addSublayer(conic)
        maskLayer.frame = bounds
        orb.mask = maskLayer
        host.addSublayer(orb)

        rebake()
    }
    required init?(coder: NSCoder) { fatalError() }

    /// Пересобрать текстуры, зависящие от структурных параметров.
    func rebake() {
        let scale = window?.backingScaleFactor ?? 2
        conic.colors = palette()
        let outline = notchOutline(notchLocal, radius: min(notchLocal.height, notchLocal.width/2) * p.corner)
        maskLayer.contentsScale = scale
        maskLayer.contents = bakeMask(size: bounds.size, scale: scale, outline: outline)
    }

    private func palette() -> [CGColor] {
        let base: [(CGFloat, CGFloat, CGFloat)] = [
            (0.20,0.85,1.00),(0.30,0.45,1.00),(0.65,0.30,1.00),
            (1.00,0.30,0.70),(1.00,0.55,0.30),(0.20,0.85,1.00)]
        return base.map { rgb in
            let c = NSColor(srgbRed: rgb.0, green: rgb.1, blue: rgb.2, alpha: 1).usingColorSpace(.deviceRGB)!
            var h: CGFloat=0, s: CGFloat=0, b: CGFloat=0, a: CGFloat=0
            c.getHue(&h, saturation:&s, brightness:&b, alpha:&a)
            return NSColor(hue: h, saturation: min(s * p.saturation, 1),
                           brightness: min(b * p.brightness, 1), alpha: 1).cgColor
        }
    }

    private func notchOutline(_ r: CGRect, radius rad: CGFloat) -> CGPath {
        let p = CGMutablePath()
        let x0 = r.minX, x1 = r.maxX, yTop = r.maxY, yBot = r.minY
        p.move(to: CGPoint(x: x0, y: yTop))
        p.addArc(tangent1End: CGPoint(x: x0, y: yBot), tangent2End: CGPoint(x: x0+rad, y: yBot), radius: rad)
        p.addArc(tangent1End: CGPoint(x: x1, y: yBot), tangent2End: CGPoint(x: x1, y: yTop), radius: rad)
        p.addLine(to: CGPoint(x: x1, y: yTop))
        return p
    }

    private func bakeMask(size: CGSize, scale: CGFloat, outline: CGPath) -> CGImage? {
        let W = Int(size.width*scale), H = Int(size.height*scale)
        guard W > 0, H > 0 else { return nil }
        let cs = CGColorSpaceCreateDeviceRGB(); let bi = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let cm = CGContext(data: nil, width: W, height: H, bitsPerComponent: 8,
                                 bytesPerRow: 0, space: cs, bitmapInfo: bi) else { return nil }
        cm.scaleBy(x: scale, y: scale)
        cm.setLineCap(.round); cm.setLineJoin(.round)
        // Стек обводок от широкой тусклой к узкой яркой (пик у кромки, спад наружу).
        let steps = 6
        for i in 0..<steps {
            let t = CGFloat(i) / CGFloat(steps - 1)             // 0..1 (0 = широкая)
            let w = p.outlineWidth * (1 - t) + 3 * t
            // Как в main: широкая обводка не гаснет в 0 (нижний порог 0.1) —
            // сохраняется мягкий внешний ореол, яркость растёт к кромке.
            let a = 0.1 + 0.9 * pow(t, p.falloff)
            cm.setStrokeColor(CGColor(gray: 1, alpha: a))
            cm.setLineWidth(max(w, 1))
            cm.addPath(outline); cm.strokePath()
        }
        guard let img = cm.makeImage() else { return nil }
        let ci = CIImage(cgImage: img)
        let blurred = ci.clampedToExtent().applyingGaussianBlur(sigma: Double(p.blur*scale)).cropped(to: ci.extent)
        return CIContext(options: [.useSoftwareRenderer: false]).createCGImage(blurred, from: ci.extent)
    }

    func startLink() {
        guard link == nil else { return }
        let l = displayLink(target: self, selector: #selector(tick(_:)))
        l.preferredFrameRateRange = CAFrameRateRange(minimum: 80, maximum: 120, preferred: 120)
        l.add(to: .main, forMode: .common)
        link = l
    }

    @objc private func tick(_ link: CADisplayLink) {
        mic.attack = 0.85; mic.release = 0.16; mic.sensitivity = p.sensitivity   // как в main
        let target = mic.level
        let k: Float = target > dispLevel ? p.attack : p.release
        dispLevel += (target - dispLevel) * k
        let g = powf(dispLevel, p.gamma)
        angle += .pi * 2 * CGFloat(link.duration > 0 ? link.duration : 1.0/120) / max(p.rotSec, 0.5)

        CATransaction.begin(); CATransaction.setDisableActions(true)
        conic.transform = CATransform3DMakeRotation(angle, 0, 0, 1)
        let sx = p.scaleX * CGFloat(1 + CGFloat(g) * p.voiceScale)
        let sy = p.scaleY * CGFloat(1 + CGFloat(g) * p.voiceScale)
        orb.transform = CATransform3DMakeScale(sx, sy, 1)
        orb.opacity = p.idleOpacity + Float(g) * p.voiceOpacity
        CATransaction.commit()
    }
}

// MARK: - Панель с ползунками

final class Slider: NSView {
    let onChange: (CGFloat) -> Void
    private let valueLabel = NSTextField(labelWithString: "")
    private let slider = NSSlider()
    private let fmt: String
    init(title: String, min: Double, max: Double, def: Double, format: String,
         onChange: @escaping (CGFloat) -> Void) {
        self.onChange = onChange; self.fmt = format
        super.init(frame: .zero)
        let name = NSTextField(labelWithString: title)
        name.font = .systemFont(ofSize: 11); name.frame = CGRect(x: 0, y: 2, width: 150, height: 16)
        slider.minValue = min; slider.maxValue = max; slider.doubleValue = def
        slider.frame = CGRect(x: 155, y: 0, width: 190, height: 20)
        slider.target = self; slider.action = #selector(changed)
        valueLabel.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        valueLabel.frame = CGRect(x: 350, y: 2, width: 60, height: 16)
        valueLabel.stringValue = String(format: format, def)
        addSubview(name); addSubview(slider); addSubview(valueLabel)
        frame = CGRect(x: 0, y: 0, width: 415, height: 22)
    }
    required init?(coder: NSCoder) { fatalError() }
    @objc private func changed() {
        valueLabel.stringValue = String(format: fmt, slider.doubleValue)
        onChange(CGFloat(slider.doubleValue))
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    var panel: NSWindow!
    var overlayWin: NSWindow!
    var overlay: GlowOverlay!
    let dump = NSTextField(wrappingLabelWithString: "")

    func notchRect(on s: NSScreen) -> CGRect {
        let f = s.frame; let top = s.safeAreaInsets.top
        if top > 0, let l = s.auxiliaryTopLeftArea, let r = s.auxiliaryTopRightArea {
            return CGRect(x: l.maxX, y: f.maxY - top, width: r.minX - l.maxX, height: top)
        }
        return CGRect(x: f.midX - 95, y: f.maxY - 34, width: 190, height: 34)
    }

    func applicationDidFinishLaunching(_ n: Notification) {
        let screen = NSScreen.screens.first(where: { $0.safeAreaInsets.top > 0 }) ?? NSScreen.main!
        let nrect = notchRect(on: screen)

        // Оверлей эффекта — большое окно с запасом вокруг выреза (фиксированное).
        let ow: CGFloat = 640, oh: CGFloat = 380
        let of = CGRect(x: nrect.midX - ow/2, y: nrect.maxY - oh, width: ow, height: oh)
        let notchLocal = CGRect(x: nrect.minX - of.minX, y: nrect.minY - of.minY,
                                width: nrect.width, height: nrect.height)
        overlayWin = NSWindow(contentRect: of, styleMask: [.borderless], backing: .buffered, defer: false)
        overlayWin.isOpaque = false; overlayWin.backgroundColor = .clear; overlayWin.hasShadow = false
        overlayWin.level = NSWindow.Level(rawValue: Int(CGShieldingWindowLevel()))
        overlayWin.ignoresMouseEvents = true
        overlayWin.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        overlay = GlowOverlay(frame: NSRect(origin: .zero, size: of.size), notchLocal: notchLocal)
        overlayWin.contentView = overlay
        overlayWin.orderFrontRegardless()
        overlay.startLink()

        buildPanel()

        AVCaptureDevice.requestAccess(for: .audio) { [weak self] ok in
            DispatchQueue.main.async { if ok { self?.overlay.mic.start() } }
        }
    }

    private func buildPanel() {
        let rows: [(String, Double, Double, Double, String, (CGFloat) -> Void)] = [
            ("Ширина обёртки", 0.5, 2.0, 1.0, "%.2f",   { self.overlay.p.scaleX = $0; self.live() }),
            ("Высота обёртки", 0.5, 2.0, 1.0, "%.2f",   { self.overlay.p.scaleY = $0; self.live() }),
            ("Толщина ореола", 20, 140, 64, "%.0f",     { self.overlay.p.outlineWidth = $0; self.bake() }),
            ("Размытие", 0, 30, 5, "%.0f",              { self.overlay.p.blur = $0; self.bake() }),
            ("Скругление углов", 0, 1, 0.7, "%.2f",     { self.overlay.p.corner = $0; self.bake() }),
            ("Спад к краю", 0.5, 3, 1.5, "%.2f",        { self.overlay.p.falloff = $0; self.bake() }),
            ("Насыщенность", 0.5, 2.0, 1.4, "%.2f",     { self.overlay.p.saturation = $0; self.bake() }),
            ("Яркость цвета", 0.4, 1.0, 1.0, "%.2f",    { self.overlay.p.brightness = $0; self.bake() }),
            ("Период вращения, с", 2, 30, 9, "%.1f",    { self.overlay.p.rotSec = $0; self.live() }),
            ("Размах на голос", 0, 2.5, 0.7, "%.2f",    { self.overlay.p.voiceScale = $0; self.live() }),
            ("Прозрачность покоя", 0, 1, 0.85, "%.2f",  { self.overlay.p.idleOpacity = Float($0); self.live() }),
            ("Прирост прозрачности", 0, 1, 0.15, "%.2f",{ self.overlay.p.voiceOpacity = Float($0); self.live() }),
            ("Attack (рост)", 0.05, 1, 0.28, "%.2f",    { self.overlay.p.attack = Float($0); self.live() }),
            ("Release (спад)", 0.05, 1, 0.16, "%.2f",   { self.overlay.p.release = Float($0); self.live() }),
            ("Gamma (контраст)", 0.4, 2.5, 1.1, "%.2f", { self.overlay.p.gamma = Float($0); self.live() }),
            ("Чувствительность", 0.5, 2.0, 1.0, "%.2f", { self.overlay.p.sensitivity = Float($0); self.live() }),
        ]
        let top: CGFloat = 30
        let panelH = top + CGFloat(rows.count) * 26 + 120
        let container = NSView(frame: CGRect(x: 0, y: 0, width: 440, height: panelH))
        var y = panelH - top
        for r in rows {
            y -= 26
            let s = Slider(title: r.0, min: r.1, max: r.2, def: r.3, format: r.4, onChange: r.5)
            s.frame.origin = CGPoint(x: 15, y: y)
            container.addSubview(s)
        }
        // Дамп внизу — копируемый.
        dump.frame = CGRect(x: 15, y: 10, width: 410, height: 96)
        dump.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        dump.isSelectable = true; dump.isBezeled = true; dump.isEditable = false
        dump.backgroundColor = NSColor.textBackgroundColor
        dump.stringValue = overlay.p.dump()
        container.addSubview(dump)

        panel = NSWindow(contentRect: container.frame, styleMask: [.titled, .closable],
                         backing: .buffered, defer: false)
        panel.title = "Tuner — обёртка выреза"
        panel.contentView = container
        panel.setFrameOrigin(CGPoint(x: 60, y: 120))
        panel.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func live() { dump.stringValue = overlay.p.dump() }         // без пересборки
    private func bake() { overlay.rebake(); dump.stringValue = overlay.p.dump() }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
