import AppKit
import AVFoundation
import QuartzCore

// =============================================================================
// form — прозрачная форма вверху экрана (поверх всех окон) с красивым голос-
// реактивным свечением: радужное пятно с мягко размытыми краями, переливается и
// «дышит», а на голос — разрастается и ярчает. Всё на Core Animation +
// CADisplayLink (vsync-locked 120 Гц, доказанно плавно на прозрачном окне).
//
// Размытые края даёт РАДИАЛЬНАЯ альфа-маска (центр непрозрачный → края тают), без
// CIGaussianBlur в рантайме — дёшево и плавно.
// =============================================================================

// MARK: - Микрофон: уровень громкости 0..1

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
            var lvl = (db + 52) / 40
            lvl = min(max(lvl, 0), 1)
            let cur = self.level
            self.level = lvl > cur ? cur + (lvl - cur) * 0.85 : cur + (lvl - cur) * 0.16
        }
        do { try engine.start()
            FileHandle.standardError.write("[form] микрофон запущен\n".data(using: .utf8)!)
        } catch {
            FileHandle.standardError.write("[form] аудио ошибка: \(error)\n".data(using: .utf8)!)
        }
    }
}

// MARK: - Свечение

/// Один слой: радужный конический градиент (переливается вращением) под мягкой
/// радиальной маской (размытые края). Наложение core+halo даёт объёмный орб.
private final class Orb {
    let holder = CALayer()            // масштаб/яркость по голосу
    let gradient = CAGradientLayer()  // радужный конический; вращается
    init() {}
}

final class GlowView: NSView {
    let mic = MicLevel()
    private var core = Orb()
    private var halo = Orb()
    private var link: CADisplayLink?
    private var angle: CGFloat = 0
    private var t: CGFloat = 0

    private static let palette: [CGColor] = {
        (0..<24).map { i -> CGColor in
            let f = Double(i) / 23; let tri = f < 0.5 ? f * 2 : (1 - f) * 2
            return NSColor(hue: CGFloat((170 + tri * 160) / 360), saturation: 0.82,
                           brightness: 1, alpha: 1).cgColor
        }
    }()

    override init(frame: NSRect) {
        super.init(frame: frame)
        let host = CALayer()
        layer = host
        wantsLayer = true
        host.backgroundColor = NSColor.clear.cgColor

        let c = CGPoint(x: frame.midX, y: frame.midY)
        setup(halo, radius: min(frame.width, frame.height) * 0.46, host: host, center: c)
        setup(core, radius: min(frame.width, frame.height) * 0.28, host: host, center: c)
    }

    required init?(coder: NSCoder) { fatalError() }

    private func setup(_ orb: Orb, radius r: CGFloat, host: CALayer, center c: CGPoint) {
        let box = CGRect(x: c.x - r, y: c.y - r, width: r * 2, height: r * 2)
        orb.holder.frame = box
        orb.holder.anchorPoint = CGPoint(x: 0.5, y: 0.5)
        orb.holder.position = c

        // Радужный конический градиент заполняет орб.
        orb.gradient.frame = orb.holder.bounds
        orb.gradient.type = .conic
        orb.gradient.colors = GlowView.palette
        orb.gradient.startPoint = CGPoint(x: 0.5, y: 0.5)
        orb.gradient.endPoint = CGPoint(x: 0.5, y: 0.0)
        orb.holder.addSublayer(orb.gradient)

        // Радиальная альфа-маска: непрозрачный центр → прозрачные края = мягкое
        // размытое пятно без CIFilter.
        let mask = CAGradientLayer()
        mask.frame = orb.holder.bounds
        mask.type = .radial
        mask.colors = [
            NSColor(white: 1, alpha: 1).cgColor,
            NSColor(white: 1, alpha: 1).cgColor,
            NSColor(white: 1, alpha: 0).cgColor,
        ]
        mask.locations = [0.0, 0.25, 1.0]   // плавный спад к краю = размытие
        mask.startPoint = CGPoint(x: 0.5, y: 0.5)
        mask.endPoint = CGPoint(x: 1.0, y: 1.0)
        orb.holder.mask = mask

        host.addSublayer(orb.holder)
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        guard window != nil, link == nil else { return }
        let l = displayLink(target: self, selector: #selector(tick(_:)))
        l.preferredFrameRateRange = CAFrameRateRange(minimum: 80, maximum: 120, preferred: 120)
        l.add(to: .main, forMode: .common)
        link = l
    }

    @objc private func tick(_ link: CADisplayLink) {
        let dt = CGFloat(link.duration > 0 ? link.duration : 1.0 / 120.0)
        t += dt
        angle += .pi * 2 * dt / 10          // перелив: оборот за 10с

        // «Дыхание» в покое + резкий буст по голосу.
        let breathe = (sin(t * 1.2) * 0.5 + 0.5) * 0.12 + 0.06   // 0.06..0.18
        let g = powf(max(mic.level, Float(breathe)), 0.6)

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        core.gradient.transform = CATransform3DMakeRotation(angle, 0, 0, 1)
        halo.gradient.transform = CATransform3DMakeRotation(-angle * 0.7, 0, 0, 1)
        let sc = CGFloat(0.85 + g * 0.9)
        let sh = CGFloat(0.9 + g * 1.3)
        core.holder.transform = CATransform3DMakeScale(sc, sc, 1)
        halo.holder.transform = CATransform3DMakeScale(sh, sh, 1)
        core.holder.opacity = Float(0.7 + g * 0.3)
        halo.holder.opacity = Float(0.25 + g * 0.55)
        CATransaction.commit()
    }
}

// MARK: - App

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var view: GlowView!

    func applicationDidFinishLaunching(_ notification: Notification) {
        let scr = (NSScreen.main ?? NSScreen.screens[0]).frame
        let w: CGFloat = min(1000, scr.width), h: CGFloat = 420
        let frame = CGRect(x: scr.midX - w / 2, y: scr.maxY - h, width: w, height: h)

        window = NSWindow(contentRect: frame, styleMask: [.borderless],
                          backing: .buffered, defer: false)
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = false
        window.level = NSWindow.Level(rawValue: Int(CGShieldingWindowLevel()))
        window.ignoresMouseEvents = true
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]

        view = GlowView(frame: NSRect(origin: .zero, size: frame.size))
        window.contentView = view
        window.orderFrontRegardless()

        AVCaptureDevice.requestAccess(for: .audio) { [weak self] ok in
            DispatchQueue.main.async { if ok { self?.view.mic.start() } }
        }
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
