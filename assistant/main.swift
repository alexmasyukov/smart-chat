import AppKit
import AVFoundation
import QuartzCore
import CoreImage

// =============================================================================
// Assistant — нативный macOS-оверлей вокруг выреза под камеру (notch).
//
// Формы/окна нет. Прозрачный click-through оверлей рисует переливающееся размытое
// свечение по контуру выреза — как у Siri. Микрофон слушается постоянно: пока
// тихо — статичное мягкое свечение (НИКАКОЙ анимации, WindowServer не грузится),
// когда говоришь — свечение оживает, переливается и «прыгает» по уровню голоса.
//
// ВАЖНО про производительность (см. research): здесь НЕ используется Metal.
// CAMetalLayer — swapchain-поверхность, которую WindowServer перекомпоновывает
// каждый vsync (120 Гц) даже без презентов, из-за чего WindowServer жрёт CPU
// постоянно. Core Animation вместо этого: CABasicAnimation крутится внутри
// render-server (наш процесс спит), размытие запекается ОДИН раз в картинку,
// а в покое анимации нет вовсе → композитор кеширует статичный кадр.
// =============================================================================

// MARK: - Микрофон: мгновенный уровень громкости 0..1

/// Слушает вход по умолчанию через AVAudioEngine и держит сглаженный уровень
/// громкости в диапазоне 0..1 (быстрый attack, медленный release).
final class MicLevel {
    private let engine = AVAudioEngine()
    /// Читается из главного потока (аниматором). Пишется из аудио-потока.
    /// Гонка безвредна — это лишь визуальная величина.
    var level: Float = 0
    /// Вызывается (из аудио-потока), когда голос пересекает порог снизу вверх —
    /// сигнал «просыпайся» для спящего свечения. Внутри нужен переход на main.
    var onVoice: (() -> Void)?
    private var awake = false

    func start() {
        let input = engine.inputNode
        let fmt = input.inputFormat(forBus: 0)
        guard fmt.channelCount > 0 else {
            FileHandle.standardError.write("[assistant] нет входных каналов\n".data(using: .utf8)!)
            return
        }
        input.installTap(onBus: 0, bufferSize: 1024, format: fmt) { [weak self] buf, _ in
            guard let self, let ch = buf.floatChannelData?[0] else { return }
            let n = Int(buf.frameLength)
            if n == 0 { return }
            var sum: Float = 0
            for i in 0..<n { let s = ch[i]; sum += s * s }
            let rms = (sum / Float(n)).squareRoot()
            let db = 20 * log10(max(rms, 1e-7))
            // Порог тишины ~-52 dB, «в полный голос» ~-12 dB → нормируем в 0..1.
            var lvl = (db + 52) / 40
            lvl = min(max(lvl, 0), 1)
            // Сглаживание: резкий рост (свечение мгновенно «прыгает»), плавный спад.
            let cur = self.level
            self.level = lvl > cur ? cur + (lvl - cur) * 0.85
                                   : cur + (lvl - cur) * 0.16
            // Гистерезис на пробуждение: будим при 0.10, взводим снова ниже 0.04.
            if !self.awake, self.level > 0.10 { self.awake = true; self.onVoice?() }
            else if self.awake, self.level < 0.04 { self.awake = false }
        }
        do {
            try engine.start()
            FileHandle.standardError.write("[assistant] микрофон запущен (\(fmt.sampleRate)Гц, \(fmt.channelCount)ch)\n".data(using: .utf8)!)
        } catch {
            FileHandle.standardError.write("[assistant] ошибка запуска аудио: \(error)\n".data(using: .utf8)!)
        }
    }
}

// MARK: - Геометрия выреза

/// Прямоугольник выреза в глобальных координатах экрана (origin снизу-слева).
/// Если notch нет — фиктивный «вырез» по центру верхней грани.
func notchRect(on screen: NSScreen) -> CGRect {
    let f = screen.frame
    let top = screen.safeAreaInsets.top
    if top > 0, let l = screen.auxiliaryTopLeftArea, let r = screen.auxiliaryTopRightArea {
        return CGRect(x: l.maxX, y: f.maxY - top, width: r.minX - l.maxX, height: top)
    }
    let w: CGFloat = 190, h: CGFloat = 34
    return CGRect(x: f.midX - w / 2, y: f.maxY - h, width: w, height: h)
}

// MARK: - Вид: орб по центру выреза (Core Animation, без Metal)

// Радужный орб с мягкими краями ЗАПЕЧЁН один раз в текстуру (conic + радиальное
// затухание альфы). В рантайме только вращаем/масштабируем готовую текстуру —
// никакой маски и offscreen-прохода, поэтому 120 Гц плавно даже при увеличении.
// Голос раздувает орб; уровень сглаживается НА КАЖДОМ кадре (не ступеньками).
final class GlowView: NSView {
    let mic = MicLevel()
    private var notchLocal: CGRect
    private let orb = CALayer()
    private let orbR: CGFloat = 105
    private var link: CADisplayLink?
    private var animating = false
    private var lastVoice = CACurrentMediaTime()
    private let holdTime = 0.7
    private var orbAngle: CGFloat = 0
    private var dispLevel: Float = 0

    // Палитра (2-й коммит): холодные тона + тёплый оранжевый акцент.
    private static let palette: [CGColor] = [
        NSColor(srgbRed: 0.20, green: 0.85, blue: 1.00, alpha: 1).cgColor,
        NSColor(srgbRed: 0.30, green: 0.45, blue: 1.00, alpha: 1).cgColor,
        NSColor(srgbRed: 0.65, green: 0.30, blue: 1.00, alpha: 1).cgColor,
        NSColor(srgbRed: 1.00, green: 0.30, blue: 0.70, alpha: 1).cgColor,
        NSColor(srgbRed: 1.00, green: 0.55, blue: 0.30, alpha: 1).cgColor,
        NSColor(srgbRed: 0.20, green: 0.85, blue: 1.00, alpha: 1).cgColor,
    ]

    init(frame: NSRect, notchLocal: CGRect) {
        self.notchLocal = notchLocal
        super.init(frame: frame)
        let host = CALayer()
        layer = host
        wantsLayer = true
        host.backgroundColor = NSColor.clear.cgColor

        let scale = window?.backingScaleFactor ?? 2
        let center = CGPoint(x: notchLocal.midX, y: notchLocal.midY)   // центр выреза
        orb.anchorPoint = CGPoint(x: 0.5, y: 0.5)
        orb.bounds = CGRect(x: 0, y: 0, width: orbR * 2, height: orbR * 2)
        orb.position = center
        orb.contents = bakeOrb(diameter: orbR * 2, scale: scale, colors: GlowView.palette)
        orb.contentsScale = scale
        host.addSublayer(orb)
    }

    required init?(coder: NSCoder) { fatalError() }

    /// Запекает конический радужный градиент с радиальным затуханием альфы в CGImage.
    private func bakeOrb(diameter d: CGFloat, scale: CGFloat, colors: [CGColor]) -> CGImage? {
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
        let rad = CGGradient(colorsSpace: cs,
                             colors: [NSColor(white: 1, alpha: 1).cgColor,
                                      NSColor(white: 1, alpha: 1).cgColor,
                                      NSColor(white: 1, alpha: 0).cgColor] as CFArray,
                             locations: [0.0, 0.35, 1.0])!
        let mid = CGPoint(x: px / 2, y: px / 2)
        c2.drawRadialGradient(rad, startCenter: mid, startRadius: 0,
                              endCenter: mid, endRadius: CGFloat(px) / 2,
                              options: [.drawsAfterEndLocation])
        return c2.makeImage()
    }

    // MARK: сон/пробуждение

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        mic.onVoice = { [weak self] in
            DispatchQueue.main.async { self?.wake() }
        }
        applyIdle()   // стартуем со статичного покойного кадра
    }

    /// Разбудить: CADisplayLink 120 Гц (плавно), в тишине засыпает.
    func wake() {
        lastVoice = CACurrentMediaTime()
        if animating { return }
        animating = true
        let l = displayLink(target: self, selector: #selector(tick(_:)))
        l.preferredFrameRateRange = CAFrameRateRange(minimum: 80, maximum: 120, preferred: 120)
        l.add(to: .main, forMode: .common)
        link = l
    }

    @objc private func tick(_ link: CADisplayLink) {
        let now = CACurrentMediaTime()
        if mic.level > 0.06 { lastVoice = now }
        if now - lastVoice > holdTime { sleep(); return }

        // Сглаживаем уровень КАЖДЫЙ кадр (аудио обновляется реже рендера) — рост
        // без ступенек. Вращаем готовую текстуру.
        let target = mic.level
        let k: Float = target > dispLevel ? 0.28 : 0.16
        dispLevel += (target - dispLevel) * k
        let g = powf(dispLevel, 1.1)
        orbAngle += .pi * 2 * CGFloat(link.duration > 0 ? link.duration : 1.0 / 120.0) / 8

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        let s = CGFloat(0.55 + CGFloat(g) * 1.3)
        var m = CATransform3DMakeScale(s, s, 1)
        m = CATransform3DRotate(m, orbAngle, 0, 0, 1)
        orb.transform = m
        orb.opacity = Float(0.5 + g * 0.5)
        CATransaction.commit()
    }

    /// Уснуть: остановить display link, орб — маленький статичный.
    private func sleep() {
        animating = false
        link?.invalidate(); link = nil
        dispLevel = 0
        applyIdle()
    }

    private func applyIdle() {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        orb.transform = CATransform3DMakeScale(0.55, 0.55, 1)
        orb.opacity = 0.5
        CATransaction.commit()
    }
}

// MARK: - App delegate

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var view: GlowView!

    func applicationDidFinishLaunching(_ notification: Notification) {
        let screen = NSScreen.screens.first(where: { $0.safeAreaInsets.top > 0 })
            ?? NSScreen.main
            ?? NSScreen.screens[0]

        let n = notchRect(on: screen)
        // Окно туго вокруг выреза + запас под свечение (меньше площадь — дешевле
        // композитинг). Вверх некуда — вырез у самого края экрана.
        let mx: CGFloat = 170, my: CGFloat = 150
        let frame = CGRect(x: n.minX - mx, y: n.minY - my,
                           width: n.width + 2 * mx, height: n.height + my)

        window = NSWindow(contentRect: frame, styleMask: [.borderless],
                          backing: .buffered, defer: false)
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = false
        window.level = NSWindow.Level(rawValue: Int(CGShieldingWindowLevel()))
        window.ignoresMouseEvents = true                 // click-through
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]

        let notchLocal = CGRect(x: mx, y: my, width: n.width, height: n.height)
        view = GlowView(frame: NSRect(origin: .zero, size: frame.size), notchLocal: notchLocal)
        window.contentView = view
        window.orderFrontRegardless()

        FileHandle.standardError.write("[assistant] окно \(frame) вырез \(n)\n".data(using: .utf8)!)

        AVCaptureDevice.requestAccess(for: .audio) { [weak self] ok in
            DispatchQueue.main.async {
                if ok { self?.view.mic.start() }
                else { FileHandle.standardError.write("[assistant] нет доступа к микрофону\n".data(using: .utf8)!) }
            }
        }
    }
}

// MARK: - Bootstrap

let app = NSApplication.shared
app.setActivationPolicy(.accessory)   // без иконки в доке, не крадёт фокус
let delegate = AppDelegate()
app.delegate = delegate
app.run()
