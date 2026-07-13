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
    private let orb = CALayer()               // контейнер: маска-форма + масштаб
    private let conic = CAGradientLayer()     // радужный градиент; вращается (перелив)
    private var link: CADisplayLink?
    private var animating = false
    private var lastVoice = CACurrentMediaTime()
    private let holdTime = 0.7
    private var orbAngle: CGFloat = 0
    private var dispLevel: Float = 0

    // Прежние цвета (2-й коммит), но с ПОДНЯТОЙ насыщенностью — оттенки те же.
    private static let palette: [CGColor] = {
        let base: [(CGFloat, CGFloat, CGFloat)] = [
            (0.20, 0.85, 1.00),  // cyan
            (0.30, 0.45, 1.00),  // blue
            (0.65, 0.30, 1.00),  // purple
            (1.00, 0.30, 0.70),  // pink
            (1.00, 0.55, 0.30),  // orange
            (0.20, 0.85, 1.00),  // cyan (замыкание)
        ]
        return base.map { rgb -> CGColor in
            let c = NSColor(srgbRed: rgb.0, green: rgb.1, blue: rgb.2, alpha: 1)
                .usingColorSpace(.deviceRGB)!
            var h: CGFloat = 0, s: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
            c.getHue(&h, saturation: &s, brightness: &b, alpha: &a)
            return NSColor(hue: h, saturation: min(s * 1.4, 1), brightness: b, alpha: 1).cgColor
        }
    }()

    init(frame: NSRect, notchLocal: CGRect) {
        self.notchLocal = notchLocal
        super.init(frame: frame)
        let host = CALayer()
        layer = host
        wantsLayer = true
        host.backgroundColor = NSColor.clear.cgColor

        let scale = window?.backingScaleFactor ?? 2
        let center = CGPoint(x: notchLocal.midX, y: notchLocal.midY)   // центр выреза

        // Радужный конический слой — большой квадрат с центром в вырезе; ВРАЩАЕТСЯ
        // (перелив цветов). Насыщенная палитра.
        let side = max(bounds.width, bounds.height) * 1.8
        conic.frame = CGRect(x: center.x - side / 2, y: center.y - side / 2,
                             width: side, height: side)
        conic.type = .conic
        conic.colors = GlowView.palette
        conic.startPoint = CGPoint(x: 0.5, y: 0.5)
        conic.endPoint = CGPoint(x: 0.5, y: 0.0)

        // Контейнер размером с окно, якорь в центре выреза (масштаб раздувает
        // свечение наружу). Маска — ЗАПЕЧЁННАЯ размытая обводка выреза: форма стоит
        // на месте, а под ней крутится радуга → цвета переливаются, вырез обёрнут.
        orb.bounds = CGRect(origin: .zero, size: bounds.size)
        orb.anchorPoint = CGPoint(x: center.x / bounds.width, y: center.y / bounds.height)
        orb.position = center
        orb.addSublayer(conic)
        let outline = notchOutline(notchLocal,
                                   radius: min(notchLocal.height, notchLocal.width / 2) * 0.7)
        if let maskImg = bakeMask(size: bounds.size, scale: scale, outline: outline,
                                  softBlur: 5) {
            let m = CALayer()
            m.frame = bounds
            m.contentsScale = scale
            m.contents = maskImg
            orb.mask = m
        }
        host.addSublayer(orb)
    }

    required init?(coder: NSCoder) { fatalError() }

    /// Контур выреза: открытая «U» — по левой стороне вниз, скруглённый низ,
    /// по правой стороне вверх. Верхней грани нет (она у самого края экрана).
    private func notchOutline(_ r: CGRect, radius rad: CGFloat) -> CGPath {
        let p = CGMutablePath()
        let x0 = r.minX, x1 = r.maxX, yTop = r.maxY, yBot = r.minY
        p.move(to: CGPoint(x: x0, y: yTop))
        p.addArc(tangent1End: CGPoint(x: x0, y: yBot),
                 tangent2End: CGPoint(x: x0 + rad, y: yBot), radius: rad)
        p.addArc(tangent1End: CGPoint(x: x1, y: yBot),
                 tangent2End: CGPoint(x: x1, y: yTop), radius: rad)
        p.addLine(to: CGPoint(x: x1, y: yTop))
        return p
    }

    /// Запекает маску-обводку выреза ОДИН раз. Стек обводок от широкой тусклой к
    /// узкой яркой → альфа максимальна ПРЯМО У КРОМКИ выреза и спадает наружу
    /// (а не однообразный ровный ободок). Лёгкий блюр сглаживает ступени.
    private func bakeMask(size: CGSize, scale: CGFloat, outline: CGPath,
                          softBlur: CGFloat) -> CGImage? {
        let W = Int(size.width * scale), H = Int(size.height * scale)
        guard W > 0, H > 0 else { return nil }
        let cs = CGColorSpaceCreateDeviceRGB()
        let bi = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let cm = CGContext(data: nil, width: W, height: H, bitsPerComponent: 8,
                                 bytesPerRow: 0, space: cs, bitmapInfo: bi) else { return nil }
        cm.scaleBy(x: scale, y: scale)
        cm.setLineCap(.round); cm.setLineJoin(.round)
        // ширина ↓, альфа ↑ — узкие яркие обводки поверх дают пик у самой кромки.
        let stack: [(w: CGFloat, a: CGFloat)] = [
            (64, 0.10), (44, 0.18), (28, 0.32), (16, 0.55), (8, 0.85), (3, 1.0),
        ]
        for s in stack {
            cm.setStrokeColor(CGColor(gray: 1, alpha: s.a))
            cm.setLineWidth(s.w)
            cm.addPath(outline); cm.strokePath()
        }
        guard let strokeImg = cm.makeImage() else { return nil }
        let ci = CIImage(cgImage: strokeImg)
        let blurred = ci.clampedToExtent()
            .applyingGaussianBlur(sigma: Double(softBlur * scale)).cropped(to: ci.extent)
        return CIContext(options: [.useSoftwareRenderer: false])
            .createCGImage(blurred, from: ci.extent)
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
        // без ступенек.
        let target = mic.level
        let k: Float = target > dispLevel ? 0.28 : 0.16
        dispLevel += (target - dispLevel) * k
        let g = powf(dispLevel, 1.1)
        // Радуга крутится под статичной маской-формой — цвета переливаются, форма
        // выреза стоит на месте. Оборот за 9с.
        orbAngle += .pi * 2 * CGFloat(link.duration > 0 ? link.duration : 1.0 / 120.0) / 9

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        conic.transform = CATransform3DMakeRotation(orbAngle, 0, 0, 1)
        // В покое обводка облегает вырез (scale 1), на голосе свечение разрастается
        // наружу от выреза и ярчает.
        let s = CGFloat(1.0 + CGFloat(g) * 0.7)
        orb.transform = CATransform3DMakeScale(s, s, 1)
        orb.opacity = Float(0.85 + g * 0.15)   // плотнее (меньше прозрачности)
        CATransaction.commit()
    }

    /// Уснуть: остановить display link, обводка облегает вырез (статично).
    private func sleep() {
        animating = false
        link?.invalidate(); link = nil
        dispLevel = 0
        applyIdle()
    }

    private func applyIdle() {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        orb.transform = CATransform3DIdentity
        orb.opacity = 0.85
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
