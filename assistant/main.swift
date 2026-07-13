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

// MARK: - Вид: свечение на Core Animation (без Metal)

// Один «слой свечения»: конический радужный градиент, маскированный ЗАПЕЧЁННОЙ
// размытой обводкой выреза. Градиент вращается через render-server-анимацию
// (перелив), маска-картинка статична и уже размыта — мягкие края бесплатно.
private final class Glow {
    let holder = CALayer()           // несёт маску-картинку; масштабируется по голосу
    let gradient = CAGradientLayer() // радужный конический градиент; вращается
    init() {}
}

final class GlowView: NSView {
    let mic = MicLevel()
    private var notchLocal: CGRect
    private var core = Glow()
    private var halo = Glow()
    private var animating = false
    private var link: CADisplayLink?
    private var lastVoice = CACurrentMediaTime()
    private let holdTime = 0.7        // сколько ещё анимировать после тишины
    private var angleCore: CGFloat = 0
    private var angleHalo: CGFloat = 0

    // Плотное замкнутое кольцо оттенков (cyan→blue→purple→magenta→pink→…→cyan):
    // 24 близких стопа вместо шести далёких → переходы плавные, без «границ радуги».
    // Идём по hue туда-обратно (170°→330°→170°), минуя зелёный/жёлтый (холодный Siri).
    private static let palette: [CGColor] = {
        let stops = 24
        return (0..<stops).map { i -> CGColor in
            let f = Double(i) / Double(stops - 1)
            let tri = f < 0.5 ? f * 2 : (1 - f) * 2
            let hue = (170.0 + tri * 160.0) / 360.0
            return NSColor(hue: CGFloat(hue), saturation: 0.82, brightness: 1.0, alpha: 1).cgColor
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
        let outline = notchOutline(notchLocal,
                                   radius: min(notchLocal.height, notchLocal.width / 2) * 0.7)
        let center = CGPoint(x: notchLocal.midX, y: notchLocal.midY)

        // halo — широкое сильно размытое сияние; core — узкая яркая обводка.
        setup(halo, outlineLineWidth: 22, blur: 34, idleOpacity: 0.35, host: host,
              center: center, scale: scale, outline: outline)
        setup(core, outlineLineWidth: 6, blur: 9, idleOpacity: 0.85, host: host,
              center: center, scale: scale, outline: outline)
    }

    required init?(coder: NSCoder) { fatalError() }

    private func setup(_ glow: Glow, outlineLineWidth lw: CGFloat, blur: CGFloat,
                       idleOpacity: Float, host: CALayer, center: CGPoint,
                       scale: CGFloat, outline: CGPath) {
        // Конический радужный градиент — квадрат с центром в вырезе, чтобы цвета
        // «оборачивались» вокруг него при вращении.
        let side = max(bounds.width, bounds.height) * 1.7
        glow.gradient.frame = CGRect(x: center.x - side / 2, y: center.y - side / 2,
                                     width: side, height: side)
        glow.gradient.type = .conic
        glow.gradient.colors = GlowView.palette
        glow.gradient.startPoint = CGPoint(x: 0.5, y: 0.5)
        glow.gradient.endPoint = CGPoint(x: 0.5, y: 0.0)

        glow.holder.frame = bounds
        // Масштабируем свечение из центра выреза (голос «раздувает» сияние наружу).
        glow.holder.anchorPoint = CGPoint(x: center.x / bounds.width,
                                          y: center.y / bounds.height)
        glow.holder.position = center
        glow.holder.opacity = idleOpacity
        glow.holder.addSublayer(glow.gradient)

        // Маска: ЗАПЕЧЁННАЯ один раз размытая обводка (настоящий гаусс, но не в
        // рантайме каждый кадр). Даёт мягкие размытые края почти бесплатно.
        if let img = bakeGlow(size: bounds.size, scale: scale, outline: outline,
                              lineWidth: lw, blurSigma: blur) {
            let mask = CALayer()
            mask.frame = bounds
            mask.contentsScale = scale
            mask.contents = img
            glow.holder.mask = mask
        }
        host.addSublayer(glow.holder)
    }

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

    /// Запекает размытую обводку в CGImage ОДИН раз: рисуем штрих пути, применяем
    /// гаусс через Core Image единожды. В рантайме фильтр больше не гоняется.
    private func bakeGlow(size: CGSize, scale: CGFloat, outline: CGPath,
                          lineWidth: CGFloat, blurSigma: CGFloat) -> CGImage? {
        let w = Int(size.width * scale), h = Int(size.height * scale)
        guard w > 0, h > 0,
              let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        ctx.scaleBy(x: scale, y: scale)
        ctx.setStrokeColor(CGColor(gray: 1, alpha: 1))
        ctx.setLineWidth(lineWidth)
        ctx.setLineCap(.round)
        ctx.setLineJoin(.round)
        ctx.addPath(outline)
        ctx.strokePath()
        guard let base = ctx.makeImage() else { return nil }
        let ci = CIImage(cgImage: base)
        let blurred = ci.clampedToExtent()
            .applyingGaussianBlur(sigma: Double(blurSigma * scale))
            .cropped(to: ci.extent)
        return CIContext(options: [.useSoftwareRenderer: false])
            .createCGImage(blurred, from: ci.extent)
    }

    // MARK: сон/пробуждение

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        // Голос будит спящее свечение (переход на main из аудио-потока).
        mic.onVoice = { [weak self] in
            DispatchQueue.main.async { self?.wake() }
        }
        applyIdle()   // стартуем со статичного покойного кадра
    }

    /// Разбудить: запустить CADisplayLink на 120 Гц. Всю анимацию (перелив +
    /// реакцию на голос) ведём покадрово синхронно с дисплеем — это доказанно
    /// плавные 120 на прозрачном окне (проверено fpstest), в отличие от Timer,
    /// который не привязан к vsync и давал дёрганье.
    func wake() {
        lastVoice = CACurrentMediaTime()
        if animating { return }
        animating = true
        let l = displayLink(target: self, selector: #selector(tick(_:)))
        l.preferredFrameRateRange = CAFrameRateRange(minimum: 80, maximum: 120, preferred: 120)
        l.add(to: .main, forMode: .common)
        link = l
        FileHandle.standardError.write("[assistant] проснулся (голос)\n".data(using: .utf8)!)
    }

    @objc private func tick(_ link: CADisplayLink) {
        let now = CACurrentMediaTime()
        let lv = mic.level
        if lv > 0.06 { lastVoice = now }
        if now - lastVoice > holdTime { sleep(); return }

        // Угол вращения ведём по реальному времени кадра — оборот за 9с/14с.
        let dt = CGFloat(link.duration > 0 ? link.duration : 1.0 / 120.0)
        angleCore += .pi * 2 * dt / 9
        angleHalo += .pi * 2 * dt / 14
        let g = powf(max(lv, 0.06), 0.6)

        CATransaction.begin()
        CATransaction.setDisableActions(true)
        core.gradient.transform = CATransform3DMakeRotation(angleCore, 0, 0, 1)
        halo.gradient.transform = CATransform3DMakeRotation(angleHalo, 0, 0, 1)
        // Голос раздувает свечение наружу и ярчит — резкий «прыжок».
        let sc = CGFloat(1.0 + g * 1.4)
        let sh = CGFloat(1.0 + g * 2.0)
        core.holder.transform = CATransform3DMakeScale(sc, sc, 1)
        halo.holder.transform = CATransform3DMakeScale(sh, sh, 1)
        core.holder.opacity = 0.6 + g * 0.4
        halo.holder.opacity = 0.25 + g * 0.6
        CATransaction.commit()
    }

    /// Уснуть: остановить display link, зафиксировать статичный покойный кадр.
    private func sleep() {
        animating = false
        link?.invalidate(); link = nil
        applyIdle()
        FileHandle.standardError.write("[assistant] уснул (тишина) — анимаций нет\n".data(using: .utf8)!)
    }

    /// Статичное мягкое свечение в покое: без анимаций, WindowServer кеширует.
    private func applyIdle() {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        core.holder.transform = CATransform3DIdentity
        halo.holder.transform = CATransform3DIdentity
        core.holder.opacity = 0.7
        halo.holder.opacity = 0.28
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
