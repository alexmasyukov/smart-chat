import AppKit
import AVFoundation
import QuartzCore
import CoreImage

// =============================================================================
// NotchGlow — переносимый модуль: голос-реактивное свечение вокруг выреза камеры.
//
// Один файл, ноль зависимостей кроме системных фреймворков (AppKit, AVFoundation,
// QuartzCore, CoreImage). Без окна настроек — только эффект + конфиг.
//
// Использование (минимум):
//     let glow = NotchGlow()                 // конфиг по умолчанию
//     glow.start()                           // показать оверлей + слушать микрофон
//
// С конфигом из JSON:
//     let cfg = NotchGlowConfig.load(Bundle.main.url(forResource: "notch-glow",
//                                                    withExtension: "json"))
//     let glow = NotchGlow(config: cfg)
//     glow.start()
//
// Со своим источником уровня (если аудио уже есть в проекте):
//     var cfg = NotchGlowConfig(); cfg.useBuiltInMic = false
//     let glow = NotchGlow(config: cfg); glow.start()
//     ... glow.setLevel(0.0...1.0)           // из своего аудио-пайплайна
//
// Живая смена настроек: glow.apply(newConfig)
// Остановить: glow.stop()
//
// Требования: macOS 14+ (CADisplayLink на NSView). Для встроенного микрофона
// приложение должно быть .app-бандлом с NSMicrophoneUsageDescription в Info.plist.
//
// ВАЖНО про производительность: здесь НЕТ Metal. CAMetalLayer на прозрачном окне
// заставляет WindowServer перекомпоновывать поверхность каждый vsync → постоянный
// расход CPU. Здесь всё на Core Animation: тяжёлое (маска-обводка + размытие)
// запекается ОДИН раз в CGImage, в рантайме только transform/opacity. В тишине
// CADisplayLink останавливается — кадр статичный, накладных расходов ~0.
// =============================================================================

// MARK: - Конфиг

/// Все настройки эффекта. Codable — можно держать в JSON и передавать в проект.
/// Значения по умолчанию = боевой пресет (одобренный визуал).
public struct NotchGlowConfig: Codable {
    // — форма и ореол (структурные: их смена требует перезапекания маски) —
    /// Толщина ореола: максимальная ширина стека обводок, px.
    public var outlineWidth: CGFloat = 64
    /// Размытие маски (сигма гаусса), px.
    public var blur: CGFloat = 5
    /// Скругление нижних углов выреза, доля от высоты выреза (0…1).
    public var corner: CGFloat = 0.7
    /// Спад яркости к внешнему краю ореола (больше = резче прижат к кромке).
    public var falloff: CGFloat = 1.5
    /// Насыщенность палитры (множитель к базовым оттенкам).
    public var saturation: CGFloat = 1.4
    /// Яркость палитры (множитель).
    public var brightness: CGFloat = 1.0

    // — размер и движение (можно менять на лету, без перезапекания) —
    /// Базовый масштаб по ширине (1.0 = облегает вырез).
    public var scaleX: CGFloat = 1.0
    /// Базовый масштаб по высоте.
    public var scaleY: CGFloat = 1.0
    /// Период оборота перелива цветов, сек.
    public var rotSec: CGFloat = 9
    /// Размах раздувания на голос (прибавка к масштабу при уровне 1.0).
    public var voiceScale: CGFloat = 0.7
    /// Непрозрачность в покое (0…1).
    public var idleOpacity: Float = 0.85
    /// Прирост непрозрачности на голос.
    public var voiceOpacity: Float = 0.15

    // — реакция на звук —
    /// Сглаживание роста на кадр (больше = резче «прыгает»).
    public var attack: Float = 0.28
    /// Сглаживание спада на кадр (меньше = дольше затухает).
    public var release: Float = 0.16
    /// Контраст реакции: level^gamma.
    public var gamma: Float = 1.1
    /// Чувствительность микрофона (множитель к нормированному уровню).
    public var sensitivity: Float = 1.0

    // — интеграция —
    /// Слушать микрофон самому. false — уровень подаёт хост через `setLevel(_:)`.
    public var useBuiltInMic: Bool = true
    /// Сколько секунд тишины до засыпания (остановка CADisplayLink).
    public var holdTime: Double = 0.7
    /// Запас окна вокруг выреза по горизонтали, px (должен вмещать ореол при
    /// максимальном раздувании, иначе свечение обрежется).
    public var marginX: CGFloat = 140
    /// Запас окна вниз от выреза, px.
    public var marginY: CGFloat = 120

    public init() {}

    /// Прочитать конфиг из JSON-файла. Любое поле можно опустить — подставится
    /// значение по умолчанию. При ошибке/nil возвращает дефолтный конфиг.
    public static func load(_ url: URL?) -> NotchGlowConfig {
        guard let url, let data = try? Data(contentsOf: url),
              let cfg = try? JSONDecoder().decode(NotchGlowConfig.self, from: data)
        else { return NotchGlowConfig() }
        return cfg
    }

    /// Прочитать конфиг из JSON-строки.
    public static func load(json: String) -> NotchGlowConfig {
        guard let data = json.data(using: .utf8),
              let cfg = try? JSONDecoder().decode(NotchGlowConfig.self, from: data)
        else { return NotchGlowConfig() }
        return cfg
    }

    /// Сериализовать в JSON (удобно, чтобы сохранить подобранный пресет).
    public func jsonString() -> String {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        return (try? enc.encode(self)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
    }
}

// MARK: - Микрофон

/// Мгновенный уровень громкости 0…1 со сглаживанием (резкий рост, плавный спад).
/// `level` пишется из аудио-потока, читается из главного — гонка безвредна,
/// это чисто визуальная величина.
final class NotchGlowMic {
    private let engine = AVAudioEngine()
    var level: Float = 0
    var sensitivity: Float = 1.0
    /// Сглаживание в самом аудио-потоке (поверх него ещё покадровое в tick).
    var attack: Float = 0.85
    var release: Float = 0.16
    /// Зовётся из аудио-потока при пересечении порога снизу вверх («просыпайся»).
    var onVoice: (() -> Void)?
    private var awake = false

    func start() {
        let input = engine.inputNode
        let fmt = input.inputFormat(forBus: 0)
        guard fmt.channelCount > 0 else { return }
        // Буфер поменьше (512) — уровень обновляется чаще, рост плавнее.
        input.installTap(onBus: 0, bufferSize: 512, format: fmt) { [weak self] buf, _ in
            guard let self, let ch = buf.floatChannelData?[0] else { return }
            let n = Int(buf.frameLength); if n == 0 { return }
            var sum: Float = 0
            for i in 0..<n { let s = ch[i]; sum += s * s }
            let db = 20 * log10(max((sum / Float(n)).squareRoot(), 1e-7))
            // Тишина ~-52 dB, полный голос ~-12 dB → нормируем в 0…1.
            var lvl = (db + 52) / 40 * self.sensitivity
            lvl = min(max(lvl, 0), 1)
            let cur = self.level
            self.level = lvl > cur ? cur + (lvl - cur) * self.attack
                                   : cur + (lvl - cur) * self.release
            // Гистерезис пробуждения: будим на 0.10, взводим обратно ниже 0.04.
            if !self.awake, self.level > 0.10 { self.awake = true; self.onVoice?() }
            else if self.awake, self.level < 0.04 { self.awake = false }
        }
        try? engine.start()
    }

    func stop() { engine.stop() }
}

// MARK: - Геометрия выреза

/// Прямоугольник выреза в координатах экрана (origin снизу-слева).
/// Если notch нет — фиктивный «вырез» по центру верхней грани.
public func notchGlowNotchRect(on screen: NSScreen) -> CGRect {
    let f = screen.frame
    let top = screen.safeAreaInsets.top
    if top > 0, let l = screen.auxiliaryTopLeftArea, let r = screen.auxiliaryTopRightArea {
        return CGRect(x: l.maxX, y: f.maxY - top, width: r.minX - l.maxX, height: top)
    }
    let w: CGFloat = 190, h: CGFloat = 34
    return CGRect(x: f.midX - w / 2, y: f.maxY - h, width: w, height: h)
}

// MARK: - Вид

/// Радужный конический градиент вращается ПОД статичной запечённой маской-обводкой
/// выреза: цвета переливаются, форма стоит на месте. Голос раздувает и ярчит.
final class NotchGlowView: NSView {
    let mic = NotchGlowMic()
    var cfg: NotchGlowConfig
    /// Уровень от внешнего источника, когда useBuiltInMic == false.
    private var externalLevel: Float = 0

    private var notchLocal: CGRect
    private let orb = CALayer()               // контейнер: маска-форма + масштаб
    private let conic = CAGradientLayer()     // радуга; вращается (перелив)
    private let maskLayer = CALayer()
    private var link: CADisplayLink?
    private var lastVoice = CACurrentMediaTime()
    private var angle: CGFloat = 0
    private var dispLevel: Float = 0

    init(frame: NSRect, notchLocal: CGRect, cfg: NotchGlowConfig) {
        self.notchLocal = notchLocal
        self.cfg = cfg
        super.init(frame: frame)
        let host = CALayer()
        layer = host
        wantsLayer = true
        host.backgroundColor = NSColor.clear.cgColor

        let center = CGPoint(x: notchLocal.midX, y: notchLocal.midY)
        let side = max(bounds.width, bounds.height) * 1.8
        conic.frame = CGRect(x: center.x - side / 2, y: center.y - side / 2,
                             width: side, height: side)
        conic.type = .conic
        conic.startPoint = CGPoint(x: 0.5, y: 0.5)
        conic.endPoint = CGPoint(x: 0.5, y: 0.0)

        // Якорь контейнера — в центре выреза, чтобы масштаб раздувал наружу от него.
        orb.bounds = CGRect(origin: .zero, size: bounds.size)
        orb.anchorPoint = CGPoint(x: center.x / bounds.width, y: center.y / bounds.height)
        orb.position = center
        orb.addSublayer(conic)
        maskLayer.frame = bounds
        orb.mask = maskLayer
        host.addSublayer(orb)

        rebake()
        applyIdle()
    }

    required init?(coder: NSCoder) { fatalError() }

    // MARK: запекание (только при смене структурных параметров)

    /// Пересобрать текстуры, зависящие от структурных параметров (форма, ореол,
    /// размытие, цвета). Дорого — зовётся при старте и при apply(), не в кадре.
    func rebake() {
        let scale = window?.backingScaleFactor ?? 2
        conic.colors = palette()
        let rad = min(notchLocal.height, notchLocal.width / 2) * cfg.corner
        maskLayer.contentsScale = scale
        maskLayer.contents = bakeMask(size: bounds.size, scale: scale,
                                      outline: notchOutline(notchLocal, radius: rad))
    }

    private func palette() -> [CGColor] {
        let base: [(CGFloat, CGFloat, CGFloat)] = [
            (0.20, 0.85, 1.00),  // cyan
            (0.30, 0.45, 1.00),  // blue
            (0.65, 0.30, 1.00),  // purple
            (1.00, 0.30, 0.70),  // pink
            (1.00, 0.55, 0.30),  // orange
            (0.20, 0.85, 1.00),  // cyan (замыкание круга)
        ]
        return base.map { rgb in
            let c = NSColor(srgbRed: rgb.0, green: rgb.1, blue: rgb.2, alpha: 1)
                .usingColorSpace(.deviceRGB)!
            var h: CGFloat = 0, s: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
            c.getHue(&h, saturation: &s, brightness: &b, alpha: &a)
            return NSColor(hue: h, saturation: min(s * cfg.saturation, 1),
                           brightness: min(b * cfg.brightness, 1), alpha: 1).cgColor
        }
    }

    /// Контур выреза: открытая «U» — вниз по левой стороне, скруглённый низ,
    /// вверх по правой. Верхней грани нет (она у самого края экрана).
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

    /// Стек обводок от широкой тусклой к узкой яркой → альфа максимальна у самой
    /// кромки выреза и спадает наружу. Гаусс сглаживает ступени между слоями.
    private func bakeMask(size: CGSize, scale: CGFloat, outline: CGPath) -> CGImage? {
        let W = Int(size.width * scale), H = Int(size.height * scale)
        guard W > 0, H > 0 else { return nil }
        let cs = CGColorSpaceCreateDeviceRGB()
        let bi = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let cm = CGContext(data: nil, width: W, height: H, bitsPerComponent: 8,
                                 bytesPerRow: 0, space: cs, bitmapInfo: bi) else { return nil }
        cm.scaleBy(x: scale, y: scale)
        cm.setLineCap(.round); cm.setLineJoin(.round)
        let steps = 6
        for i in 0..<steps {
            let t = CGFloat(i) / CGFloat(steps - 1)          // 0 = широкая тусклая
            let w = cfg.outlineWidth * (1 - t) + 3 * t
            // Нижний порог 0.1: широкая обводка не гаснет в ноль — остаётся мягкий
            // внешний ореол, яркость нарастает к кромке.
            let a = 0.1 + 0.9 * pow(t, cfg.falloff)
            cm.setStrokeColor(CGColor(gray: 1, alpha: a))
            cm.setLineWidth(max(w, 1))
            cm.addPath(outline); cm.strokePath()
        }
        guard let img = cm.makeImage() else { return nil }
        let ci = CIImage(cgImage: img)
        let blurred = ci.clampedToExtent()
            .applyingGaussianBlur(sigma: Double(cfg.blur * scale)).cropped(to: ci.extent)
        return CIContext(options: [.useSoftwareRenderer: false])
            .createCGImage(blurred, from: ci.extent)
    }

    // MARK: сон / пробуждение

    /// Разбудить: CADisplayLink 120 Гц. В тишине сам заснёт через holdTime.
    func wake() {
        lastVoice = CACurrentMediaTime()
        guard link == nil else { return }
        let l = displayLink(target: self, selector: #selector(tick(_:)))
        l.preferredFrameRateRange = CAFrameRateRange(minimum: 80, maximum: 120, preferred: 120)
        l.add(to: .main, forMode: .common)
        link = l
    }

    func sleepNow() {
        link?.invalidate(); link = nil
        dispLevel = 0
        applyIdle()
    }

    /// Подать уровень 0…1 извне (когда useBuiltInMic == false).
    func setExternalLevel(_ v: Float) {
        externalLevel = min(max(v, 0), 1)
        if externalLevel > 0.10 { wake() }
    }

    @objc private func tick(_ link: CADisplayLink) {
        if cfg.useBuiltInMic {
            mic.sensitivity = cfg.sensitivity
        }
        let target = cfg.useBuiltInMic ? mic.level : externalLevel
        let now = CACurrentMediaTime()
        if target > 0.06 { lastVoice = now }
        if now - lastVoice > cfg.holdTime { sleepNow(); return }

        // Уровень сглаживается КАЖДЫЙ кадр: аудио обновляется реже рендера, иначе
        // рост идёт ступеньками.
        let k: Float = target > dispLevel ? cfg.attack : cfg.release
        dispLevel += (target - dispLevel) * k
        let g = powf(dispLevel, cfg.gamma)
        angle += .pi * 2 * CGFloat(link.duration > 0 ? link.duration : 1.0 / 120)
                 / max(cfg.rotSec, 0.5)

        CATransaction.begin(); CATransaction.setDisableActions(true)
        conic.transform = CATransform3DMakeRotation(angle, 0, 0, 1)
        orb.transform = CATransform3DMakeScale(cfg.scaleX * CGFloat(1 + CGFloat(g) * cfg.voiceScale),
                                               cfg.scaleY * CGFloat(1 + CGFloat(g) * cfg.voiceScale), 1)
        orb.opacity = cfg.idleOpacity + Float(g) * cfg.voiceOpacity
        CATransaction.commit()
    }

    private func applyIdle() {
        CATransaction.begin(); CATransaction.setDisableActions(true)
        orb.transform = CATransform3DMakeScale(cfg.scaleX, cfg.scaleY, 1)
        orb.opacity = cfg.idleOpacity
        CATransaction.commit()
    }
}

// MARK: - Публичный контроллер

/// Точка входа модуля: создаёт прозрачный click-through оверлей вокруг выреза,
/// слушает микрофон (или принимает уровень извне) и анимирует свечение.
public final class NotchGlow {
    public private(set) var config: NotchGlowConfig
    private var window: NSWindow?
    private var view: NotchGlowView?
    private var micStarted = false

    public init(config: NotchGlowConfig = NotchGlowConfig()) {
        self.config = config
    }

    /// Показать оверлей и (если включено) запросить доступ к микрофону + слушать.
    /// Звать из главного потока после запуска NSApplication.
    public func start() {
        guard window == nil else { return }
        let screen = NSScreen.screens.first(where: { $0.safeAreaInsets.top > 0 })
            ?? NSScreen.main ?? NSScreen.screens[0]
        let n = notchGlowNotchRect(on: screen)
        // Окно туго вокруг выреза + запас под раздувание (меньше площадь —
        // дешевле композитинг). Вверх запаса нет: вырез у самой кромки экрана.
        let mx = config.marginX, my = config.marginY
        let frame = CGRect(x: n.minX - mx, y: n.minY - my,
                           width: n.width + 2 * mx, height: n.height + my)

        let w = NSWindow(contentRect: frame, styleMask: [.borderless],
                         backing: .buffered, defer: false)
        w.isOpaque = false
        w.backgroundColor = .clear
        w.hasShadow = false
        w.level = NSWindow.Level(rawValue: Int(CGShieldingWindowLevel()))  // выше меню-бара
        w.ignoresMouseEvents = true                                        // click-through
        w.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]

        let notchLocal = CGRect(x: mx, y: my, width: n.width, height: n.height)
        let v = NotchGlowView(frame: NSRect(origin: .zero, size: frame.size),
                              notchLocal: notchLocal, cfg: config)
        w.contentView = v
        w.orderFrontRegardless()
        window = w; view = v

        guard config.useBuiltInMic else { return }
        v.mic.onVoice = { [weak v] in DispatchQueue.main.async { v?.wake() } }
        AVCaptureDevice.requestAccess(for: .audio) { [weak self] ok in
            DispatchQueue.main.async {
                guard ok, let self, let v = self.view, !self.micStarted else { return }
                self.micStarted = true
                v.mic.start()
            }
        }
    }

    /// Убрать оверлей и остановить всё.
    public func stop() {
        view?.sleepNow()
        view?.mic.stop()
        window?.orderOut(nil)
        window = nil; view = nil; micStarted = false
    }

    /// Применить новый конфиг на лету (перезапекает маску — не звать в каждом кадре).
    public func apply(_ cfg: NotchGlowConfig) {
        config = cfg
        guard let v = view else { return }
        v.cfg = cfg
        v.rebake()
        v.wake()   // короткий кадр, чтобы изменения стали видны сразу
    }

    /// Подать уровень 0…1 из своего аудио-источника (useBuiltInMic == false).
    public func setLevel(_ level: Float) {
        view?.setExternalLevel(level)
    }
}
