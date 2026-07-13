import AppKit
import AVFoundation
import QuartzCore

// =============================================================================
// Assistant — нативный macOS-оверлей вокруг выреза под камеру (notch).
//
// Формы/окна у приложения нет. Прозрачный click-through оверлей на самом верхнем
// уровне рисует переливающееся размытое свечение по контуру выреза — как у Siri.
// Микрофон слушается постоянно: пока тихо — мягкое «дыхание», когда говоришь —
// свечение разрастается и ярчает по уровню голоса.
//
// Сборка: build.sh собирает .app-бандл (нужен Info.plist с доступом к микрофону).
// =============================================================================

// MARK: - Микрофон: мгновенный уровень громкости 0..1

/// Слушает вход по умолчанию через AVAudioEngine и держит сглаженный уровень
/// громкости в диапазоне 0..1 (быстрый attack, медленный release).
final class MicLevel {
    private let engine = AVAudioEngine()
    /// Читается из главного потока (аниматором). Пишется из аудио-потока.
    /// Гонка безвредна — это лишь визуальная величина.
    var level: Float = 0

    func start() {
        let input = engine.inputNode
        let fmt = input.inputFormat(forBus: 0)
        // Некоторые устройства отдают 0 каналов до прогрева — тогда пропускаем tap.
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
            // Сглаживание: очень резкий рост (свечение мгновенно «прыгает» на
            // голос), заметно более быстрый спад — чтобы движение было живым.
            let cur = self.level
            self.level = lvl > cur ? cur + (lvl - cur) * 0.85
                                   : cur + (lvl - cur) * 0.16
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
/// Если notch нет — возвращает фиктивный «вырез» по центру верхней грани,
/// чтобы приложение имело смысл и на экранах без выреза.
func notchRect(on screen: NSScreen) -> CGRect {
    let f = screen.frame
    let top = screen.safeAreaInsets.top
    if top > 0, let l = screen.auxiliaryTopLeftArea, let r = screen.auxiliaryTopRightArea {
        return CGRect(x: l.maxX, y: f.maxY - top, width: r.minX - l.maxX, height: top)
    }
    let w: CGFloat = 190, h: CGFloat = 34
    return CGRect(x: f.midX - w / 2, y: f.maxY - h, width: w, height: h)
}

// MARK: - Вид: переливающееся свечение по контуру выреза

/// Один «слой свечения»: конический градиент (переливается цветами), вращается
/// под маской-обводкой выреза, а сверху размывается гауссом → мягкое сияние.
private struct GlowStack {
    let clip = CALayer()            // несёт маску-обводку и блюр; не вращается
    let gradient = CAGradientLayer() // конический градиент, вращается внутри clip
    let mask = CAShapeLayer()       // обводка контура выреза (задаёт форму свечения)
    let blur: CIFilter
    let filterKey: String

    init(key: String) {
        filterKey = key
        blur = CIFilter(name: "CIGaussianBlur")!
        blur.setValue(4, forKey: "inputRadius")
        blur.name = key
    }
}

final class GlowView: NSView {
    private var halo: GlowStack!   // толстый сильно размытый ореол
    private var core: GlowStack!   // тонкая яркая обводка
    private var notchLocal: CGRect = .zero
    private var breath: CGFloat = 0
    let mic = MicLevel()

    // Палитра Siri: холодные переливы с тёплым акцентом; массив замкнут для
    // бесшовного конического градиента.
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

        halo = makeStack(key: "haloBlur", clockwise: true)
        core = makeStack(key: "coreBlur", clockwise: false)
        host.addSublayer(halo.clip)
        host.addSublayer(core.clip)

        startRotation(halo, duration: 14, reverse: false)
        startRotation(core, duration: 9, reverse: true)
    }

    required init?(coder: NSCoder) { fatalError() }

    private var link: CADisplayLink?

    // Привязываемся к дисплею вида: кадры идут на частоте экрана (до 120 Гц на
    // ProMotion). Создаём только когда вид уже в окне (есть к какому дисплею).
    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        guard window != nil, link == nil else { return }
        let l = displayLink(target: self, selector: #selector(tick(_:)))
        l.preferredFrameRateRange = CAFrameRateRange(minimum: 60, maximum: 120, preferred: 120)
        l.add(to: .main, forMode: .common)
        link = l
    }

    private func makeStack(key: String, clockwise: Bool) -> GlowStack {
        let s = GlowStack(key: key)
        s.clip.frame = bounds
        s.clip.masksToBounds = false

        // Конический градиент — большой квадрат с центром в центре выреза,
        // чтобы цвета «оборачивались» вокруг него при вращении.
        let side = max(bounds.width, bounds.height) * 1.6
        let c = CGPoint(x: notchLocal.midX, y: notchLocal.midY)
        s.gradient.frame = CGRect(x: c.x - side / 2, y: c.y - side / 2, width: side, height: side)
        s.gradient.type = .conic
        s.gradient.colors = GlowView.palette
        s.gradient.startPoint = CGPoint(x: 0.5, y: 0.5)
        s.gradient.endPoint = CGPoint(x: 0.5, y: 0.0)
        s.clip.addSublayer(s.gradient)

        // Маска-обводка задаёт форму свечения (U вокруг выреза).
        s.mask.frame = bounds
        s.mask.path = notchOutline(notchLocal, radius: min(notchLocal.height, notchLocal.width / 2) * 0.7)
        s.mask.fillColor = NSColor.clear.cgColor
        s.mask.strokeColor = NSColor.white.cgColor  // маска использует альфу
        s.mask.lineWidth = 6
        s.mask.lineCap = .round
        s.clip.mask = s.mask

        s.clip.filters = [s.blur]
        return s
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

    private func startRotation(_ s: GlowStack, duration: CFTimeInterval, reverse: Bool) {
        let a = CABasicAnimation(keyPath: "transform.rotation.z")
        a.fromValue = reverse ? CGFloat.pi * 2 : 0
        a.toValue = reverse ? 0 : CGFloat.pi * 2
        a.duration = duration
        a.repeatCount = .infinity
        s.gradient.add(a, forKey: "spin")
    }

    @objc private func tick(_ link: CADisplayLink) {
        // Реальное время кадра, чтобы «дыхание» шло одинаково при 60 и 120 Гц.
        breath += CGFloat(link.duration > 0 ? link.duration : 1.0 / 60.0)
        // Тихое «дыхание» в покое — слабое, чтобы речь резко его перебивала.
        let pulse = (sin(breath * 1.1) * 0.5 + 0.5) * 0.08 + 0.02   // 0.02..0.10
        let v = CGFloat(mic.level)
        // Голос доминирует; gamma<1 подчёркивает пики — свечение ярко «прыгает».
        let e = max(v, pulse)
        let g = pow(e, 0.6)

        CATransaction.begin()
        CATransaction.setDisableActions(true)

        // Толщина обводки скачет широко: тонкая нить в покое → жирная лента на голосе.
        core.mask.lineWidth = 2 + g * 22
        halo.mask.lineWidth = 10 + g * 70
        // Больше размытия в целом — мягкое сияние, на пиках разливается ещё сильнее.
        core.clip.setValue(4 + g * 8, forKeyPath: "filters.coreBlur.inputRadius")
        halo.clip.setValue(18 + g * 44, forKeyPath: "filters.haloBlur.inputRadius")
        // Яркость тоже прыгает: приглушено в покое, вспыхивает на речи.
        core.clip.opacity = Float(0.35 + g * 0.65)
        halo.clip.opacity = Float(0.18 + g * 0.82)

        CATransaction.commit()
    }
}

// MARK: - App delegate

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var view: GlowView!

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Экран с вырезом (обычно встроенный); иначе — основной.
        let screen = NSScreen.screens.first(where: { $0.safeAreaInsets.top > 0 })
            ?? NSScreen.main
            ?? NSScreen.screens[0]

        let n = notchRect(on: screen)
        // Запас вокруг выреза под свечение: по бокам и вниз (вверх некуда — край).
        // Большой — блюр/толщина сильно разрастаются на пиках голоса.
        let mx: CGFloat = 260, my: CGFloat = 220
        let frame = CGRect(x: n.minX - mx, y: n.minY - my,
                           width: n.width + 2 * mx, height: n.height + my)

        window = NSWindow(contentRect: frame, styleMask: [.borderless],
                          backing: .buffered, defer: false)
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = false
        // Выше строки меню и выреза — рисуем прямо в зоне notch.
        window.level = NSWindow.Level(rawValue: Int(CGShieldingWindowLevel()))
        window.ignoresMouseEvents = true                 // click-through
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]

        // Вырез в локальных координатах вида.
        let notchLocal = CGRect(x: mx, y: my, width: n.width, height: n.height)
        view = GlowView(frame: NSRect(origin: .zero, size: frame.size), notchLocal: notchLocal)
        window.contentView = view
        window.orderFrontRegardless()

        FileHandle.standardError.write("[assistant] окно \(frame) вырез \(n)\n".data(using: .utf8)!)

        // Микрофон только после разрешения TCC.
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
