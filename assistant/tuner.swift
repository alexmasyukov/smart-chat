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

struct Params: Codable {
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
    /// Программно выставить значение (при загрузке шаблона) без вызова onChange.
    func setValue(_ v: Double) {
        slider.doubleValue = v
        valueLabel.stringValue = String(format: fmt, v)
    }
}

/// Именованный шаблон настроек, сохраняется на диск.
struct Preset: Codable { var name: String; var params: Params }

/// Описание одного ползунка + как он читает/пишет поле Params (для загрузки
/// шаблонов: по нему выставляем и слайдер, и параметр).
struct Row {
    let title: String, lo: Double, hi: Double, def: Double, fmt: String, bake: Bool
    let get: (Params) -> Double
    let set: (inout Params, Double) -> Void
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    var panel: NSWindow!
    var overlayWin: NSWindow!
    var overlay: GlowOverlay!
    let dump = NSTextField(wrappingLabelWithString: "")
    var sliders: [Slider] = []
    var presets: [Preset] = []
    let popup = NSPopUpButton(frame: CGRect(x: 15, y: 0, width: 240, height: 24), pullsDown: false)

    let rows: [Row] = [
        Row(title: "Ширина обёртки", lo: 0.5, hi: 2.0, def: 1.0, fmt: "%.2f", bake: false,
            get: { Double($0.scaleX) }, set: { $0.scaleX = $1 }),
        Row(title: "Высота обёртки", lo: 0.5, hi: 2.0, def: 1.0, fmt: "%.2f", bake: false,
            get: { Double($0.scaleY) }, set: { $0.scaleY = $1 }),
        Row(title: "Толщина ореола", lo: 20, hi: 140, def: 64, fmt: "%.0f", bake: true,
            get: { Double($0.outlineWidth) }, set: { $0.outlineWidth = $1 }),
        Row(title: "Размытие", lo: 0, hi: 30, def: 5, fmt: "%.0f", bake: true,
            get: { Double($0.blur) }, set: { $0.blur = $1 }),
        Row(title: "Скругление углов", lo: 0, hi: 1, def: 0.7, fmt: "%.2f", bake: true,
            get: { Double($0.corner) }, set: { $0.corner = $1 }),
        Row(title: "Спад к краю", lo: 0.5, hi: 3, def: 1.5, fmt: "%.2f", bake: true,
            get: { Double($0.falloff) }, set: { $0.falloff = $1 }),
        Row(title: "Насыщенность", lo: 0.5, hi: 2.0, def: 1.4, fmt: "%.2f", bake: true,
            get: { Double($0.saturation) }, set: { $0.saturation = $1 }),
        Row(title: "Яркость цвета", lo: 0.4, hi: 1.0, def: 1.0, fmt: "%.2f", bake: true,
            get: { Double($0.brightness) }, set: { $0.brightness = $1 }),
        Row(title: "Период вращения, с", lo: 2, hi: 30, def: 9, fmt: "%.1f", bake: false,
            get: { Double($0.rotSec) }, set: { $0.rotSec = $1 }),
        Row(title: "Размах на голос", lo: 0, hi: 2.5, def: 0.7, fmt: "%.2f", bake: false,
            get: { Double($0.voiceScale) }, set: { $0.voiceScale = $1 }),
        Row(title: "Прозрачность покоя", lo: 0, hi: 1, def: 0.85, fmt: "%.2f", bake: false,
            get: { Double($0.idleOpacity) }, set: { $0.idleOpacity = Float($1) }),
        Row(title: "Прирост прозрачности", lo: 0, hi: 1, def: 0.15, fmt: "%.2f", bake: false,
            get: { Double($0.voiceOpacity) }, set: { $0.voiceOpacity = Float($1) }),
        Row(title: "Attack (рост)", lo: 0.05, hi: 1, def: 0.28, fmt: "%.2f", bake: false,
            get: { Double($0.attack) }, set: { $0.attack = Float($1) }),
        Row(title: "Release (спад)", lo: 0.05, hi: 1, def: 0.16, fmt: "%.2f", bake: false,
            get: { Double($0.release) }, set: { $0.release = Float($1) }),
        Row(title: "Gamma (контраст)", lo: 0.4, hi: 2.5, def: 1.1, fmt: "%.2f", bake: false,
            get: { Double($0.gamma) }, set: { $0.gamma = Float($1) }),
        Row(title: "Чувствительность", lo: 0.5, hi: 2.0, def: 1.0, fmt: "%.2f", bake: false,
            get: { Double($0.sensitivity) }, set: { $0.sensitivity = Float($1) }),
    ]

    // MARK: пресеты на диске (~/Library/Application Support/AssistantTuner/presets.json)
    private func presetsURL() -> URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("AssistantTuner", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("presets.json")
    }
    private func loadPresets() {
        guard let data = try? Data(contentsOf: presetsURL()),
              let list = try? JSONDecoder().decode([Preset].self, from: data) else { return }
        presets = list
    }
    private func savePresetsToDisk() {
        if let data = try? JSONEncoder().encode(presets) {
            try? data.write(to: presetsURL())
        }
    }

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

        loadPresets()
        buildPanel()

        AVCaptureDevice.requestAccess(for: .audio) { [weak self] ok in
            DispatchQueue.main.async { if ok { self?.overlay.mic.start() } }
        }
    }

    private func buildPanel() {
        let W: CGFloat = 520
        let presetH: CGFloat = 40
        let panelH = presetH + CGFloat(rows.count) * 26 + 120
        let container = NSView(frame: CGRect(x: 0, y: 0, width: W, height: panelH))

        // Верхний ряд: селект шаблонов + Сохранить + Удалить + Дефолт.
        popup.target = self; popup.action = #selector(selectPreset)
        popup.frame = CGRect(x: 15, y: panelH - 32, width: 185, height: 24)
        let saveBtn = NSButton(title: "Сохранить…", target: self, action: #selector(savePreset))
        saveBtn.frame = CGRect(x: 204, y: panelH - 33, width: 100, height: 26)
        let delBtn = NSButton(title: "Удалить", target: self, action: #selector(deletePreset))
        delBtn.frame = CGRect(x: 306, y: panelH - 33, width: 72, height: 26)
        let defBtn = NSButton(title: "Дефолт", target: self, action: #selector(loadDefaults))
        defBtn.frame = CGRect(x: 380, y: panelH - 33, width: 90, height: 26)
        container.addSubview(popup); container.addSubview(saveBtn)
        container.addSubview(delBtn); container.addSubview(defBtn)
        refreshPopup()

        // Ползунки.
        var y = panelH - presetH
        for r in rows {
            y -= 26
            let s = Slider(title: r.title, min: r.lo, max: r.hi, def: r.def, format: r.fmt) { [weak self] v in
                guard let self else { return }
                r.set(&self.overlay.p, Double(v))
                if r.bake { self.bake() } else { self.live() }
            }
            s.frame.origin = CGPoint(x: 15, y: y)
            container.addSubview(s)
            sliders.append(s)
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

    // MARK: применение / пресеты

    /// Применить Params ко всем ползункам и к эффекту (при выборе шаблона).
    private func applyParams(_ p: Params) {
        overlay.p = p
        for (i, r) in rows.enumerated() { sliders[i].setValue(r.get(p)) }
        overlay.rebake()
        dump.stringValue = p.dump()
    }

    private func refreshPopup() {
        popup.removeAllItems()
        popup.addItem(withTitle: presets.isEmpty ? "— нет шаблонов —" : "— выбрать шаблон —")
        for pr in presets { popup.addItem(withTitle: pr.name) }
    }

    @objc private func selectPreset() {
        let idx = popup.indexOfSelectedItem - 1   // 0 — плейсхолдер
        guard idx >= 0, idx < presets.count else { return }
        applyParams(presets[idx].params)
    }

    @objc private func savePreset() {
        let a = NSAlert()
        a.messageText = "Сохранить шаблон"
        a.informativeText = "Имя шаблона (если совпадает — перезапишется):"
        let tf = NSTextField(frame: CGRect(x: 0, y: 0, width: 240, height: 24))
        a.accessoryView = tf
        a.addButton(withTitle: "Сохранить"); a.addButton(withTitle: "Отмена")
        guard a.runModal() == .alertFirstButtonReturn else { return }
        let name = tf.stringValue.trimmingCharacters(in: .whitespaces)
        guard !name.isEmpty else { return }
        if let i = presets.firstIndex(where: { $0.name == name }) {
            presets[i].params = overlay.p          // перезапись
        } else {
            presets.append(Preset(name: name, params: overlay.p))
        }
        savePresetsToDisk()
        refreshPopup()
        popup.selectItem(withTitle: name)
    }

    @objc private func deletePreset() {
        let idx = popup.indexOfSelectedItem - 1
        guard idx >= 0, idx < presets.count else { return }
        let name = presets[idx].name
        let a = NSAlert()
        a.messageText = "Удалить шаблон «\(name)»?"
        a.informativeText = "Действие нельзя отменить."
        a.alertStyle = .warning
        a.addButton(withTitle: "Да, удалить")
        a.addButton(withTitle: "Нет")
        guard a.runModal() == .alertFirstButtonReturn else { return }
        presets.remove(at: idx)
        savePresetsToDisk()
        refreshPopup()
    }

    /// Сбросить все ползунки и эффект на дефолтные настройки (стартовый вид main).
    @objc private func loadDefaults() {
        applyParams(Params())
        popup.selectItem(at: 0)
    }

    private func live() { dump.stringValue = overlay.p.dump() }         // без пересборки
    private func bake() { overlay.rebake(); dump.stringValue = overlay.p.dump() }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
