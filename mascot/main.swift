import AppKit
import Foundation

// =============================================================================
// smart-chat mascot — нативная Swift-оболочка (десктоп-пет).
// UI: прозрачное плавающее окно с пиксельным персонажем, облачком и вводом.
// «Мозг» — наш Node-движок (src/engine.js), запущенный как sidecar-процесс,
// общение по NDJSON через stdin/stdout.
// =============================================================================

// MARK: - Спрайт персонажа (16×16 пиксель-арт робота)

enum Sprite {
    static let palette: [Character: NSColor] = [
        "o": NSColor(srgbRed: 0.086, green: 0.133, blue: 0.180, alpha: 1), // контур
        "b": NSColor(srgbRed: 0.50,  green: 0.80,  blue: 1.00,  alpha: 1), // корпус
        "d": NSColor(srgbRed: 0.25,  green: 0.62,  blue: 0.88,  alpha: 1), // тень корпуса
        "a": NSColor(srgbRed: 0.73,  green: 0.97,  blue: 1.00,  alpha: 1), // глаза/антенна (свечение)
        "m": NSColor(srgbRed: 0.086, green: 0.133, blue: 0.180, alpha: 1), // рот
        "s": NSColor(srgbRed: 1.00,  green: 0.95,  blue: 0.65,  alpha: 1), // искры
    ]

    static let base: [String] = [
        "................",
        ".......aa.......",
        ".......oo.......",
        "...oooooooooo...",
        "...obbbbbbbbo...",
        "...obaabbaabo...",
        "...obaabbaabo...",
        "...obbbbbbbbo...",
        "...obbbbbbbbo...",
        "...obbmmmmbbo...",
        "...obbbbbbbbo...",
        "...obbbbbbbbo...",
        "...oddddddddo...",
        "...oooooooooo...",
        "....oo..oo......",
        "................",
    ]

    static func frame(_ overrides: [Int: String]) -> [String] {
        var rows = base
        for (i, s) in overrides { rows[i] = s }
        return rows
    }

    // Производные кадры
    static let blink   = frame([5: "...obbbbbbbbo...", 6: "...oboobboobo..."])
    static let think   = frame([1: "......aaaa......"])
    static let talkOpen  = frame([9: "...obbmmmmbbo...", 10: "...obbmmmmbbo..."])
    static let talkClosed = frame([9: "...obbbmmbbbo..."])
    static let workA = frame([
        2: ".s.....oo.......",
        5: "...obaabbaabo.s.",
        7: ".bbobbbbbbbbobb.",
        8: ".bbobbbbbbbbobb.",
    ])
    static let workB = frame([
        4: ".s.obbbbbbbbo...",
        7: ".bbobbbbbbbbobb.",
        8: ".bbobbbbbbbbobb.",
        10: "...obbbbbbbbo.s.",
    ])

    static func color(_ ch: Character) -> NSColor? { palette[ch] }
}

// MARK: - Состояние персонажа

enum MascotState {
    case idle, thinking, talking, working
}

// MARK: - Окно (borderless должно уметь становиться key для ввода)

final class PetWindow: NSWindow {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
}

// MARK: - API-клиент (HTTP + SSE к локальному серверу src/server.js)

final class ApiClient: NSObject, URLSessionDataDelegate {
    private let base: String
    private var session: URLSession!
    private var sseBuffer = ""

    var onEvent: (([String: Any]) -> Void)?

    override init() {
        base = ProcessInfo.processInfo.environment["API_URL"] ?? "http://127.0.0.1:8787"
        super.init()
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 600
        cfg.timeoutIntervalForResource = 3600
        session = URLSession(configuration: cfg, delegate: self, delegateQueue: nil)
    }

    private func emit(_ dict: [String: Any]) {
        DispatchQueue.main.async { [weak self] in self?.onEvent?(dict) }
    }

    func start() {
        connectEvents()
    }

    // Подписка на широковещательный поток сервера. Облако только читает события.
    private func connectEvents() {
        guard let url = URL(string: base + "/api/events") else { return }
        var req = URLRequest(url: url)
        req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        req.timeoutInterval = 86400
        sseBuffer = ""
        session.dataTask(with: req).resume()
    }

    private func scheduleReconnect() {
        DispatchQueue.global().asyncAfter(deadline: .now() + 1.0) { [weak self] in
            self?.connectEvents()
        }
    }

    // SSE приходит кусками — режем по "\n\n" на события.
    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        sseBuffer += String(decoding: data, as: UTF8.self)
        while let range = sseBuffer.range(of: "\n\n") {
            let block = String(sseBuffer[sseBuffer.startIndex..<range.lowerBound])
            sseBuffer.removeSubrange(sseBuffer.startIndex..<range.upperBound)
            parseBlock(block)
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        // Поток событий разорвался (сервер ещё не поднят / перезапуск) — переподключаемся.
        emit(["event": "disconnected"])
        scheduleReconnect()
    }

    private func parseBlock(_ block: String) {
        var event = "message"
        var dataStr = ""
        for line in block.split(separator: "\n", omittingEmptySubsequences: false) {
            if line.hasPrefix("event:") {
                event = line.dropFirst(6).trimmingCharacters(in: .whitespaces)
            } else if line.hasPrefix("data:") {
                dataStr += line.dropFirst(5).trimmingCharacters(in: .whitespaces)
            }
        }
        var dict: [String: Any] = [:]
        if let d = dataStr.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
            dict = obj
        }
        dict["event"] = event
        emit(dict)
    }
}

// MARK: - Корневое представление (рисует пета + облачко, держит контролы)

final class RootView: NSView, NSWindowDelegate {
    let api: ApiClient

    private var state: MascotState = .idle
    private var tick: Int = 0
    private var bubbleText: String = ""   // пусто, пока модель не ответит
    private var settingsOpen = false      // пока настройки открыты — облако показано
    private let previewText = "Пример сообщения — настрой размер пета и шрифт."
    private var toolBadge: String?
    private var toolBadgeUntil: Int = 0
    private var dismissAtTick: Int = 0   // когда скрыть облако (0 = не скрывать)
    private var placed = false   // окно уже спозиционировано хотя бы раз

    private let closeButton = NSButton()
    private var settingsWindow: NSWindow?

    // Геометрия. Размер персонажа и шрифт настраиваются в «Настройках».
    private var charBox: CGFloat = 112
    private var fontSize: CGFloat = 13
    private let sideMargin: CGFloat = 14
    private let topMargin: CGFloat = 10
    private let bottomMargin: CGFloat = 12
    private let tailH: CGFloat = 12
    private let maxBubbleW: CGFloat = 240
    private let bubblePad: CGFloat = 11

    private var textAttrs: [NSAttributedString.Key: Any] {
        let para = NSMutableParagraphStyle()
        para.lineBreakMode = .byWordWrapping
        return [
            .font: NSFont.systemFont(ofSize: fontSize),
            .foregroundColor: NSColor.white,
            .paragraphStyle: para,
        ]
    }

    override var isFlipped: Bool { true }

    init(frame: NSRect, api: ApiClient) {
        self.api = api
        super.init(frame: frame)
        setupControls()
        Timer.scheduledTimer(withTimeInterval: 0.09, repeats: true) { [weak self] _ in
            self?.tickAnimation()
        }
    }

    required init?(coder: NSCoder) { fatalError() }

    // MARK: контролы (только кнопка закрытия — облако без чата и выбора модели)

    private func setupControls() {
        let w = bounds.width
        closeButton.frame = NSRect(x: w - 28, y: 6, width: 20, height: 20)
        closeButton.title = "✕"
        closeButton.bezelStyle = .circular
        closeButton.font = NSFont.systemFont(ofSize: 10)
        closeButton.target = self
        closeButton.action = #selector(closeApp)
        addSubview(closeButton)
    }

    @objc private func closeApp() {
        NSApp.terminate(nil)
    }

    // MARK: контекстное меню (правый клик по пету)

    override func rightMouseDown(with event: NSEvent) {
        let menu = NSMenu()
        let settings = NSMenuItem(title: "Настройки", action: #selector(openSettings), keyEquivalent: "")
        let quit = NSMenuItem(title: "Выход", action: #selector(closeApp), keyEquivalent: "")
        settings.target = self
        quit.target = self
        menu.addItem(settings)
        menu.addItem(.separator())
        menu.addItem(quit)
        NSMenu.popUpContextMenu(menu, with: event, for: self)
    }

    @objc private func openSettings() {
        if settingsWindow == nil { settingsWindow = makeSettingsWindow() }
        settingsOpen = true            // пока настройки открыты — облако видно
        dismissAtTick = 0              // не скрывать облако в это время
        relayout()
        settingsWindow?.center()
        settingsWindow?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    // Закрытие окна настроек — облако снова прячется (если нет ответа).
    func windowWillClose(_ notification: Notification) {
        settingsOpen = false
        relayout()
    }

    // Диапазоны параметров (слайдеры работают по шкале 0–100 с шагом 5).
    private let RANGE_SIZE = (70.0, 200.0)
    private let RANGE_FONT = (10.0, 24.0)
    private let RANGE_OPACITY = (0.3, 1.0)
    private func norm(_ v: Double, _ r: (Double, Double)) -> Double { (v - r.0) / (r.1 - r.0) * 100 }
    private func denorm(_ v: Double, _ r: (Double, Double)) -> Double { r.0 + v / 100 * (r.1 - r.0) }

    private func makeSettingsWindow() -> NSWindow {
        let win = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 320, height: 210),
            styleMask: [.titled, .closable], backing: .buffered, defer: false)
        win.title = "Настройки пета"
        win.isReleasedWhenClosed = false
        win.delegate = self

        let stack = NSStackView()
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 16
        stack.edgeInsets = NSEdgeInsets(top: 18, left: 18, bottom: 18, right: 18)
        stack.translatesAutoresizingMaskIntoConstraints = false

        stack.addArrangedSubview(sliderRow("Размер пета",
                                           value: norm(Double(charBox), RANGE_SIZE), action: #selector(petSizeChanged)))
        stack.addArrangedSubview(sliderRow("Размер шрифта сообщений",
                                           value: norm(Double(fontSize), RANGE_FONT), action: #selector(fontChanged)))
        stack.addArrangedSubview(sliderRow("Прозрачность пета",
                                           value: norm(Double(window?.alphaValue ?? 1.0), RANGE_OPACITY), action: #selector(opacityChanged)))

        let content = win.contentView!
        content.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: content.leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: content.trailingAnchor),
            stack.topAnchor.constraint(equalTo: content.topAnchor),
        ])
        return win
    }

    // Строка настройки: подпись + NSSlider по шкале 0–100 с засечками (шаг 5).
    private func sliderRow(_ title: String, value: Double, action: Selector) -> NSView {
        let label = NSTextField(labelWithString: title)
        label.font = NSFont.systemFont(ofSize: 12)

        let slider = NSSlider(value: value, minValue: 0, maxValue: 100, target: self, action: action)
        slider.isContinuous = true
        slider.numberOfTickMarks = 21               // 0,5,10,…,100
        slider.allowsTickMarkValuesOnly = true       // привязка строго к рискам
        slider.tickMarkPosition = .below
        slider.translatesAutoresizingMaskIntoConstraints = false
        slider.widthAnchor.constraint(equalToConstant: 284).isActive = true

        let row = NSStackView(views: [label, slider])
        row.orientation = .vertical
        row.alignment = .leading
        row.spacing = 5
        return row
    }

    @objc private func petSizeChanged(_ sender: NSSlider) {
        charBox = CGFloat(denorm(sender.doubleValue, RANGE_SIZE))
        relayout()
    }

    @objc private func fontChanged(_ sender: NSSlider) {
        fontSize = CGFloat(denorm(sender.doubleValue, RANGE_FONT))
        relayout()
    }

    @objc private func opacityChanged(_ sender: NSSlider) {
        window?.alphaValue = CGFloat(denorm(sender.doubleValue, RANGE_OPACITY))
    }

    // MARK: события сервера (только чтение — облако зеркалит ответ)

    func handle(_ e: [String: Any]) {
        switch e["event"] as? String {
        case "hello":
            state = .idle         // ничего не показываем — ждём ответа модели
        case "prompt":
            bubbleText = ""       // новый ход — чистим пузырь
            dismissAtTick = 0     // отменяем плановое скрытие
            state = .thinking
        case "state":
            switch e["value"] as? String {
            case "thinking": state = .thinking
            case "talking":  state = .talking
            case "working":  state = .working
            case "idle":     state = .idle
            default: break
            }
        case "token":
            if let t = e["text"] as? String { bubbleText += t }
            dismissAtTick = 0     // пока печатает — не скрываем
        case "tool":
            if (e["phase"] as? String) == "start", let name = e["name"] as? String {
                let short = name.components(separatedBy: "__").last ?? name
                toolBadge = "⚙ " + short
                toolBadgeUntil = tick + 30
            }
        case "done":
            state = .idle
            scheduleDismiss()     // прячем облако через паузу
        case "error":
            bubbleText = "⚠️ " + (e["message"] as? String ?? "ошибка")
            state = .idle
            scheduleDismiss()
        case "disconnected":
            // тихо ждём переподключения, ничего не меняем
            break
        default:
            break
        }
        relayout()
    }

    func initialLayout() { relayout() }

    // MARK: динамическая раскладка — размер окна/облака по объёму ответа

    // Текст в облаке: ответ модели, либо «…» во время ответа, иначе ничего.
    // Пока открыты настройки — показываем облако (ответ или превью-текст).
    private func displayText() -> String? {
        if !bubbleText.isEmpty { return bubbleText }
        if settingsOpen { return previewText }
        if state == .thinking || state == .working { return "…" }
        return nil
    }

    private func bubbleSize(for text: String) -> CGSize {
        let maxTextW = maxBubbleW - 2 * bubblePad
        let str = NSAttributedString(string: text, attributes: textAttrs)
        let bound = str.boundingRect(
            with: CGSize(width: maxTextW, height: .greatestFiniteMagnitude),
            options: [.usesLineFragmentOrigin])
        let w = min(maxBubbleW, ceil(bound.width) + 2 * bubblePad)
        let h = ceil(bound.height) + 2 * bubblePad
        return CGSize(width: max(w, 54), height: max(h, 34))
    }

    // Пересчитывает размер окна под текущий ответ и держит нижний край/центр на месте.
    private func relayout() {
        guard let window = window, let screen = window.screen ?? NSScreen.main else {
            needsDisplay = true
            return
        }

        let text = displayText()
        let bs = text.map { bubbleSize(for: $0) }
        let bubbleW = bs?.width ?? 0
        let bubbleH = bs?.height ?? 0
        let bubbleBlock = bs == nil ? 0 : (bubbleH + tailH)

        let contentW = max(charBox, bubbleW) + 2 * sideMargin
        let contentH = topMargin + bubbleBlock + charBox + bottomMargin

        let vf = screen.visibleFrame
        let old = window.frame
        // первый раз — правый нижний угол; дальше держим центр по X и нижний край
        // (чтобы при росте облако «раскрывалось» вверх и не убегало от места, куда перетащили)
        let centerX = placed ? old.midX : (vf.maxX - 24 - contentW / 2)
        let bottomY = placed ? old.minY : (vf.minY + 24)
        placed = true

        var x = centerX - contentW / 2
        var y = bottomY
        x = min(max(x, vf.minX + 8), vf.maxX - contentW - 8)
        if y + contentH > vf.maxY { y = vf.maxY - contentH }
        y = max(y, vf.minY + 8)

        window.setFrame(NSRect(x: x, y: y, width: contentW, height: contentH), display: true)
        frame = NSRect(origin: .zero, size: NSSize(width: contentW, height: contentH))
        closeButton.frame = NSRect(x: contentW - 24, y: 5, width: 18, height: 18)
        closeButton.isHidden = (bs == nil) // прячем крестик, когда облака нет
        needsDisplay = true
    }

    // MARK: анимация

    // Время скрытия зависит от длины ответа: 2.5–9 сек (длиннее — дольше держим).
    private func scheduleDismiss() {
        let seconds = min(9.0, max(2.5, Double(bubbleText.count) * 0.02))
        dismissAtTick = tick + Int(seconds / 0.09)
    }

    private func tickAnimation() {
        tick += 1
        if let _ = toolBadge, tick > toolBadgeUntil { toolBadge = nil }
        if dismissAtTick != 0 && tick >= dismissAtTick && !settingsOpen {
            dismissAtTick = 0
            bubbleText = ""
            relayout()            // облако исчезает, остаётся только персонаж
            return
        }
        needsDisplay = true
    }

    private func currentFrame() -> [String] {
        switch state {
        case .idle:
            let phase = tick % 32
            return (phase == 0 || phase == 1) ? Sprite.blink : Sprite.base
        case .thinking:
            return (tick / 3) % 2 == 0 ? Sprite.think : Sprite.base
        case .talking:
            return (tick / 2) % 2 == 0 ? Sprite.talkOpen : Sprite.talkClosed
        case .working:
            return (tick / 2) % 2 == 0 ? Sprite.workA : Sprite.workB
        }
    }

    // MARK: отрисовка

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        let w = bounds.width

        // персонаж — снизу по центру
        let charArea = CGRect(x: (w - charBox) / 2,
                              y: bounds.height - bottomMargin - charBox,
                              width: charBox, height: charBox)

        // облако рисуем только когда есть что показать
        if let text = displayText() {
            let bs = bubbleSize(for: text)
            let bubbleRect = CGRect(x: (w - bs.width) / 2, y: topMargin,
                                    width: bs.width, height: bs.height)
            drawBubble(in: bubbleRect, text: text, charTop: charArea.minY, ctx: ctx)
        }

        // лёгкая тень-«грунт» под персонажем
        let bob = (state == .working)
            ? CGFloat(abs((tick % 4) - 2))            // быстрый поскок при работе
            : CGFloat(sin(Double(tick) * 0.22) * 2.0) // спокойное дыхание
        let shScale = charBox / 160
        let shadow = CGRect(x: charArea.midX - 46 * shScale, y: charArea.maxY - 12,
                            width: 92 * shScale, height: 12)
        NSColor(calibratedWhite: 0, alpha: 0.18).setFill()
        NSBezierPath(ovalIn: shadow).fill()

        drawSprite(currentFrame(), in: charArea, ctx: ctx, bob: bob)

        if let badge = toolBadge {
            drawBadge(badge, near: charArea)
        }
    }

    private func drawSprite(_ frame: [String], in rect: CGRect, ctx: CGContext, bob: CGFloat) {
        let scale = floor(min(rect.width, rect.height) / 16.0)
        let dim = scale * 16
        let ox = rect.minX + (rect.width - dim) / 2
        let oy = rect.minY + (rect.height - dim) / 2 + bob
        ctx.interpolationQuality = .none
        for (r, row) in frame.enumerated() {
            for (c, ch) in row.enumerated() {
                guard let col = Sprite.color(ch) else { continue }
                col.setFill()
                ctx.fill(CGRect(x: ox + CGFloat(c) * scale,
                                y: oy + CGFloat(r) * scale,
                                width: scale, height: scale))
            }
        }
    }

    private func drawBubble(in area: CGRect, text: String, charTop: CGFloat, ctx: CGContext) {
        let body = NSBezierPath(roundedRect: area, xRadius: 13, yRadius: 13)
        NSColor(calibratedWhite: 0.12, alpha: 0.93).setFill()
        body.fill()
        NSColor(srgbRed: 0.50, green: 0.80, blue: 1.0, alpha: 0.5).setStroke()
        body.lineWidth = 1.5
        body.stroke()

        // хвостик вниз, к персонажу
        let tail = NSBezierPath()
        tail.move(to: CGPoint(x: area.midX - 9, y: area.maxY - 1))
        tail.line(to: CGPoint(x: area.midX + 9, y: area.maxY - 1))
        tail.line(to: CGPoint(x: area.midX, y: min(area.maxY + 11, charTop - 2)))
        tail.close()
        NSColor(calibratedWhite: 0.12, alpha: 0.93).setFill()
        tail.fill()

        // текст (облако ровно под объём — обрезки нет)
        let inset = area.insetBy(dx: bubblePad, dy: bubblePad)
        NSAttributedString(string: text, attributes: textAttrs)
            .draw(with: inset, options: [.usesLineFragmentOrigin])
    }

    private func drawBadge(_ text: String, near rect: CGRect) {
        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.boldSystemFont(ofSize: 11),
            .foregroundColor: NSColor(srgbRed: 0.09, green: 0.13, blue: 0.18, alpha: 1),
        ]
        let str = NSAttributedString(string: text, attributes: attrs)
        let size = str.size()
        let pill = CGRect(x: rect.midX - size.width / 2 - 8,
                          y: rect.maxY - 2,
                          width: size.width + 16, height: size.height + 6)
        NSColor(srgbRed: 1.0, green: 0.95, blue: 0.65, alpha: 0.95).setFill()
        NSBezierPath(roundedRect: pill, xRadius: 8, yRadius: 8).fill()
        str.draw(at: CGPoint(x: pill.minX + 8, y: pill.minY + 3))
    }
}

// MARK: - App delegate

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: PetWindow!
    let api = ApiClient()
    var root: RootView!

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Стартовый размер — только под персонажа; дальше окно растёт под ответ.
        let size = NSSize(width: 140, height: 134)
        let screen = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
        let origin = NSPoint(x: screen.maxX - size.width - 24, y: screen.minY + 24)

        window = PetWindow(
            contentRect: NSRect(origin: origin, size: size),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false)
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = false
        window.level = .floating
        window.isMovableByWindowBackground = true
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]

        root = RootView(frame: NSRect(origin: .zero, size: size), api: api)
        window.contentView = root

        api.onEvent = { [weak self] e in self?.root.handle(e) }
        api.start()

        // показываем поверх, но НЕ забираем фокус у терминала (ты печатаешь в CLI)
        window.orderFrontRegardless()
        root.initialLayout()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

// MARK: - Bootstrap

let app = NSApplication.shared
// .accessory: без иконки в доке и без перехвата фокуса у терминала/CLI.
app.setActivationPolicy(.accessory)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
