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

final class RootView: NSView {
    let api: ApiClient

    private var state: MascotState = .idle
    private var tick: Int = 0
    private var bubbleText: String = "Жду чат… Спроси что-нибудь в CLI."
    private var toolBadge: String?
    private var toolBadgeUntil: Int = 0

    private let closeButton = NSButton()

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

    // MARK: события сервера (только чтение — облако зеркалит ответ)

    func handle(_ e: [String: Any]) {
        switch e["event"] as? String {
        case "hello":
            bubbleText = "На связи. Пиши в CLI — я покажу ответ здесь."
            state = .idle
        case "prompt":
            bubbleText = ""       // новый ход — чистим пузырь
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
        case "tool":
            if (e["phase"] as? String) == "start", let name = e["name"] as? String {
                let short = name.components(separatedBy: "__").last ?? name
                toolBadge = "⚙ " + short
                toolBadgeUntil = tick + 30
            }
        case "done":
            state = .idle
        case "error":
            bubbleText = "⚠️ " + (e["message"] as? String ?? "ошибка")
            state = .idle
        case "disconnected":
            // тихо ждём переподключения, ничего не меняем
            break
        default:
            break
        }
        needsDisplay = true
    }

    // MARK: анимация

    private func tickAnimation() {
        tick += 1
        if let _ = toolBadge, tick > toolBadgeUntil { toolBadge = nil }
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

        // области
        let bubbleArea = CGRect(x: 16, y: 30, width: w - 32, height: 150)
        let charArea = CGRect(x: 0, y: 188, width: w, height: 160)

        drawBubble(in: bubbleArea, ctx: ctx)

        // лёгкая тень-«грунт» под персонажем
        let bob = (state == .working)
            ? CGFloat(abs((tick % 4) - 2))           // быстрый поскок при работе
            : CGFloat(sin(Double(tick) * 0.22) * 2.0) // спокойное дыхание
        let shadow = CGRect(x: charArea.midX - 46, y: charArea.maxY - 14, width: 92, height: 14)
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

    private func drawBubble(in area: CGRect, ctx: CGContext) {
        let body = NSBezierPath(roundedRect: area, xRadius: 14, yRadius: 14)
        NSColor(calibratedWhite: 0.12, alpha: 0.93).setFill()
        body.fill()
        NSColor(srgbRed: 0.50, green: 0.80, blue: 1.0, alpha: 0.5).setStroke()
        body.lineWidth = 1.5
        body.stroke()

        // хвостик вниз, к персонажу
        let tail = NSBezierPath()
        tail.move(to: CGPoint(x: area.midX - 10, y: area.maxY - 1))
        tail.line(to: CGPoint(x: area.midX + 10, y: area.maxY - 1))
        tail.line(to: CGPoint(x: area.midX, y: area.maxY + 12))
        tail.close()
        NSColor(calibratedWhite: 0.12, alpha: 0.93).setFill()
        tail.fill()

        // текст
        let inset = area.insetBy(dx: 12, dy: 10)
        let para = NSMutableParagraphStyle()
        para.lineBreakMode = .byWordWrapping
        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 13),
            .foregroundColor: NSColor.white,
            .paragraphStyle: para,
        ]
        var display = bubbleText
        if display.isEmpty && state == .thinking {
            let dots = String(repeating: "•", count: (tick / 4) % 3 + 1)
            display = dots
        }
        let str = NSAttributedString(string: display, attributes: attrs)

        ctx.saveGState()
        NSBezierPath(rect: inset).addClip()
        let bound = str.boundingRect(
            with: CGSize(width: inset.width, height: .greatestFiniteMagnitude),
            options: [.usesLineFragmentOrigin])
        var drawRect = inset
        if bound.height > inset.height {
            drawRect.origin.y -= (bound.height - inset.height) // показываем самые свежие строки
        }
        str.draw(with: drawRect, options: [.usesLineFragmentOrigin])
        ctx.restoreGState()
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
        let size = NSSize(width: 280, height: 360)
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
