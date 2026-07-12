import AppKit
import Foundation

// Обычное macOS-окно (title bar, таскается за само окно) с кнопкой "Scan" и
// двумя ползунками: количество уточнений границы (1..10) и шаг сетки
// сканирования (10..150, шаг 10). Клик -> GET /scan на detect.py с
// ?bisect=<...>&step=<...>.

/// Плоская кнопка со своим layer'ом (реально растягивается на весь фрейм,
/// в отличие от .rounded bezel): тёмно-серая, светлеет при наведении,
/// темнеет при нажатии, курсор — рука.
final class FlatButton: NSButton {
    var normalColor: NSColor = NSColor(white: 0.16, alpha: 1)
    var hoverColor: NSColor = NSColor(white: 0.24, alpha: 1)
    var pressedColor: NSColor = NSColor(white: 0.08, alpha: 1)
    private var isHovering = false

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        trackingAreas.forEach { removeTrackingArea($0) }
        addTrackingArea(NSTrackingArea(rect: bounds,
                                        options: [.mouseEnteredAndExited, .activeAlways, .cursorUpdate],
                                        owner: self))
    }

    override func resetCursorRects() {
        addCursorRect(bounds, cursor: .pointingHand)
    }

    override func mouseEntered(with event: NSEvent) {
        isHovering = true
        layer?.backgroundColor = hoverColor.cgColor
    }

    override func mouseExited(with event: NSEvent) {
        isHovering = false
        layer?.backgroundColor = normalColor.cgColor
    }

    override func mouseDown(with event: NSEvent) {
        layer?.backgroundColor = pressedColor.cgColor
        super.mouseDown(with: event)          // блокирует до mouseUp, тогда и жмёт action
        layer?.backgroundColor = (isHovering ? hoverColor : normalColor).cgColor
    }
}

extension NSColor {
    convenience init?(hex: String) {
        var s = hex
        if s.hasPrefix("#") { s.removeFirst() }
        guard s.count == 6, let v = UInt32(s, radix: 16) else { return nil }
        let r = CGFloat((v >> 16) & 0xFF) / 255
        let g = CGFloat((v >> 8) & 0xFF) / 255
        let b = CGFloat(v & 0xFF) / 255
        self.init(srgbRed: r, green: g, blue: b, alpha: 1)
    }
}

/// Источник данных таблицы лога: колонки "№" (фикс. ширина), "hex", "цвет"
/// (кружок-превью). NSTableView сам держит колонки ровными — без "лесенки".
final class LogDataSource: NSObject, NSTableViewDataSource, NSTableViewDelegate {
    var numbers: [Int] = []
    var colorsHex: [String] = []

    func numberOfRows(in tableView: NSTableView) -> Int { numbers.count }

    func tableView(_ tableView: NSTableView, viewFor tableColumn: NSTableColumn?, row: Int) -> NSView? {
        guard let colId = tableColumn?.identifier.rawValue else { return nil }
        let hex = colorsHex[row]
        let font = NSFont.monospacedSystemFont(ofSize: 14, weight: .regular)

        if colId == "swatch" {
            let cell = NSTableCellView()
            let swatch = NSView(frame: NSRect(x: 6, y: 3, width: 16, height: 16))
            swatch.wantsLayer = true
            swatch.layer?.backgroundColor = (NSColor(hex: hex) ?? .gray).cgColor
            swatch.layer?.cornerRadius = 8
            swatch.layer?.borderWidth = 1
            swatch.layer?.borderColor = NSColor.white.withAlphaComponent(0.3).cgColor
            cell.addSubview(swatch)
            return cell
        }

        let text = colId == "num" ? "\(numbers[row])" : hex
        let tf = NSTextField(labelWithString: text)
        tf.font = font
        tf.alignment = colId == "num" ? .right : .left
        tf.frame = NSRect(x: 4, y: 2, width: (tableColumn?.width ?? 60) - 8, height: 18)
        let cell = NSTableCellView()
        cell.addSubview(tf)
        cell.textField = tf
        return cell
    }

    func tableView(_ tableView: NSTableView, heightOfRow row: Int) -> CGFloat { 22 }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var scanBaseURL: URL!
    var predomSlider: NSSlider!
    var predomLabel: NSTextField!
    var stepSlider: NSSlider!
    var stepLabel: NSTextField!
    var pointsSlider: NSSlider!
    var pointsLabel: NSTextField!
    var fullCheck: NSButton!
    var midCheck: NSButton!
    var logTable: NSTableView!
    let logDataSource = LogDataSource()

    func applicationDidFinishLaunching(_ notification: Notification) {
        let scanURLStr = ProcessInfo.processInfo.environment["SCAN_URL"]
            ?? "http://127.0.0.1:8132/scan"
        scanBaseURL = URL(string: scanURLStr)!

        let size = NSSize(width: 660, height: 340)
        window = NSWindow(contentRect: NSRect(origin: .zero, size: size),
                           styleMask: [.titled, .closable, .miniaturizable],
                           backing: .buffered, defer: false)
        window.title = "screen-seed"
        window.level = .floating
        window.isMovableByWindowBackground = true      // таскается за окно, не только за title bar
        window.sharingType = .none                      // не попадать в screencapture
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        window.center()

        predomLabel = NSTextField(labelWithString: "Точек на преобладание: 3")
        predomLabel.frame = NSRect(x: 16, y: 300, width: 290, height: 20)
        predomLabel.alignment = .center
        window.contentView?.addSubview(predomLabel)

        predomSlider = NSSlider(frame: NSRect(x: 16, y: 276, width: 290, height: 24))
        predomSlider.minValue = 2
        predomSlider.maxValue = 8
        predomSlider.integerValue = 3
        predomSlider.numberOfTickMarks = 7          // 2,3,...,8
        predomSlider.allowsTickMarkValuesOnly = true
        predomSlider.target = self
        predomSlider.action = #selector(predomMoved)
        window.contentView?.addSubview(predomSlider)

        stepLabel = NSTextField(labelWithString: "Шаг сетки: 60px")
        stepLabel.frame = NSRect(x: 16, y: 252, width: 290, height: 20)
        stepLabel.alignment = .center
        window.contentView?.addSubview(stepLabel)

        stepSlider = NSSlider(frame: NSRect(x: 16, y: 228, width: 290, height: 24))
        stepSlider.minValue = 10
        stepSlider.maxValue = 300
        stepSlider.integerValue = 60
        stepSlider.numberOfTickMarks = 30                // 10,20,...,300
        stepSlider.allowsTickMarkValuesOnly = true
        stepSlider.target = self
        stepSlider.action = #selector(stepMoved)
        window.contentView?.addSubview(stepSlider)

        pointsLabel = NSTextField(labelWithString: "Точек вниз: 18")
        pointsLabel.frame = NSRect(x: 16, y: 204, width: 290, height: 20)
        pointsLabel.alignment = .center
        window.contentView?.addSubview(pointsLabel)

        pointsSlider = NSSlider(frame: NSRect(x: 16, y: 180, width: 290, height: 24))
        pointsSlider.minValue = 2
        pointsSlider.maxValue = 20
        pointsSlider.integerValue = 18
        pointsSlider.numberOfTickMarks = 19              // 2,3,...,20
        pointsSlider.allowsTickMarkValuesOnly = true
        pointsSlider.target = self
        pointsSlider.action = #selector(pointsMoved)
        window.contentView?.addSubview(pointsSlider)

        fullCheck = NSButton(checkboxWithTitle: "До конца экрана", target: self, action: #selector(fullToggled))
        fullCheck.frame = NSRect(x: 16, y: 154, width: 290, height: 24)
        fullCheck.state = .on                            // по умолчанию включено
        window.contentView?.addSubview(fullCheck)

        midCheck = NSButton(checkboxWithTitle: "Строить пятую точку", target: nil, action: nil)
        midCheck.frame = NSRect(x: 16, y: 126, width: 290, height: 24)
        midCheck.state = .off                            // по умолчанию выключено
        window.contentView?.addSubview(midCheck)

        fullToggled()                                    // применить начальное состояние (points slider off)

        // .rounded bezel игнорирует высоту фрейма (фикс. Aqua-высота) — поэтому
        // делаем плоскую кнопку своим layer'ом, она реально растягивается.
        let button = FlatButton(frame: NSRect(x: 20, y: 30, width: 280, height: 80))
        button.title = "Scan"
        button.isBordered = false
        button.wantsLayer = true
        button.layer?.backgroundColor = button.normalColor.cgColor
        button.layer?.cornerRadius = 14
        button.font = NSFont.systemFont(ofSize: 28, weight: .semibold)
        button.contentTintColor = .white
        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 28, weight: .semibold),
            .foregroundColor: NSColor.white,
        ]
        button.attributedTitle = NSAttributedString(string: "Scan", attributes: attrs)
        button.target = self
        button.action = #selector(tapped)
        window.contentView?.addSubview(button)

        // Лог справа: таблица (тот же порядок, что в out/points_*.txt) —
        // колонки "№" (фикс. ширина, без "лесенки"), hex, кружок-превью.
        // Обновляется после каждого скана.
        let logScroll = NSScrollView(frame: NSRect(x: 328, y: 16, width: 316, height: 308))
        logScroll.hasVerticalScroller = true
        logScroll.autohidesScrollers = false
        logScroll.borderType = .bezelBorder
        logScroll.drawsBackground = true

        logTable = NSTableView(frame: NSRect(origin: .zero, size: logScroll.contentSize))
        logTable.dataSource = logDataSource
        logTable.delegate = logDataSource
        logTable.usesAlternatingRowBackgroundColors = true
        logTable.headerView = nil                    // без заголовка — панель узкая

        let colNum = NSTableColumn(identifier: NSUserInterfaceItemIdentifier("num"))
        colNum.width = 34
        colNum.minWidth = 34
        colNum.maxWidth = 34
        logTable.addTableColumn(colNum)

        let colHex = NSTableColumn(identifier: NSUserInterfaceItemIdentifier("hex"))
        colHex.width = 180
        logTable.addTableColumn(colHex)

        let colSwatch = NSTableColumn(identifier: NSUserInterfaceItemIdentifier("swatch"))
        colSwatch.width = 40
        colSwatch.minWidth = 40
        colSwatch.maxWidth = 40
        logTable.addTableColumn(colSwatch)

        logScroll.documentView = logTable
        window.contentView?.addSubview(logScroll)

        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func refreshLog(numbers: [Int], colorsHex: [String]) {
        logDataSource.numbers = numbers
        logDataSource.colorsHex = colorsHex
        logTable.reloadData()
    }

    @objc private func predomMoved() {
        predomLabel.stringValue = "Точек на преобладание: \(predomSlider.integerValue)"
    }

    @objc private func stepMoved() {
        // шаг сетки 10px, поэтому округляем к ближайшему кратному 10
        let rounded = (stepSlider.integerValue / 10) * 10
        stepSlider.integerValue = rounded
        stepLabel.stringValue = "Шаг сетки: \(rounded)px"
    }

    @objc private func pointsMoved() {
        pointsLabel.stringValue = "Точек вниз: \(pointsSlider.integerValue)"
    }

    @objc private func fullToggled() {
        // «До конца экрана» блокирует ручной ползунок «Точек вниз»
        let on = fullCheck.state == .on
        pointsSlider.isEnabled = !on
        pointsLabel.textColor = on ? .disabledControlTextColor : .labelColor
    }

    @objc private func tapped() {
        var comps = URLComponents(url: scanBaseURL, resolvingAgainstBaseURL: false)!
        comps.queryItems = [
            URLQueryItem(name: "predom", value: "\(predomSlider.integerValue)"),
            URLQueryItem(name: "step", value: "\(stepSlider.integerValue)"),
            URLQueryItem(name: "ndown", value: "\(pointsSlider.integerValue)"),
            URLQueryItem(name: "mid", value: midCheck.state == .on ? "1" : "0"),
            URLQueryItem(name: "full", value: fullCheck.state == .on ? "1" : "0"),
        ]
        guard let url = comps.url else { return }
        URLSession.shared.dataTask(with: url) { [weak self] data, _, _ in
            guard let self, let data,
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let numbers = obj["numbers"] as? [Int],
                  let colors = obj["colors"] as? [String] else { return }
            DispatchQueue.main.async { self.refreshLog(numbers: numbers, colorsHex: colors) }
        }.resume()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
