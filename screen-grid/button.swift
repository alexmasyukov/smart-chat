import AppKit
import Foundation

// Обычное macOS-окно (title bar, таскается за само окно) с кнопкой "Scan" и
// двумя ползунками: количество уточнений границы (1..10) и шаг сетки
// сканирования (10..150, шаг 10). Клик -> GET /scan на detect.py с
// ?bisect=<...>&step=<...>.

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var scanBaseURL: URL!
    var bisectSlider: NSSlider!
    var bisectLabel: NSTextField!
    var stepSlider: NSSlider!
    var stepLabel: NSTextField!

    func applicationDidFinishLaunching(_ notification: Notification) {
        let scanURLStr = ProcessInfo.processInfo.environment["SCAN_URL"]
            ?? "http://127.0.0.1:8132/scan"
        scanBaseURL = URL(string: scanURLStr)!

        let size = NSSize(width: 320, height: 300)
        window = NSWindow(contentRect: NSRect(origin: .zero, size: size),
                           styleMask: [.titled, .closable, .miniaturizable],
                           backing: .buffered, defer: false)
        window.title = "screen-grid"
        window.level = .floating
        window.isMovableByWindowBackground = true      // таскается за окно, не только за title bar
        window.sharingType = .none                      // не попадать в screencapture
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        window.center()

        bisectLabel = NSTextField(labelWithString: "Уточнений границы: 3")
        bisectLabel.frame = NSRect(x: 16, y: 260, width: 290, height: 20)
        bisectLabel.alignment = .center
        window.contentView?.addSubview(bisectLabel)

        bisectSlider = NSSlider(frame: NSRect(x: 16, y: 230, width: 290, height: 24))
        bisectSlider.minValue = 1
        bisectSlider.maxValue = 10
        bisectSlider.integerValue = 3
        bisectSlider.numberOfTickMarks = 10
        bisectSlider.allowsTickMarkValuesOnly = true
        bisectSlider.target = self
        bisectSlider.action = #selector(bisectMoved)
        window.contentView?.addSubview(bisectSlider)

        stepLabel = NSTextField(labelWithString: "Шаг сетки: 150px")
        stepLabel.frame = NSRect(x: 16, y: 190, width: 290, height: 20)
        stepLabel.alignment = .center
        window.contentView?.addSubview(stepLabel)

        stepSlider = NSSlider(frame: NSRect(x: 16, y: 160, width: 290, height: 24))
        stepSlider.minValue = 10
        stepSlider.maxValue = 300
        stepSlider.integerValue = 150                    // дефолт как раньше
        stepSlider.numberOfTickMarks = 30                // 10,20,...,300
        stepSlider.allowsTickMarkValuesOnly = true
        stepSlider.target = self
        stepSlider.action = #selector(stepMoved)
        window.contentView?.addSubview(stepSlider)

        let button = NSButton(frame: NSRect(x: 20, y: 30, width: 280, height: 80))  // вдвое крупнее
        button.title = "Scan"
        button.bezelStyle = .rounded
        button.font = NSFont.systemFont(ofSize: 28)
        button.target = self
        button.action = #selector(tapped)
        window.contentView?.addSubview(button)

        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func bisectMoved() {
        bisectLabel.stringValue = "Уточнений границы: \(bisectSlider.integerValue)"
    }

    @objc private func stepMoved() {
        // шаг сетки 10px, поэтому округляем к ближайшему кратному 10
        let rounded = (stepSlider.integerValue / 10) * 10
        stepSlider.integerValue = rounded
        stepLabel.stringValue = "Шаг сетки: \(rounded)px"
    }

    @objc private func tapped() {
        var comps = URLComponents(url: scanBaseURL, resolvingAgainstBaseURL: false)!
        comps.queryItems = [
            URLQueryItem(name: "bisect", value: "\(bisectSlider.integerValue)"),
            URLQueryItem(name: "step", value: "\(stepSlider.integerValue)"),
        ]
        guard let url = comps.url else { return }
        URLSession.shared.dataTask(with: url).resume()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
