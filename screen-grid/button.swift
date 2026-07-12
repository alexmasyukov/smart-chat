import AppKit
import Foundation

// Обычное macOS-окно (title bar, таскается за само окно) с кнопкой "Scan"
// и ползунком количества уточнений границы (1..10). Клик -> GET /scan на
// detect.py с ?bisect=<значение ползунка>.

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var scanBaseURL: URL!
    var slider: NSSlider!
    var sliderLabel: NSTextField!

    func applicationDidFinishLaunching(_ notification: Notification) {
        let scanURLStr = ProcessInfo.processInfo.environment["SCAN_URL"]
            ?? "http://127.0.0.1:8132/scan"
        scanBaseURL = URL(string: scanURLStr)!

        let size = NSSize(width: 220, height: 180)
        window = NSWindow(contentRect: NSRect(origin: .zero, size: size),
                           styleMask: [.titled, .closable, .miniaturizable],
                           backing: .buffered, defer: false)
        window.title = "screen-grid"
        window.level = .floating
        window.isMovableByWindowBackground = true      // таскается за окно, не только за title bar
        window.sharingType = .none                      // не попадать в screencapture
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        window.center()

        sliderLabel = NSTextField(labelWithString: "Уточнений границы: 3")
        sliderLabel.frame = NSRect(x: 16, y: 130, width: 190, height: 20)
        sliderLabel.alignment = .center
        window.contentView?.addSubview(sliderLabel)

        slider = NSSlider(frame: NSRect(x: 16, y: 100, width: 190, height: 24))
        slider.minValue = 1
        slider.maxValue = 10
        slider.integerValue = 3
        slider.numberOfTickMarks = 10
        slider.allowsTickMarkValuesOnly = true
        slider.target = self
        slider.action = #selector(sliderMoved)
        window.contentView?.addSubview(slider)

        let button = NSButton(frame: NSRect(x: 40, y: 35, width: 140, height: 40))
        button.title = "Scan"
        button.bezelStyle = .rounded
        button.font = NSFont.systemFont(ofSize: 16)
        button.target = self
        button.action = #selector(tapped)
        window.contentView?.addSubview(button)

        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func sliderMoved() {
        sliderLabel.stringValue = "Уточнений границы: \(slider.integerValue)"
    }

    @objc private func tapped() {
        var comps = URLComponents(url: scanBaseURL, resolvingAgainstBaseURL: false)!
        comps.queryItems = [URLQueryItem(name: "bisect", value: "\(slider.integerValue)")]
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
