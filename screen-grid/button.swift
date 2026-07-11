import AppKit
import Foundation

// Обычное macOS-окно (title bar, таскается за само окно) с одной стандартной
// кнопкой "Scan". Клик -> GET /scan на detect.py.

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var scanURL: URL!

    func applicationDidFinishLaunching(_ notification: Notification) {
        let scanURLStr = ProcessInfo.processInfo.environment["SCAN_URL"]
            ?? "http://127.0.0.1:8132/scan"
        scanURL = URL(string: scanURLStr)!

        let size = NSSize(width: 200, height: 120)
        window = NSWindow(contentRect: NSRect(origin: .zero, size: size),
                           styleMask: [.titled, .closable, .miniaturizable],
                           backing: .buffered, defer: false)
        window.title = "screen-grid"
        window.level = .floating
        window.isMovableByWindowBackground = true      // таскается за окно, не только за title bar
        window.sharingType = .none                      // не попадать в screencapture
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        window.center()

        let button = NSButton(frame: NSRect(x: 30, y: 35, width: 140, height: 40))
        button.title = "Scan"
        button.bezelStyle = .rounded
        button.font = NSFont.systemFont(ofSize: 16)
        button.target = self
        button.action = #selector(tapped)
        window.contentView?.addSubview(button)

        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func tapped() {
        URLSession.shared.dataTask(with: scanURL).resume()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
