import Foundation
import CoreMotion
import Combine

/// Captures barometric pressure and relative altitude via CMAltimeter.
final class BarometerManager: ObservableObject {
    @Published var isRunning = false
    @Published var latestPressure: Double = 0
    @Published var latestRelativeAltitude: Double = 0

    private let altimeter = CMAltimeter()
    private var dataStore: SensorDataStore?

    var isAvailable: Bool {
        CMAltimeter.isRelativeAltitudeAvailable()
    }

    func start(dataStore: SensorDataStore) {
        guard isAvailable else { return }
        self.dataStore = dataStore

        altimeter.startRelativeAltitudeUpdates(to: .main) { [weak self] altitudeData, error in
            guard let self, let altitudeData else { return }

            let ts = TimestampProvider.shared.now
            let pressure = altitudeData.pressure.doubleValue  // kPa
            let relAlt = altitudeData.relativeAltitude.doubleValue  // meters

            self.latestPressure = pressure
            self.latestRelativeAltitude = relAlt

            let data = BarometerData(
                timestamp: ts,
                pressure: pressure,
                relativeAltitude: relAlt
            )

            Task { @MainActor in
                self.dataStore?.barometerSamples.append(data)
            }
        }

        isRunning = true
    }

    func stop() {
        altimeter.stopRelativeAltitudeUpdates()
        isRunning = false
    }
}
