import Foundation
import CoreLocation
import Combine

/// Captures GPS data (dual-frequency L1+L5 when available) via CoreLocation.
final class LocationManager: NSObject, ObservableObject {
    @Published var isRunning = false
    @Published var latestLocation: CLLocation?
    @Published var authorizationStatus: CLAuthorizationStatus = .notDetermined

    private let manager = CLLocationManager()
    private var dataStore: SensorDataStore?

    override init() {
        super.init()
        manager.delegate = self
    }

    func requestPermission() {
        manager.requestWhenInUseAuthorization()
    }

    func start(dataStore: SensorDataStore) {
        self.dataStore = dataStore

        manager.desiredAccuracy = kCLLocationAccuracyBest
        manager.distanceFilter = kCLDistanceFilterNone
        manager.allowsBackgroundLocationUpdates = false
        manager.startUpdatingLocation()

        isRunning = true
    }

    func stop() {
        manager.stopUpdatingLocation()
        isRunning = false
    }
}

extension LocationManager: CLLocationManagerDelegate {
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else { return }

        let ts = TimestampProvider.shared.now
        latestLocation = location

        let data = GPSData(
            timestamp: ts,
            latitude: location.coordinate.latitude,
            longitude: location.coordinate.longitude,
            altitude: location.altitude,
            horizontalAccuracy: location.horizontalAccuracy,
            verticalAccuracy: location.verticalAccuracy,
            speed: location.speed,
            course: location.course,
            floor: location.floor?.level
        )

        Task { @MainActor in
            self.dataStore?.gpsSamples.append(data)
        }
    }

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        authorizationStatus = manager.authorizationStatus
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("[LocationManager] Error: \(error.localizedDescription)")
    }
}
