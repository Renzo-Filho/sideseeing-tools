function initSummaryMap() {
    const mapDiv = document.getElementById('summary-overview-map');
    if (!mapDiv) return;

    const geoFiles = typeof GEO_FILES !== 'undefined' ? GEO_FILES : {};

    if (Object.keys(geoFiles).length === 0) {
        mapDiv.innerHTML = '<div class="no-data-message">No route data to display.</div>';
        return;
    }

    let mapCenter = [-23.5505, -46.6333]; // Default center
    let mapZoom = 10;
    let initialBounds = null;

    const overviewMap = L.map(mapDiv).setView(mapCenter, mapZoom);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(overviewMap);

    const dataPromises = Object.entries(geoFiles).map(([name, path]) =>
        fetch(`${path}?v=${Date.now()}`).then(res => res.json()).then(data => ({ name, data })));

    Promise.all(dataPromises).then(results => {
        const combinedBounds = L.latLngBounds();

        // The smoothed track, or the phone's raw fixes for older reports without one.
        results.forEach(({ name, data }) => {
            const smoothed = data.smoothed_path && data.smoothed_path.length > 0;
            const path = smoothed ? data.smoothed_path : data.path;
            if (!path || path.length === 0) return;

            const polyline = L.polyline(path, { color: smoothed ? '#3388ff' : '#dc3545', weight: 5 });
            polyline.bindPopup(`<b>Sample:</b><br>${name}`);
            polyline.addTo(overviewMap);
            combinedBounds.extend(polyline.getBounds());
        });

        if (combinedBounds.isValid()) {
            overviewMap.fitBounds(combinedBounds, { padding: [50, 50] });
            initialBounds = combinedBounds;
        } else if (results.length > 0 && results[0].data.center) {
            overviewMap.setView(results[0].data.center, 15);
            initialBounds = overviewMap.getBounds();
        }

        addFullscreenControl(overviewMap, 'map-wrapper');
        addRecenterControl(overviewMap, initialBounds);
        if (window.lucide) window.lucide.createIcons();
    });

    mapDiv.closest('.section-content').dataset.mapId = 'summary-overview-map';
    window.activeMaps['summary-overview-map'] = overviewMap;
}
