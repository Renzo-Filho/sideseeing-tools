document.addEventListener('DOMContentLoaded', () => {
    window.activeMaps = {};

    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('mobile-overlay');

    function toggleSidebar() {
        sidebar.classList.toggle('-translate-x-full');
        overlay.classList.toggle('hidden');
    }

    document.getElementById('sidebar-open-btn').addEventListener('click', toggleSidebar);
    document.getElementById('sidebar-close-btn').addEventListener('click', toggleSidebar);
    overlay.addEventListener('click', toggleSidebar);

    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', event => {
            event.preventDefault();
            showSection(link.dataset.section);
        });
    });

    function showSection(sectionId) {
        const section = document.getElementById(`${sectionId}-section`);
        if (!section) return;

        document.querySelectorAll('.section-content').forEach(element => element.classList.remove('active'));
        section.classList.add('active');

        document.querySelectorAll('.nav-link').forEach(link => {
            const active = link.dataset.section === sectionId;
            link.classList.toggle('bg-blue-600', active);
            link.classList.toggle('text-white', active);
            link.classList.toggle('text-slate-400', !active);
        });

        // Leaflet needs a resize once its container becomes visible
        setTimeout(() => {
            const map = window.activeMaps[section.dataset.mapId];
            if (map) {
                map.invalidateSize();
                if (map.recenter) map.recenter();
            }
        }, 150);
    }

    initSidewalkAssessmentMap(ANALYSIS_FILES);
    showSection('sidewalk-assessment');
    lucide.createIcons();
});
