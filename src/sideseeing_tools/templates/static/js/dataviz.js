let currentDatavizState = {
    instance: null,
    frames: [],
    currentIndex: 0,
    bboxes: {},
    groups: [],           // [{name, fill, classes}] from the server (prompt_groups.json, else one group)
    class_colors: {},
    activeClasses: new Set(),
    collapsedGroups: new Set(),
    minScore: 0.5,
    highlight: null       // index of the detection hovered in the per-frame list
};

document.addEventListener("DOMContentLoaded", () => {
    const select = document.getElementById("dataviz-instance-select");
    if(select && typeof SAMPLES_DATA !== 'undefined') {
        SAMPLES_DATA.forEach(sample => {
            const opt = document.createElement("option");
            opt.value = sample.name;
            opt.textContent = sample.name;
            select.appendChild(opt);
        });
        select.addEventListener("change", (e) => loadInstanceData(e.target.value));
    }
});

async function loadInstanceData(instanceName) {
    if(!instanceName) return;
    resetDatavizUI();

    try {
        const res = await fetch(`/api/dataviz/data/${encodeURIComponent(instanceName)}`);
        const data = await res.json();

        Object.assign(currentDatavizState, {
            instance: instanceName,
            frames: data.frames,
            bboxes: data.bboxes,
            groups: data.groups || [],
            class_colors: data.class_colors || {},
            currentIndex: 0,
            highlight: null,
            collapsedGroups: new Set(),
        });
        // Start with the first group only: with dozens of prompts, everything at once is unreadable.
        const first = currentDatavizState.groups[0];
        currentDatavizState.activeClasses = new Set(first ? first.classes : []);

        const badge = document.getElementById('dataviz-redaction');
        badge.classList.remove('hidden', 'text-emerald-300', 'text-amber-300');
        badge.classList.add(data.redacted ? 'text-emerald-300' : 'text-amber-300');
        badge.textContent = data.redacted ? 'Redacted frames' : 'Original frames (not redacted)';

        const overlay = document.getElementById('extraction-overlay');
        const btnExtract = document.getElementById('btn-trigger-extract');

        if (data.frames.length === 0) {
            overlay.classList.remove('hidden');
            btnExtract.onclick = () => triggerExtraction(instanceName);
            return;
        }

        overlay.classList.add('hidden');
        renderPresets();
        renderFrame();

    } catch (e) {
        console.error("Failed to load instance AI data", e);
    }
}

async function triggerExtraction(instanceName) {
    const btn = document.getElementById('btn-trigger-extract');
    const progress = document.getElementById('extraction-progress');

    btn.classList.add('hidden');
    progress.classList.remove('hidden');
    progress.classList.add('flex');

    try {
        const res = await fetch(`/api/dataviz/extract/${encodeURIComponent(instanceName)}`, {
            method: 'POST'
        });
        const data = await res.json();
        console.log("Extraction Status:", data.message);

        // Check back every 5 seconds until frames appear
        const pollInterval = setInterval(async () => {
            const checkRes = await fetch(`/api/dataviz/data/${encodeURIComponent(instanceName)}`);
            const checkData = await checkRes.json();

            if (checkData.frames.length > 0) {
                clearInterval(pollInterval);
                loadInstanceData(instanceName);
            }
        }, 5000);

    } catch (e) {
        alert("Failed to start extraction. Check server logs.");
        btn.classList.remove('hidden');
        progress.classList.add('hidden');
    }
}

function renderFrame() {
    if(currentDatavizState.frames.length === 0) return;

    const state = currentDatavizState;
    const filename = state.frames[state.currentIndex];
    document.getElementById("dataviz-filename").innerText = `${filename}`;
    state.highlight = null;

    const baseImage = document.getElementById("base-image");
    baseImage.src = `/api/dataviz/image/${encodeURIComponent(state.instance)}/${encodeURIComponent(filename)}`;
    baseImage.onload = () => {
        const svg = document.getElementById("svg-layer");
        svg.setAttribute("viewBox", `0 0 ${baseImage.naturalWidth} ${baseImage.naturalHeight}`);
        document.getElementById("dataviz-canvas").style.aspectRatio = `${baseImage.naturalWidth} / ${baseImage.naturalHeight}`;
    };

    const params = new URLSearchParams({
        classes: Array.from(state.activeClasses).join(","),
        instance: state.instance,
        min_score: state.minScore,
    });
    document.getElementById("mask-layer").src = `/api/dataviz/mask/${encodeURIComponent(filename)}?${params}`;

    document.getElementById("frame-input").value = state.currentIndex + 1;
    document.getElementById("total-frames-count").innerText = state.frames.length;

    renderBBoxes(filename);
    renderClassList();
    renderGallery();
}

// Detections of the current frame that pass the score filter (and the class filter), best first.
function frameDetections(filename, onlyActive = true) {
    const state = currentDatavizState;
    return (state.bboxes[filename] || [])
        .filter(box => (box.confidence == null || box.confidence >= state.minScore)
                    && (!onlyActive || state.activeClasses.has(box.class_name)))
        .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));
}

function svgElement(tag, attributes) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    return element;
}

function drawBox(svg, box, emphasized) {
    const color = box.color || "#00ff00";
    const label = box.confidence == null ? box.class_name : `${box.class_name} ${box.confidence.toFixed(2)}`;
    const group = svgElement("g", {});
    group.appendChild(svgElement("rect", {
        x: box.xmin, y: box.ymin, width: box.xmax - box.xmin, height: box.ymax - box.ymin,
        fill: "none", stroke: color, "stroke-width": emphasized ? 6 : 2,
    }));
    group.appendChild(svgElement("rect", {
        x: box.xmin, y: box.ymin - 18, width: label.length * 7 + 8, height: 18, fill: color, rx: 2,
    }));
    const text = svgElement("text", {
        x: box.xmin + 4, y: box.ymin - 5, fill: "white", "font-size": "12px",
        "font-family": "sans-serif", "font-weight": "bold",
    });
    text.textContent = label;
    group.appendChild(text);
    svg.appendChild(group);
}

// Boxes are drawn when the Boxes toggle is on; the detection hovered in the list is always drawn.
function renderBBoxes(filename, listToo = true) {
    const svg = document.getElementById("svg-layer");
    svg.innerHTML = "";
    if(!filename) return;

    const detections = frameDetections(filename);
    const showAll = document.getElementById("cb-all-bboxes").checked;
    detections.forEach((box, index) => {
        const emphasized = index === currentDatavizState.highlight;
        if(showAll || emphasized) drawBox(svg, box, emphasized);
    });
    if(listToo) renderFrameList(detections);
}

// --- Groups, presets and classes ---
function renderPresets() {
    const container = document.getElementById("dataviz-presets");
    container.innerHTML = "";
    const chip = (label, classes) => {
        const button = document.createElement("button");
        button.className = "text-[10px] px-2 py-0.5 rounded-full border border-gray-300 text-gray-600 hover:border-indigo-400 hover:text-indigo-600";
        button.textContent = label;
        button.onclick = () => { currentDatavizState.activeClasses = new Set(classes); renderFrame(); };
        container.appendChild(button);
    };
    const groups = currentDatavizState.groups;
    if(groups.length > 1) groups.forEach(group => chip(group.name, group.classes));
    chip("All", groups.flatMap(group => group.classes));
    chip("None", []);
}

function classCounts() {
    const counts = {};
    const filename = currentDatavizState.frames[currentDatavizState.currentIndex];
    frameDetections(filename, false).forEach(box => { counts[box.class_name] = (counts[box.class_name] || 0) + 1; });
    return counts;
}

function renderClassList() {
    const state = currentDatavizState;
    const container = document.getElementById("dataviz-class-list");
    container.innerHTML = "";

    if(state.groups.length === 0) {
        container.innerHTML = '<div class="text-xs text-gray-400 italic">No detections for this route.</div>';
        return;
    }

    const query = document.getElementById("class-search").value.trim().toLowerCase();
    const counts = classCounts();

    state.groups.forEach(group => {
        const classes = group.classes.filter(name => !query || name.toLowerCase().includes(query));
        if(classes.length === 0) return;

        const active = group.classes.filter(name => state.activeClasses.has(name)).length;
        const collapsed = state.collapsedGroups.has(group.name) && !query;
        const total = group.classes.reduce((sum, name) => sum + (counts[name] || 0), 0);

        const header = document.createElement("div");
        header.className = "flex items-center gap-2 px-1 py-1 mt-1 text-[11px] font-semibold text-gray-700";
        const box = document.createElement("input");
        box.type = "checkbox";
        box.className = "accent-indigo-600 w-3 h-3";
        box.checked = active === group.classes.length;
        box.indeterminate = active > 0 && active < group.classes.length;
        box.title = "Show or hide the whole group";
        box.onchange = () => {
            group.classes.forEach(name => box.checked ? state.activeClasses.add(name) : state.activeClasses.delete(name));
            renderFrame();
        };
        const swatch = document.createElement("span");
        swatch.className = "w-3 h-3 rounded-sm shrink-0";
        const shades = group.classes.map(name => state.class_colors[name]);
        swatch.style.background = `linear-gradient(90deg, ${(shades.length > 1 ? shades : [shades[0], shades[0]]).join(", ")})`;
        const title = document.createElement("button");
        title.className = "flex-1 text-left truncate hover:text-indigo-600";
        title.textContent = `${collapsed ? "▸" : "▾"} ${group.name}${group.fill ? " · surface" : ""}`;
        title.onclick = () => {
            collapsed ? state.collapsedGroups.delete(group.name) : state.collapsedGroups.add(group.name);
            renderClassList();
        };
        const count = document.createElement("span");
        count.className = "text-[10px] font-mono text-gray-400";
        count.textContent = total || "";
        header.append(box, swatch, title, count);
        container.appendChild(header);

        if(collapsed) return;

        classes.forEach(name => {
            const row = document.createElement("label");
            row.className = "flex items-center gap-2 pl-5 pr-1 py-1 rounded hover:bg-gray-100 cursor-pointer text-xs";
            const check = document.createElement("input");
            check.type = "checkbox";
            check.className = "accent-indigo-600 w-3 h-3";
            check.checked = state.activeClasses.has(name);
            check.onchange = () => toggleClass(name);
            const dot = document.createElement("span");
            dot.className = "w-2.5 h-2.5 rounded-full shrink-0";
            dot.style.background = state.class_colors[name] || "#ccc";
            const label = document.createElement("span");
            label.className = "truncate flex-1 text-gray-700";
            label.textContent = name;
            const n = document.createElement("span");
            n.className = "text-[10px] font-mono " + (counts[name] ? "text-gray-700 font-bold" : "text-gray-300");
            n.textContent = counts[name] || 0;
            n.title = "Detections in this frame above the minimum score";
            row.append(check, dot, label, n);
            container.appendChild(row);
        });
    });
}

function toggleClass(className) {
    const active = currentDatavizState.activeClasses;
    active.has(className) ? active.delete(className) : active.add(className);
    renderFrame();
}

// --- Per-frame detection list: hover to highlight, click to show only that class ---
function renderFrameList(detections) {
    document.getElementById('stat-total').innerText = detections.length;
    const list = document.getElementById('stat-list');
    list.innerHTML = '';

    if(detections.length === 0) {
        list.innerHTML = '<div class="w-full text-center text-[10px] text-gray-400 italic">No visible detections</div>';
        return;
    }

    detections.forEach((box, index) => {
        const row = document.createElement("button");
        row.className = "px-2 py-1 rounded flex items-center gap-2 text-[11px] text-left hover:bg-indigo-50";
        row.title = "Hover to highlight; click to show only this class";
        const dot = document.createElement("span");
        dot.className = "w-2.5 h-2.5 rounded-full shrink-0";
        dot.style.background = box.color;
        const name = document.createElement("span");
        name.className = "flex-1 truncate text-gray-700";
        name.textContent = box.class_name;
        const score = document.createElement("span");
        score.className = "font-mono text-gray-500";
        score.textContent = box.confidence == null ? "" : box.confidence.toFixed(2);
        row.append(dot, name, score);
        row.onmouseenter = () => setHighlight(index);
        row.onmouseleave = () => setHighlight(null);
        row.onclick = () => { currentDatavizState.activeClasses = new Set([box.class_name]); renderFrame(); };
        list.appendChild(row);
    });
}

function setHighlight(index) {
    currentDatavizState.highlight = index;
    renderBBoxes(currentDatavizState.frames[currentDatavizState.currentIndex], false);
}

// --- Navigation & Global Toggles ---
function jumpFrame(offset) {
    let newIndex = currentDatavizState.currentIndex + offset;
    newIndex = Math.max(0, Math.min(newIndex, currentDatavizState.frames.length - 1));

    if(newIndex !== currentDatavizState.currentIndex) {
        currentDatavizState.currentIndex = newIndex;
        renderFrame();
    }
}

function goToFrame(index) {
    let newIndex = parseInt(index);
    if(isNaN(newIndex)) return;
    newIndex = Math.max(0, Math.min(newIndex, currentDatavizState.frames.length - 1));
    currentDatavizState.currentIndex = newIndex;
    renderFrame();
}

function toggleBaseImage() { document.getElementById("base-image").style.opacity = document.getElementById("cb-base-img").checked ? '1' : '0'; }
function toggleAllBBoxLayer() { renderBBoxes(currentDatavizState.frames[currentDatavizState.currentIndex]); }
function toggleMaskMaster() { document.getElementById("mask-layer").style.display = document.getElementById("cb-mask").checked ? 'block' : 'none'; }
function updateMaskOpacity(val) {
    document.getElementById("mask-layer").style.opacity = val / 100;
    document.getElementById("mask-opacity-label").innerText = val + '%';
}
// While dragging, update boxes and counts; the mask image is refetched when the slider is released.
function updateMinScore(val) {
    currentDatavizState.minScore = val / 100;
    document.getElementById("min-score-label").innerText = (val / 100).toFixed(2);
    renderBBoxes(currentDatavizState.frames[currentDatavizState.currentIndex]);
    renderClassList();
}

function renderGallery() {
    const gallery = document.getElementById("dataviz-gallery");
    gallery.innerHTML = "";

    const start = Math.max(0, currentDatavizState.currentIndex - 4);
    const end = Math.min(currentDatavizState.frames.length, currentDatavizState.currentIndex + 5);

    for(let i = start; i < end; i++) {
        const filename = currentDatavizState.frames[i];
        const img = document.createElement("img");
        img.src = `/api/dataviz/thumb/${encodeURIComponent(currentDatavizState.instance)}/${encodeURIComponent(filename)}`;

        const baseClass = "w-14 h-14 object-cover rounded cursor-pointer border-2 transition-all hover:-translate-y-1 shrink-0";
        img.className = i === currentDatavizState.currentIndex
            ? `${baseClass} border-indigo-500 shadow-lg`
            : `${baseClass} border-transparent opacity-50 hover:opacity-100`;

        img.onclick = () => {
            currentDatavizState.currentIndex = i;
            renderFrame();
        };
        gallery.appendChild(img);
    }
}

function resetDatavizUI() {
    document.getElementById("base-image").src = "";
    document.getElementById("mask-layer").src = "";
    document.getElementById("svg-layer").innerHTML = "";
    document.getElementById("dataviz-gallery").innerHTML = "";
    document.getElementById("dataviz-filename").innerText = "No image loaded";
    document.getElementById("dataviz-redaction").classList.add("hidden");
    document.getElementById("dataviz-presets").innerHTML = "";
    document.getElementById("dataviz-class-list").innerHTML = '<div class="text-xs text-gray-400 italic">No classes available.</div>';
    document.getElementById("stat-total").innerText = "0";
    document.getElementById("stat-list").innerHTML = "";
}
