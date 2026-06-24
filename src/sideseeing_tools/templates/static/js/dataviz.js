let currentDatavizState = {
    instance: null,
    frames: [],
    currentIndex: 0,
    bboxes: [],
    box_classes: [],
    mask_classes: [],
    activeBoxClasses: new Set(),
    activeMaskClasses: new Set(),
    activeTab: 'boxes' // 'boxes' or 'masks'
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
        const res = await fetch(`/api/dataviz/data/${instanceName}`);
        const data = await res.json();
        
        currentDatavizState.instance = instanceName;
        currentDatavizState.frames = data.frames; 
        currentDatavizState.bboxes = data.bboxes; 
        currentDatavizState.box_classes = data.box_classes; 
        currentDatavizState.mask_classes = data.mask_classes; 
        currentDatavizState.class_colors = data.class_colors || {};
        
        // Turn everything on by default
        currentDatavizState.activeBoxClasses = new Set(data.box_classes);
        currentDatavizState.activeMaskClasses = new Set(data.mask_classes);
        currentDatavizState.currentIndex = 0;

        const overlay = document.getElementById('extraction-overlay');
        const btnExtract = document.getElementById('btn-trigger-extract');

        if (data.frames.length === 0) {
            overlay.classList.remove('hidden');
            btnExtract.onclick = () => triggerExtraction(instanceName);
            return; 
        }

        overlay.classList.add('hidden');
        
        // Disable tabs if data doesn't exist
        // (Removed to allow free switching and show empty state messages instead)
        
        // Auto-switch to masks if no boxes exist
        if(data.box_classes.length === 0 && data.mask_classes.length > 0) {
            switchTab('masks');
        } else {
            switchTab('boxes');
        }
        
        renderFrame();

    } catch (e) {
        console.error("Failed to load instance AI data", e);
    }
}

// Extraction polling remains exactly the same...
async function triggerExtraction(instanceName) {
    const btn = document.getElementById('btn-trigger-extract');
    const progress = document.getElementById('extraction-progress');
    
    btn.classList.add('hidden');
    progress.classList.remove('hidden');
    progress.classList.add('flex');

    try {
        // Call the FastAPI background task we built earlier
        const res = await fetch(`/api/dataviz/extract/${instanceName}`, {
            method: 'POST'
        });
        const data = await res.json();
        console.log("Extraction Status:", data.message);
        
        // Simple polling: Check back in 10 seconds to see if frames appeared
        // In a real production app, we'd poll a status endpoint
        const pollInterval = setInterval(async () => {
            const checkRes = await fetch(`/api/dataviz/data/${instanceName}`);
            const checkData = await checkRes.json();
            
            if (checkData.frames.length > 0) {
                clearInterval(pollInterval);
                loadInstanceData(instanceName); // Reload!
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

    const filename = currentDatavizState.frames[currentDatavizState.currentIndex];
    document.getElementById("dataviz-filename").innerText = `${filename}`;

    // 1. Update Base Image
    const imgUrl = `/api/dataviz/image/${currentDatavizState.instance}/${filename}`;
    const baseImage = document.getElementById("base-image");
    baseImage.src = imgUrl;
    
    baseImage.onload = () => {
        const svg = document.getElementById("svg-layer");
        svg.setAttribute("viewBox", `0 0 ${baseImage.naturalWidth} ${baseImage.naturalHeight}`);
        const canvas = document.getElementById("dataviz-canvas");
        canvas.style.aspectRatio = `${baseImage.naturalWidth} / ${baseImage.naturalHeight}`;
    };

    // 2. Update Mask Overlay (Pass only active mask classes)
    const activeMasksStr = Array.from(currentDatavizState.activeMaskClasses).join(",");
    const maskUrl = `/api/dataviz/mask/${filename}?classes=${activeMasksStr}&instance=${currentDatavizState.instance}`;
    document.getElementById("mask-layer").src = maskUrl;

    // 3. Update Inputs
    document.getElementById("frame-input").value = currentDatavizState.currentIndex + 1;
    document.getElementById("total-frames-count").innerText = currentDatavizState.frames.length;

    // 4. Render BBoxes
    renderBBoxes(filename);
    
    // 5. Render Gallery Thumbs
    renderGallery();
}

function renderBBoxes(filename) {
    const svg = document.getElementById("svg-layer");
    svg.innerHTML = ""; 

    const boxes = currentDatavizState.bboxes[filename] || [];
    boxes.forEach(box => {
        // Only draw if the box class is currently active
        if(currentDatavizState.activeBoxClasses.has(box.class_name)) {
            const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
            group.setAttribute("class", `bbox-group layer-${box.class_name.replace(/[^a-zA-Z0-9]/g, '-')}`);
            
            const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            rect.setAttribute("x", box.xmin);
            rect.setAttribute("y", box.ymin);
            rect.setAttribute("width", box.xmax - box.xmin);
            rect.setAttribute("height", box.ymax - box.ymin);
            rect.setAttribute("fill", "none");
            rect.setAttribute("stroke", box.color || "#00ff00");
            rect.setAttribute("stroke-width", "2");
            rect.style.transition = "stroke-width 0.2s, opacity 0.2s";

            const textBg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            textBg.setAttribute("x", box.xmin);
            textBg.setAttribute("y", box.ymin - 16);
            textBg.setAttribute("width", (box.class_name.length * 6) + 8);
            textBg.setAttribute("height", 16);
            textBg.setAttribute("fill", box.color || "#00ff00");
            textBg.setAttribute("rx", "2");

            const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
            text.setAttribute("x", box.xmin + 4);
            text.setAttribute("y", box.ymin - 4);
            text.setAttribute("fill", "white");
            text.setAttribute("font-size", "10px");
            text.setAttribute("font-family", "sans-serif");
            text.setAttribute("font-weight", "bold");
            text.textContent = box.class_name;
            
            group.appendChild(rect);
            group.appendChild(textBg);
            group.appendChild(text);
            svg.appendChild(group);
        }
    });

    calculateStats(boxes);
}

// --- Tab & Class Selection Logic ---
function switchTab(tabName) {
    currentDatavizState.activeTab = tabName;
    
    // UI Updates
    const btnBoxes = document.getElementById('tab-btn-boxes');
    const btnMasks = document.getElementById('tab-btn-masks');
    
    if(tabName === 'boxes') {
        btnBoxes.className = "flex-1 py-2 text-[10px] font-bold uppercase tracking-wider text-center border-b-2 border-indigo-500 text-indigo-600 bg-indigo-50 transition-colors";
        btnMasks.className = "flex-1 py-2 text-[10px] font-bold uppercase tracking-wider text-center border-b-2 border-transparent text-gray-500 hover:bg-gray-50 transition-colors";
        document.getElementById('dataviz-stats-container').style.display = 'block';
    } else {
        btnMasks.className = "flex-1 py-2 text-[10px] font-bold uppercase tracking-wider text-center border-b-2 border-indigo-500 text-indigo-600 bg-indigo-50 transition-colors";
        btnBoxes.className = "flex-1 py-2 text-[10px] font-bold uppercase tracking-wider text-center border-b-2 border-transparent text-gray-500 hover:bg-gray-50 transition-colors";
        document.getElementById('dataviz-stats-container').style.display = 'none'; // Stats only for boxes currently
    }
    
    renderClassList();
}

function renderClassList() {
    const container = document.getElementById("dataviz-class-list");
    container.innerHTML = "";
    
    const isBoxes = currentDatavizState.activeTab === 'boxes';
    const classSource = isBoxes ? currentDatavizState.box_classes : currentDatavizState.mask_classes;
    const activeSet = isBoxes ? currentDatavizState.activeBoxClasses : currentDatavizState.activeMaskClasses;

    if(classSource.length === 0) {
        container.innerHTML = `<div class="text-xs text-gray-400 italic">No ${isBoxes ? 'box' : 'mask'} classes found.</div>`;
        return;
    }

    classSource.forEach(cls => {
        const label = document.createElement("label");
        label.className = "flex items-center gap-2 px-2 py-1.5 rounded hover:bg-gray-100 cursor-pointer text-xs transition-colors";
        
        if (isBoxes) {
            const safeSlug = cls.replace(/[^a-zA-Z0-9]/g, '-');
            label.addEventListener('mouseenter', () => {
                document.querySelectorAll(`.layer-${safeSlug} rect:first-child`).forEach(r => r.setAttribute("stroke-width", "4"));
            });
            label.addEventListener('mouseleave', () => {
                document.querySelectorAll(`.layer-${safeSlug} rect:first-child`).forEach(r => r.setAttribute("stroke-width", "2"));
            });
        }
        
        const isChecked = activeSet.has(cls) ? "checked" : "";
        const color = currentDatavizState.class_colors[cls] || '#ccc';
        label.innerHTML = `
            <input type="checkbox" ${isChecked} onchange="toggleClass('${cls}')" class="accent-indigo-600 w-3 h-3 rounded-sm">
            <span class="w-3 h-3 rounded-full inline-block shrink-0 shadow-sm border border-gray-200" style="background-color: ${color};"></span>
            <span class="truncate flex-1 text-gray-700">${cls}</span>
        `;
        container.appendChild(label);
    });
}

function toggleClass(className) {
    const activeSet = currentDatavizState.activeTab === 'boxes' 
        ? currentDatavizState.activeBoxClasses 
        : currentDatavizState.activeMaskClasses;

    if(activeSet.has(className)) {
        activeSet.delete(className);
    } else {
        activeSet.add(className);
    }
    renderFrame();
}

function toggleAllClasses() {
    const checkboxes = document.querySelectorAll("#dataviz-class-list input");
    checkboxes.forEach(cb => cb.click());
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
function toggleAllBBoxLayer() { document.getElementById("svg-layer").style.opacity = document.getElementById("cb-all-bboxes").checked ? '1' : '0'; }
function toggleMaskMaster() { document.getElementById("mask-layer").style.display = document.getElementById("cb-mask").checked ? 'block' : 'none'; }
function updateMaskOpacity(val) {
    document.getElementById("mask-layer").style.opacity = val / 100;
    document.getElementById("mask-opacity-label").innerText = val + '%';
}

function calculateStats(boxes) {
    const counts = {};
    let total = 0;
    boxes.forEach(box => {
        if(currentDatavizState.activeBoxClasses.has(box.class_name)) {
            counts[box.class_name] = (counts[box.class_name] || 0) + 1;
            total++;
        }
    });

    document.getElementById('stat-total').innerText = total;
    const list = document.getElementById('stat-list');
    
    if(total === 0) { 
        list.innerHTML = '<div class="w-full text-center text-[10px] text-gray-400 italic">No detections</div>'; 
        return; 
    }
    
    let html = '';
    Object.entries(counts).sort((a,b) => b[1] - a[1]).forEach(([cls, count]) => {
        html += `
        <div class="px-2 py-1 bg-gray-50 rounded border border-gray-100 flex items-center justify-between text-[10px]">
            <span class="text-gray-600 truncate max-w-[120px]" title="${cls}">${cls}</span>
            <span class="font-bold text-gray-900">${count}</span>
        </div>`;
    });
    list.innerHTML = html;
}

function renderGallery() {
    const gallery = document.getElementById("dataviz-gallery");
    gallery.innerHTML = "";
    
    const start = Math.max(0, currentDatavizState.currentIndex - 4);
    const end = Math.min(currentDatavizState.frames.length, currentDatavizState.currentIndex + 5);

    for(let i = start; i < end; i++) {
        const filename = currentDatavizState.frames[i];
        const img = document.createElement("img");
        img.src = `/api/dataviz/thumb/${currentDatavizState.instance}/${filename}`;
        
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
    document.getElementById("dataviz-class-list").innerHTML = '<div class="text-xs text-gray-400 italic">No classes available.</div>';
    document.getElementById("stat-total").innerText = "0";
    document.getElementById("stat-list").innerHTML = "";
}