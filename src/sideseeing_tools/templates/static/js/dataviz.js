let currentDatavizState = {
    instance: null,
    frames: [],
    currentIndex: 0,
    bboxes: [],
    classes: [],
    activeClasses: new Set()
};

// Assuming SAMPLES_DATA is available globally from report.html
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
    
    // NEW: Wipe the UI clean before loading the new instance's data
    resetDatavizUI();
    
    try {
        const res = await fetch(`/api/vision/data/${instanceName}`);
        const data = await res.json();
        
        currentDatavizState.instance = instanceName;
        currentDatavizState.frames = data.frames; 
        currentDatavizState.bboxes = data.bboxes; 
        currentDatavizState.classes = data.classes; 
        currentDatavizState.activeClasses = new Set(data.classes);
        currentDatavizState.currentIndex = 0;

        const overlay = document.getElementById('extraction-overlay');
        const btnExtract = document.getElementById('btn-trigger-extract');

        // IF NO FRAMES: Show extraction UI
        if (data.frames.length === 0) {
            overlay.classList.remove('hidden');
            
            // Bind the extract button
            btnExtract.onclick = () => triggerExtraction(instanceName);
            
            // We can safely return here now, because the UI is completely blank underneath!
            return; 
        }

        // IF FRAMES EXIST: Hide overlay and render
        overlay.classList.add('hidden');
        renderClassList();
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
        // Call the FastAPI background task we built earlier
        const res = await fetch(`/api/vision/extract/${instanceName}`, {
            method: 'POST'
        });
        const data = await res.json();
        console.log("Extraction Status:", data.message);
        
        // Simple polling: Check back in 10 seconds to see if frames appeared
        // In a real production app, we'd poll a status endpoint
        const pollInterval = setInterval(async () => {
            const checkRes = await fetch(`/api/vision/data/${instanceName}`);
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
    document.getElementById("dataviz-filename").innerText = `${filename} (${currentDatavizState.currentIndex + 1} / ${currentDatavizState.frames.length})`;

    // 1. Update Base Image
    const imgUrl = `/api/vision/image/${currentDatavizState.instance}/${filename}`;
    document.getElementById("base-image").src = imgUrl;

    // 2. Update Mask Overlay
    const activeClassesStr = Array.from(currentDatavizState.activeClasses).join(",");
    const maskUrl = `/api/vision/mask/${filename}?classes=${activeClassesStr}&instance=${currentDatavizState.instance}`;
    document.getElementById("mask-layer").src = maskUrl;

    // 3. Render BBoxes
    renderBBoxes(filename);
    
    // 4. Render Gallery Thumbs
    renderGallery();
}

function renderBBoxes(filename) {
    const svg = document.getElementById("svg-layer");
    svg.innerHTML = ""; // Clear existing

    const boxes = currentDatavizState.bboxes[filename] || [];
    boxes.forEach(box => {
        if(currentDatavizState.activeClasses.has(box.class_name)) {
            const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            rect.setAttribute("x", box.xmin);
            rect.setAttribute("y", box.ymin);
            rect.setAttribute("width", box.xmax - box.xmin);
            rect.setAttribute("height", box.ymax - box.ymin);
            rect.setAttribute("class", "bbox-rect");
            rect.setAttribute("stroke", box.color || "#00ff00");
            
            const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
            title.textContent = `${box.class_name} (${box.confidence.toFixed(2)})`;
            rect.appendChild(title);
            
            svg.appendChild(rect);
        }
    });
}

function renderClassList() {
    const container = document.getElementById("dataviz-class-list");
    container.innerHTML = "";

    currentDatavizState.classes.forEach(cls => {
        const label = document.createElement("label");
        label.className = "flex items-center gap-2 px-2 py-1.5 rounded hover:bg-gray-100 cursor-pointer text-xs";
        
        const isChecked = currentDatavizState.activeClasses.has(cls) ? "checked" : "";
        label.innerHTML = `
            <input type="checkbox" ${isChecked} onchange="toggleClass('${cls}')" class="accent-indigo-600 w-3 h-3 rounded-sm">
            <span class="truncate flex-1 text-gray-700">${cls}</span>
        `;
        container.appendChild(label);
    });
}

// --- Interaction Controls ---

function toggleClass(className) {
    if(currentDatavizState.activeClasses.has(className)) {
        currentDatavizState.activeClasses.delete(className);
    } else {
        currentDatavizState.activeClasses.add(className);
    }
    renderFrame();
}

function toggleAllClasses() {
    const checkboxes = document.querySelectorAll("#dataviz-class-list input");
    checkboxes.forEach(cb => cb.click());
}

function prevFrame() {
    if(currentDatavizState.currentIndex > 0) {
        currentDatavizState.currentIndex--;
        renderFrame();
    }
}

function nextFrame() {
    if(currentDatavizState.currentIndex < currentDatavizState.frames.length - 1) {
        currentDatavizState.currentIndex++;
        renderFrame();
    }
}

function toggleBaseImage() {
    document.getElementById("base-image").style.opacity = document.getElementById("cb-base-img").checked ? '1' : '0';
}

function toggleAllBBoxLayer() {
    document.getElementById("svg-layer").style.opacity = document.getElementById("cb-all-bboxes").checked ? '1' : '0';
}

function toggleMaskMaster() {
    document.getElementById("mask-layer").style.display = document.getElementById("cb-mask").checked ? 'block' : 'none';
}

function updateMaskOpacity(val) {
    document.getElementById("mask-layer").style.opacity = val / 100;
    document.getElementById("mask-opacity-label").innerText = val + '%';
}

function renderGallery() {
    const gallery = document.getElementById("dataviz-gallery");
    gallery.innerHTML = "";
    
    // Show 3 frames before and 3 frames after
    const start = Math.max(0, currentDatavizState.currentIndex - 3);
    const end = Math.min(currentDatavizState.frames.length, currentDatavizState.currentIndex + 4);

    for(let i = start; i < end; i++) {
        const filename = currentDatavizState.frames[i];
        const img = document.createElement("img");
        img.src = `/api/vision/thumb/${currentDatavizState.instance}/${filename}`;
        
        const baseClass = "w-16 h-16 object-cover rounded cursor-pointer border-2 transition-all hover:-translate-y-1";
        img.className = i === currentDatavizState.currentIndex 
            ? `${baseClass} border-indigo-500 scale-110 shadow-lg` 
            : `${baseClass} border-transparent opacity-50 hover:opacity-100`;
            
        img.onclick = () => {
            currentDatavizState.currentIndex = i;
            renderFrame();
        };
        gallery.appendChild(img);
    }
}

function resetDatavizUI() {
    // Clear the main images to remove the "ghost" background
    document.getElementById("base-image").src = "";
    document.getElementById("mask-layer").src = "";
    
    // Clear the SVG bounding boxes
    document.getElementById("svg-layer").innerHTML = "";
    
    // Clear the bottom thumbnail gallery
    document.getElementById("dataviz-gallery").innerHTML = "";
    
    // Reset the frame counter text
    document.getElementById("dataviz-filename").innerText = "No image loaded";
    
    // Reset the classes sidebar
    document.getElementById("dataviz-class-list").innerHTML = '<div class="text-xs text-gray-400 italic">No classes available.</div>';
}