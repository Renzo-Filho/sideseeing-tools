let currentVideoState = {
    instance: null,
    frames: [],
    currentIndex: 0
};

document.addEventListener("DOMContentLoaded", () => {
    const select = document.getElementById("video-instance-select");
    if(select && typeof SAMPLES_DATA !== 'undefined') {
        SAMPLES_DATA.forEach(sample => {
            const opt = document.createElement("option");
            opt.value = sample.name;
            opt.textContent = sample.name;
            select.appendChild(opt);
        });
        select.addEventListener("change", (e) => loadVideoInstanceData(e.target.value));
    }
});

async function loadVideoInstanceData(instanceName) {
    if(!instanceName) return;
    videoResetUI();
    
    try {
        const res = await fetch(`/api/dataviz/data/${instanceName}`);
        
        if (!res.ok) {
            throw new Error(`Server returned ${res.status}`);
        }
        
        const data = await res.json();
        
        currentVideoState.instance = instanceName;
        currentVideoState.frames = data.frames; 
        currentVideoState.currentIndex = 0;

        const errorOverlay = document.getElementById('video-error-overlay');
        const extractionOverlay = document.getElementById('video-extraction-overlay');
        const btnExtract = document.getElementById('video-btn-trigger-extract');
        
        errorOverlay.classList.add('hidden');

        if (data.frames.length === 0) {
            extractionOverlay.classList.remove('hidden');
            btnExtract.onclick = () => triggerVideoExtraction(instanceName);
            return; 
        }

        extractionOverlay.classList.add('hidden');
        videoRenderFrame();

    } catch (e) {
        console.error("Failed to load video frames data", e);
        // Show server required fallback
        document.getElementById('video-error-overlay').classList.remove('hidden');
    }
}

async function triggerVideoExtraction(instanceName) {
    const btn = document.getElementById('video-btn-trigger-extract');
    const progress = document.getElementById('video-extraction-progress');
    
    btn.classList.add('hidden');
    progress.classList.remove('hidden');
    progress.classList.add('flex');

    try {
        const res = await fetch(`/api/dataviz/extract/${instanceName}`, {
            method: 'POST'
        });
        
        if (!res.ok) throw new Error("Failed extraction");
        
        const data = await res.json();
        console.log("Extraction Status:", data.message);
        
        // Poll for frames
        const pollInterval = setInterval(async () => {
            try {
                const checkRes = await fetch(`/api/dataviz/data/${instanceName}`);
                const checkData = await checkRes.json();
                
                if (checkData.frames.length > 0) {
                    clearInterval(pollInterval);
                    loadVideoInstanceData(instanceName); // Reload
                }
            } catch (e) {
                console.error("Polling error", e);
            }
        }, 5000);

    } catch (e) {
        alert("Failed to start extraction. Check server logs.");
        btn.classList.remove('hidden');
        progress.classList.add('hidden');
    }
}

function videoRenderFrame() {
    if(currentVideoState.frames.length === 0) return;

    const filename = currentVideoState.frames[currentVideoState.currentIndex];
    document.getElementById("video-filename").innerText = `${filename}`;

    const imgUrl = `/api/dataviz/image/${currentVideoState.instance}/${filename}`;
    document.getElementById("video-base-image").src = imgUrl;

    document.getElementById("video-frame-input").value = currentVideoState.currentIndex + 1;
    document.getElementById("video-total-frames-count").innerText = currentVideoState.frames.length;

    videoRenderGallery();
}

function videoJumpFrame(offset) {
    let newIndex = currentVideoState.currentIndex + offset;
    newIndex = Math.max(0, Math.min(newIndex, currentVideoState.frames.length - 1));
    
    if(newIndex !== currentVideoState.currentIndex) {
        currentVideoState.currentIndex = newIndex;
        videoRenderFrame();
    }
}

function videoGoToFrame(index) {
    let newIndex = parseInt(index);
    if(isNaN(newIndex)) return;
    newIndex = Math.max(0, Math.min(newIndex, currentVideoState.frames.length - 1));
    currentVideoState.currentIndex = newIndex;
    videoRenderFrame();
}

function videoRenderGallery() {
    const gallery = document.getElementById("video-gallery");
    gallery.innerHTML = "";
    
    const start = Math.max(0, currentVideoState.currentIndex - 4);
    const end = Math.min(currentVideoState.frames.length, currentVideoState.currentIndex + 5);

    for(let i = start; i < end; i++) {
        const filename = currentVideoState.frames[i];
        const img = document.createElement("img");
        img.src = `/api/dataviz/thumb/${currentVideoState.instance}/${filename}`;
        
        const baseClass = "w-14 h-14 object-cover rounded cursor-pointer border-2 transition-all hover:-translate-y-1 shrink-0";
        img.className = i === currentVideoState.currentIndex 
            ? `${baseClass} border-orange-500 shadow-lg` 
            : `${baseClass} border-transparent opacity-50 hover:opacity-100`;
            
        img.onclick = () => {
            currentVideoState.currentIndex = i;
            videoRenderFrame();
        };
        gallery.appendChild(img);
    }
}

function videoResetUI() {
    document.getElementById("video-base-image").src = "";
    document.getElementById("video-gallery").innerHTML = "";
    document.getElementById("video-filename").innerText = "No image loaded";
    document.getElementById("video-error-overlay").classList.add('hidden');
    document.getElementById("video-extraction-overlay").classList.add('hidden');
}
