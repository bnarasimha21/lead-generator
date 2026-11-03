let mediaRecorder;
let audioChunks = [];
let lastLead = null;

const FIELD_LABELS = {
    name: "Name",
    company: "Company",
    role: "Role",
    email: "Email",
    phone: "Phone",
    interest: "Interest",
    next_step: "Next step",
    budget_mentioned: "Budget mentioned",
    urgency: "Urgency",
};

function setupRecorder() {
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const audioPlayback = document.getElementById('audioPlayback');

    if (recordBtn) {
        recordBtn.onclick = async function () {
            audioChunks = [];
            let stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            // Use the browser's real recording mimetype; do not lie and call it WAV.
            mediaRecorder = new MediaRecorder(stream);
            const recordedType = mediaRecorder.mimeType || 'audio/webm';
            mediaRecorder.ondataavailable = e => {
                audioChunks.push(e.data);
            };
            mediaRecorder.onstop = function () {
                let blob = new Blob(audioChunks, { type: recordedType });
                let url = URL.createObjectURL(blob);
                if (audioPlayback) {
                    audioPlayback.src = url;
                    audioPlayback.style.display = 'block';
                }
                sendAudioForTranscription(blob, recordedType);
            };
            mediaRecorder.start();
            recordBtn.disabled = true;
            stopBtn.disabled = false;
        };
    }

    if (stopBtn) {
        stopBtn.onclick = function () {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                recordBtn.disabled = false;
                stopBtn.disabled = true;
            }
        };
    }

    const exportBtn = document.getElementById('exportCsvBtn');
    if (exportBtn) {
        exportBtn.onclick = exportLeadAsCsv;
    }
}

function extensionForType(type) {
    if (type.includes('webm')) return 'webm';
    if (type.includes('mp4') || type.includes('m4a')) return 'mp4';
    if (type.includes('ogg')) return 'ogg';
    if (type.includes('wav')) return 'wav';
    return 'webm';
}

function sendAudioForTranscription(blob, recordedType) {
    showLoading();
    let formData = new FormData();
    const ext = extensionForType(recordedType);
    formData.append('file', blob, `recording.${ext}`);
    fetch('/upload-audio', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json().then(body => ({ ok: response.ok, status: response.status, body })))
        .then(({ ok, status, body }) => {
            hideLoading();
            if (!ok) {
                showError(body.error || `Request failed (${status}).`);
                return;
            }
            renderResult(body);
        })
        .catch(err => {
            hideLoading();
            showError('Could not reach the server. Please try again.');
        });
}

function clearResults() {
    const existing = document.querySelectorAll('.result-section, .error-section');
    existing.forEach(el => el.remove());
}

function showError(text) {
    clearResults();
    const container = document.querySelector('.container');
    const box = document.createElement('div');
    box.className = 'result-section error-section';
    const p = document.createElement('p');
    p.textContent = text; // textContent, never innerHTML, for anything derived from user input
    box.appendChild(p);
    container.appendChild(box);
}

function renderResult(body) {
    clearResults();
    const container = document.querySelector('.container');
    lastLead = body.lead;

    const transcriptBox = document.createElement('div');
    transcriptBox.className = 'result-section';
    const h2a = document.createElement('h2');
    h2a.textContent = 'Transcription';
    const p = document.createElement('p');
    p.textContent = body.transcript || '';
    transcriptBox.appendChild(h2a);
    transcriptBox.appendChild(p);
    container.appendChild(transcriptBox);

    if (body.message) {
        const msgBox = document.createElement('div');
        msgBox.className = 'result-section';
        const msgP = document.createElement('p');
        msgP.textContent = body.message;
        msgBox.appendChild(msgP);
        container.appendChild(msgBox);
    }

    const leadBox = document.createElement('div');
    leadBox.className = 'result-section';
    const h2b = document.createElement('h2');
    h2b.textContent = 'Extracted Lead';
    leadBox.appendChild(h2b);

    const ul = document.createElement('ul');
    Object.keys(FIELD_LABELS).forEach(key => {
        const field = (body.lead && body.lead[key]) || { value: null, evidence: null, rejected: false };
        const li = document.createElement('li');
        if (field.rejected) {
            li.className = 'rejected-field';
        }
        const label = document.createElement('b');
        label.textContent = FIELD_LABELS[key] + ': ';
        li.appendChild(label);

        const valueSpan = document.createElement('span');
        if (field.rejected) {
            valueSpan.textContent = 'could not confirm';
        } else {
            valueSpan.textContent = field.value !== null && field.value !== undefined ? field.value : '(not stated)';
        }
        li.appendChild(valueSpan);

        if (field.evidence) {
            const quote = document.createElement('i');
            quote.className = 'evidence-quote';
            quote.textContent = ` "${field.evidence}"`;
            li.appendChild(quote);
        }
        ul.appendChild(li);
    });
    leadBox.appendChild(ul);

    const exportBtn = document.createElement('button');
    exportBtn.id = 'exportCsvBtn';
    exportBtn.textContent = 'Export CSV';
    exportBtn.onclick = exportLeadAsCsv;
    leadBox.appendChild(exportBtn);

    container.appendChild(leadBox);
}

// CSV cells are escaped against formula injection (OWASP's documented
// character set): a cell that starts with =, +, -, @, a tab (0x09), or a
// carriage return (0x0D) is prefixed with a single quote, so a
// spreadsheet never evaluates it as a formula. A leading tab or CR is
// included because Excel still treats it as leading whitespace and goes
// on to evaluate the formula that follows it.
function escapeCsvCell(raw) {
    let text = raw === null || raw === undefined ? '' : String(raw);
    if (/^[=+\-@\t\r]/.test(text)) {
        text = "'" + text;
    }
    text = text.replace(/"/g, '""');
    return `"${text}"`;
}

function exportLeadAsCsv() {
    if (!lastLead) return;
    const rows = [['Field', 'Value', 'Evidence']];
    Object.keys(FIELD_LABELS).forEach(key => {
        const field = lastLead[key] || {};
        rows.push([FIELD_LABELS[key], field.rejected ? '' : (field.value || ''), field.evidence || '']);
    });
    const csv = rows.map(row => row.map(escapeCsvCell).join(',')).join('\r\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'lead.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function showLoading() {
    let container = document.querySelector('.container');
    let loading = document.createElement('div');
    loading.className = 'loading-indicator';
    loading.style.margin = '1.5rem 0';
    const b = document.createElement('b');
    b.textContent = 'Transcribing...';
    loading.appendChild(b);
    container.appendChild(loading);
}

function hideLoading() {
    let loading = document.querySelector('.loading-indicator');
    if (loading) loading.remove();
}

// Initial setup
setupRecorder();
