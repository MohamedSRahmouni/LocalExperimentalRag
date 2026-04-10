// File Upload Functionality
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const uploadBtn = document.getElementById('uploadBtn');
const uploadForm = document.getElementById('uploadForm');
const alertContainer = document.getElementById('alertContainer');
let selectedFiles = [];

// Prevent default drag behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
});

// Highlight drop zone when item is dragged over it
['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, unhighlight, false);
});

// Handle dropped files
dropZone.addEventListener('drop', handleDrop, false);

// Handle file input change
fileInput.addEventListener('change', handleFiles, false);

// Click to upload
dropZone.addEventListener('click', () => fileInput.click());

// Handle form submission
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    await uploadFiles();
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight(e) {
    dropZone.classList.add('drag-over');
}

function unhighlight(e) {
    dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles({ target: { files: files } });
}

function handleFiles(e) {
    const files = [...e.target.files];
    selectedFiles = files;
    displayFiles(files);
    uploadBtn.disabled = files.length === 0;
}

function displayFiles(files) {
    fileList.innerHTML = '';
    
    files.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div>
                <strong>${file.name}</strong>
                <span style="color: var(--text-secondary); margin-left: 1rem;">
                    ${formatFileSize(file.size)}
                </span>
            </div>
            <button type="button" onclick="removeFile(${index})">Remove</button>
        `;
        fileList.appendChild(fileItem);
    });
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    
    // Create new FileList
    const dt = new DataTransfer();
    selectedFiles.forEach(file => dt.items.add(file));
    fileInput.files = dt.files;
    
    displayFiles(selectedFiles);
    uploadBtn.disabled = selectedFiles.length === 0;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

async function uploadFiles() {
    if (selectedFiles.length === 0) return;
    
    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files', file);
    });
    
    // Show loading state
    uploadBtn.disabled = true;
    document.getElementById('uploadBtnText').textContent = 'Uploading...';
    document.getElementById('uploadSpinner').style.display = 'inline-block';
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showAlert(data.message, 'success');
            
            // Clear file list
            selectedFiles = [];
            fileList.innerHTML = '';
            fileInput.value = '';
            uploadBtn.disabled = true;
            
            // Redirect to prompt page after 2 seconds
            setTimeout(() => {
                window.location.href = '/prompt';
            }, 2000);
        } else {
            showAlert(data.message, 'error');
        }
    } catch (error) {
        showAlert('Error uploading files: ' + error.message, 'error');
    } finally {
        // Reset button state
        document.getElementById('uploadBtnText').textContent = 'Upload Files';
        document.getElementById('uploadSpinner').style.display = 'none';
        uploadBtn.disabled = false;
    }
}

function showAlert(message, type) {
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.textContent = message;
    
    alertContainer.innerHTML = '';
    alertContainer.appendChild(alert);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alert.remove();
    }, 5000);
}