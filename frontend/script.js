// Disease Prediction System - Frontend JavaScript
const API_URL = 'http://localhost:8000';

// Global state
let currentChart = null;
let convergenceData = null;
let selectedFeatures = [];

// ============================================
// Initialize
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    checkServerStatus();
});

function initializeEventListeners() {
    // Upload form
    document.getElementById('uploadForm').addEventListener('submit', handleUpload);
    
    // Train form
    document.getElementById('trainForm').addEventListener('submit', handleTrain);
    
    // Prediction form
    document.getElementById('predictionForm').addEventListener('submit', handlePrediction);
    
    // Batch prediction form
    document.getElementById('batchPredictionForm').addEventListener('submit', handleBatchPrediction);
    
    // Tab switching
    document.querySelectorAll('.tabs .tab-btn').forEach(btn => {
        btn.addEventListener('click', handleTabSwitch);
    });
    
    // Chart tabs
    document.querySelectorAll('.chart-tabs .tab-btn').forEach(btn => {
        btn.addEventListener('click', handleChartTabSwitch);
    });
}

// ============================================
// Server Status
// ============================================

async function checkServerStatus() {
    try {
        const response = await fetch(`${API_URL}/status`);
        const data = await response.json();
        
        if (data.is_trained) {
            updateStatusBar('✅ Model trained and ready', 'success');
            await loadTrainedModelInfo();
        } else if (data.dataset_info) {
            updateStatusBar(`📊 Dataset loaded: ${data.dataset_info.dataset_name}`, 'info');
        }
    } catch (error) {
        updateStatusBar('⚠️ Server not connected', 'error');
    }
}

function updateStatusBar(message, type = 'info') {
    const statusBar = document.getElementById('statusBar');
    const statusText = document.getElementById('statusText');
    statusText.textContent = message;
    
    statusBar.className = 'status-bar';
    if (type === 'success') {
        statusBar.style.background = 'linear-gradient(135deg, #28a745, #20c997)';
    } else if (type === 'error') {
        statusBar.style.background = 'linear-gradient(135deg, #dc3545, #c82333)';
    } else {
        statusBar.style.background = 'linear-gradient(135deg, #667eea, #764ba2)';
    }
}

// ============================================
// Step 1: Upload Dataset
// ============================================

async function handleUpload(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('datasetFile');
    const file = fileInput.files[0];
    
    if (!file) {
        showResult('uploadResult', 'Please select a file', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        showResult('uploadResult', '⏳ Uploading and processing...', 'info');
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minute timeout for large files
        
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            const info = data.info;
            const message = `
                <strong>✅ ${data.message}</strong><br>
                Dataset: ${info.dataset_name}<br>
                Features: ${info.features.length}<br>
                Training samples: ${info.train_shape[0]}<br>
                Validation samples: ${info.val_shape[0]}<br>
                Test samples: ${info.test_shape[0]}
            `;
            showResult('uploadResult', message, 'success');
            updateStatusBar(`📊 Dataset loaded: ${info.dataset_name}`, 'info');
        } else {
            showResult('uploadResult', '❌ Upload failed', 'error');
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            showResult('uploadResult', '❌ Upload timeout - file too large or server too slow', 'error');
        } else {
            showResult('uploadResult', `❌ Error: ${error.message}`, 'error');
        }
    }
}

// ============================================
// Step 2: Train Model
// ============================================

async function handleTrain(e) {
    e.preventDefault();
    
    const generations = parseInt(document.getElementById('generations').value);
    const population = parseInt(document.getElementById('population').value);
    
    const progressDiv = document.getElementById('trainingProgress');
    const resultDiv = document.getElementById('trainResult');
    
    progressDiv.classList.remove('hidden');
    resultDiv.classList.add('hidden');
    
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 600000); // 10 minute timeout for training
        
        const response = await fetch(`${API_URL}/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ generations, population }),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        progressDiv.classList.add('hidden');
        
        if (data.success) {
            convergenceData = data.results.convergence;
            
            // Display ensemble metrics
            displayMetrics(data.results.metrics);
            
            // Display individual algorithm metrics
            displayAlgorithmMetrics(data.results);
            
            // Display selected features
            displaySelectedFeatures(data.results.selected_features, data.results.metrics);
            
            // Display confusion matrix
            displayConfusionMatrix(data.results.metrics.confusion_matrix);
            
            // Display convergence charts
            displayConvergenceCharts(convergenceData);
            
            showResult('trainResult', '✅ Training completed successfully!', 'success');
            updateStatusBar('✅ Model trained and ready', 'success');
            
            // Enable prediction
            await createPredictionForm();
        } else {
            showResult('trainResult', '❌ Training failed', 'error');
        }
    } catch (error) {
        progressDiv.classList.add('hidden');
        if (error.name === 'AbortError') {
            showResult('trainResult', '❌ Training timeout - try reducing generations/population (use 20/20)', 'error');
        } else {
            showResult('trainResult', `❌ Error: ${error.message}`, 'error');
        }
    }
}

function displayMetrics(metrics) {
    document.getElementById('metricsSection').classList.remove('hidden');
    
    document.getElementById('testAccuracy').textContent = (metrics.test_accuracy * 100).toFixed(2) + '%';
    document.getElementById('precision').textContent = (metrics.precision * 100).toFixed(2) + '%';
    document.getElementById('recall').textContent = (metrics.recall * 100).toFixed(2) + '%';
    document.getElementById('f1Score').textContent = (metrics.f1_score * 100).toFixed(2) + '%';
    document.getElementById('rocAuc').textContent = metrics.roc_auc ? metrics.roc_auc.toFixed(4) : 'N/A';
    document.getElementById('selectedFeatures').textContent = metrics.selected_features_count;
    
    // Feature counts
    document.getElementById('gaFeatures').textContent = metrics.ga_features;
    document.getElementById('psoFeatures').textContent = metrics.pso_features;
    document.getElementById('deFeatures').textContent = metrics.de_features;
    document.getElementById('ensembleFeatures').textContent = metrics.selected_features_count;
}

function displayAlgorithmMetrics(results) {
    // GA Metrics
    if (results.ga_metrics) {
        document.getElementById('gaFeaturesCount').textContent = results.ga_metrics.selected_features;
        document.getElementById('gaAccuracy').textContent = (results.ga_metrics.test_accuracy * 100).toFixed(2) + '%';
        document.getElementById('gaF1').textContent = results.ga_metrics.f1_score.toFixed(4);
    }
    
    // PSO Metrics
    if (results.pso_metrics) {
        document.getElementById('psoFeaturesCount').textContent = results.pso_metrics.selected_features;
        document.getElementById('psoAccuracy').textContent = (results.pso_metrics.test_accuracy * 100).toFixed(2) + '%';
        document.getElementById('psoF1').textContent = results.pso_metrics.f1_score.toFixed(4);
    }
    
    // DE Metrics
    if (results.de_metrics) {
        document.getElementById('deFeaturesCount').textContent = results.de_metrics.selected_features;
        document.getElementById('deAccuracy').textContent = (results.de_metrics.test_accuracy * 100).toFixed(2) + '%';
        document.getElementById('deF1').textContent = results.de_metrics.f1_score.toFixed(4);
    }
}

function displaySelectedFeatures(features, metrics) {
    const featuresList = document.getElementById('selectedFeaturesList');
    
    if (features.length === 0) {
        featuresList.innerHTML = '<p>No features selected</p>';
        document.getElementById('exportFeaturesBtn').style.display = 'none';
        return;
    }
    
    const ul = document.createElement('ul');
    features.forEach(feature => {
        const li = document.createElement('li');
        li.textContent = feature;
        ul.appendChild(li);
    });
    
    featuresList.innerHTML = '';
    featuresList.appendChild(ul);
    
    // Show export button
    document.getElementById('exportFeaturesBtn').style.display = 'inline-block';
    
    selectedFeatures = features;
}

function displayConfusionMatrix(cm) {
    const container = document.getElementById('confusionMatrix');
    
    // cm is [[TN, FP], [FN, TP]]
    const tn = cm[0][0];
    const fp = cm[0][1];
    const fn = cm[1][0];
    const tp = cm[1][1];
    
    container.innerHTML = `
        <div class="cm-cell tn">
            <div>True Negative</div>
            <div style="font-size: 1.5em; margin-top: 5px;">${tn}</div>
        </div>
        <div class="cm-cell fp">
            <div>False Positive</div>
            <div style="font-size: 1.5em; margin-top: 5px;">${fp}</div>
        </div>
        <div class="cm-cell fn">
            <div>False Negative</div>
            <div style="font-size: 1.5em; margin-top: 5px;">${fn}</div>
        </div>
        <div class="cm-cell tp">
            <div>True Positive</div>
            <div style="font-size: 1.5em; margin-top: 5px;">${tp}</div>
        </div>
    `;
}

function displayConvergenceCharts(data) {
    console.log('Displaying convergence charts with data:', data);
    const chartsSection = document.getElementById('chartsSection');
    
    if (!chartsSection) {
        console.error('chartsSection element not found!');
        return;
    }
    
    if (!data || !data.ga || !data.pso || !data.de) {
        console.error('Invalid convergence data:', data);
        return;
    }
    
    chartsSection.classList.remove('hidden');
    
    // Show combined chart initially
    showConvergenceChart('combined', data);
}

function showConvergenceChart(type, data) {
    const canvas = document.getElementById('convergenceChart');
    const ctx = canvas.getContext('2d');
    
    // Destroy previous chart
    if (currentChart) {
        currentChart.destroy();
    }
    
    let datasets = [];
    
    if (type === 'combined') {
        datasets = [
            {
                label: 'GA Best Fitness',
                data: data.ga,
                borderColor: '#ff6384',
                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                tension: 0.4
            },
            {
                label: 'PSO Best Fitness',
                data: data.pso,
                borderColor: '#36a2eb',
                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                tension: 0.4
            },
            {
                label: 'DE Best Fitness',
                data: data.de,
                borderColor: '#4bc0c0',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                tension: 0.4
            }
        ];
    } else if (type === 'ga') {
        datasets = [{
            label: 'GA Best Fitness',
            data: data.ga,
            borderColor: '#ff6384',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            fill: true,
            tension: 0.4
        }];
    } else if (type === 'pso') {
        datasets = [{
            label: 'PSO Best Fitness',
            data: data.pso,
            borderColor: '#36a2eb',
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            fill: true,
            tension: 0.4
        }];
    } else if (type === 'de') {
        datasets = [{
            label: 'DE Best Fitness',
            data: data.de,
            borderColor: '#4bc0c0',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            fill: true,
            tension: 0.4
        }];
    }
    
    currentChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.ga.map((_, i) => i + 1),
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                title: {
                    display: true,
                    text: `${type.toUpperCase()} Convergence Analysis`,
                    font: { size: 16 }
                },
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Generation'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Best Fitness'
                    }
                }
            }
        }
    });
}

// ============================================
// Step 3: Create Prediction Form
// ============================================

async function createPredictionForm() {
    try {
        const response = await fetch(`${API_URL}/selected_features`);
        const data = await response.json();
        
        const container = document.getElementById('featureInputs');
        container.innerHTML = '';
        
        data.selected_features.forEach(feature => {
            const group = document.createElement('div');
            group.className = 'feature-input-group';
            
            const label = document.createElement('label');
            label.textContent = feature;
            label.setAttribute('for', feature);
            
            const input = document.createElement('input');
            input.type = 'number';
            input.step = 'any';
            input.id = feature;
            input.name = feature;
            input.required = true;
            input.placeholder = 'Enter value';
            
            group.appendChild(label);
            group.appendChild(input);
            container.appendChild(group);
        });
        
        document.getElementById('predictBtn').disabled = false;
    } catch (error) {
        console.error('Error creating prediction form:', error);
    }
}

async function loadTrainedModelInfo() {
    await createPredictionForm();
    
    try {
        const response = await fetch(`${API_URL}/metrics`);
        const data = await response.json();
        
        // Show metrics section
        document.getElementById('metricsSection').classList.remove('hidden');
        document.getElementById('chartsSection').classList.remove('hidden');
        
        // Display ensemble metrics
        displayMetrics(data.metrics);
        
        // Display individual algorithm metrics
        displayAlgorithmMetrics(data);
        
        // Display selected features
        displaySelectedFeatures(data.selected_features, data.metrics);
        
        // Display convergence charts if available
        if (data.convergence) {
            convergenceData = data.convergence;
            displayConvergenceCharts(convergenceData);
        }
        
        // Display confusion matrix if available
        if (data.metrics && data.metrics.confusion_matrix) {
            displayConfusionMatrix(data.metrics.confusion_matrix);
        }
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

// ============================================
// Step 4: Make Predictions
// ============================================

async function handlePrediction(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const features = Object.fromEntries(formData);
    
    // Convert to numbers
    for (let key in features) {
        features[key] = parseFloat(features[key]);
    }
    
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            displayPredictionResult(data.prediction);
        } else {
            alert('Prediction failed: ' + (data.message || 'Unknown error'));
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

function displayPredictionResult(result) {
    const resultDiv = document.getElementById('predictionResult');
    resultDiv.classList.remove('hidden');
    
    // Main prediction
    const predictionText = result.prediction === 1 ? 'POSITIVE (Disease)' : 'NEGATIVE (No Disease)';
    document.getElementById('predictionValue').textContent = predictionText;
    document.getElementById('predictionValue').style.color = result.prediction === 1 ? '#dc3545' : '#28a745';
    
    // Confidence
    const confidence = (result.confidence * 100).toFixed(2);
    document.getElementById('confidenceValue').textContent = confidence + '%';
    document.getElementById('confidenceFill').style.width = confidence + '%';
    
    // Individual predictions
    document.getElementById('lrPred').textContent = result.individual_predictions.lr;
    document.getElementById('rfPred').textContent = result.individual_predictions.rf;
    document.getElementById('svmPred').textContent = result.individual_predictions.svm;
    document.getElementById('nnPred').textContent = result.individual_predictions.nn;
    
    // Probability chart
    displayProbabilityChart(result);
    
    // Scroll to result
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function displayProbabilityChart(result) {
    const canvas = document.getElementById('probabilityChart');
    const ctx = canvas.getContext('2d');
    
    // Destroy previous chart if exists
    if (canvas.chart) {
        canvas.chart.destroy();
    }
    
    canvas.chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Voting Ensemble', 'Neural Network'],
            datasets: [
                {
                    label: 'Class 0 (Negative)',
                    data: [result.voting_proba[0] * 100, result.nn_proba[0] * 100],
                    backgroundColor: 'rgba(40, 167, 69, 0.7)'
                },
                {
                    label: 'Class 1 (Positive)',
                    data: [result.voting_proba[1] * 100, result.nn_proba[1] * 100],
                    backgroundColor: 'rgba(220, 53, 69, 0.7)'
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Model Probability Distribution',
                    font: { size: 16 }
                },
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Probability (%)'
                    }
                }
            }
        }
    });
}

async function handleBatchPrediction(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('predictionFile');
    const file = fileInput.files[0];
    
    if (!file) {
        showResult('batchResults', 'Please select a file', 'error');
        return;
    }
    
    // Read CSV and make predictions
    const reader = new FileReader();
    reader.onload = async (event) => {
        try {
            const csv = event.target.result;
            const lines = csv.split('\n');
            const headers = lines[0].split(',').map(h => h.trim());
            
            const data = [];
            for (let i = 1; i < lines.length; i++) {
                if (lines[i].trim() === '') continue;
                
                const values = lines[i].split(',');
                const obj = {};
                headers.forEach((header, idx) => {
                    obj[header] = parseFloat(values[idx]);
                });
                data.push(obj);
            }
            
            const response = await fetch(`${API_URL}/predict_batch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                displayBatchResults(result.predictions);
            }
        } catch (error) {
            showResult('batchResults', `Error: ${error.message}`, 'error');
        }
    };
    reader.readAsText(file);
}

function displayBatchResults(predictions) {
    const container = document.getElementById('batchResults');
    container.classList.remove('hidden');
    
    let html = '<h4>Batch Prediction Results</h4><table style="width:100%; border-collapse: collapse;">';
    html += '<tr><th>Sample</th><th>Prediction</th><th>Confidence</th></tr>';
    
    predictions.forEach((pred, idx) => {
        const predText = pred.prediction === 1 ? 'Positive' : 'Negative';
        const confidence = (pred.confidence * 100).toFixed(2);
        html += `<tr>
            <td>${idx + 1}</td>
            <td style="color: ${pred.prediction === 1 ? '#dc3545' : '#28a745'}">${predText}</td>
            <td>${confidence}%</td>
        </tr>`;
    });
    
    html += '</table>';
    container.innerHTML = html;
    container.style.background = '#d4edda';
    container.style.padding = '20px';
    container.style.borderRadius = '8px';
}

// ============================================
// Tab Switching
// ============================================

function handleTabSwitch(e) {
    const targetTab = e.target.dataset.tab;
    
    // Update tab buttons
    e.target.parentElement.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    e.target.classList.add('active');
    
    // Update tab content
    if (targetTab === 'manual') {
        document.getElementById('manualTab').classList.add('active');
        document.getElementById('manualTab').classList.remove('hidden');
        document.getElementById('fileTab').classList.add('hidden');
    } else if (targetTab === 'file') {
        document.getElementById('manualTab').classList.add('hidden');
        document.getElementById('manualTab').classList.remove('active');
        document.getElementById('fileTab').classList.remove('hidden');
        document.getElementById('fileTab').classList.add('active');
    }
}

function handleChartTabSwitch(e) {
    const chartType = e.target.dataset.chart;
    
    // Update active button
    e.target.parentElement.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    e.target.classList.add('active');
    
    // Show selected chart
    if (convergenceData) {
        showConvergenceChart(chartType, convergenceData);
    }
}

// ============================================
// Utility Functions
// ============================================

function showResult(elementId, message, type) {
    const element = document.getElementById(elementId);
    element.innerHTML = message;
    element.className = `result ${type}`;
    element.classList.remove('hidden');
}

// ============================================
// Export Selected Features
// ============================================

async function exportSelectedFeatures() {
    try {
        const response = await fetch(`${API_URL}/export_features`);
        
        if (!response.ok) {
            throw new Error('Failed to export features');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'selected_features_ensemble.csv';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        alert('✅ Selected features exported successfully!');
    } catch (error) {
        alert('❌ Error exporting features: ' + error.message);
    }
}
