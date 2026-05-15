// Disease Prediction System - Frontend JavaScript
const API_URL = 'http://localhost:8000';

// Global state
let currentChart = null;
let convergenceData = null;
let selectedFeatures = [];
const CHART_TEXT_COLOR = '#1f2937';
const CHART_MUTED_TEXT_COLOR = '#374151';
const CHART_GRID_COLOR = 'rgba(31, 41, 55, 0.12)';

function getChartPixelRatio() {
    return Math.max(2, Math.min(window.devicePixelRatio || 1, 3));
}
let allFeatureNames = [];

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
        
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
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
        showResult('uploadResult', `❌ Error: ${error.message}`, 'error');
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
        const response = await fetch(`${API_URL}/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ generations, population })
        });
        
        const data = await response.json();
        
        progressDiv.classList.add('hidden');
        
        if (data.success) {
            await ensureFeatureNamesLoaded();
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
        showResult('trainResult', `❌ Error: ${error.message}`, 'error');
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
    const totalFeatures = allFeatureNames.length || inferTotalFeatureCount(results);

    // GA Metrics
    if (results.ga_metrics) {
        const m = results.ga_metrics;
        document.getElementById('gaFeaturesCount').textContent = getSelectedCount(m);
        document.getElementById('gaAccuracy').textContent = (m.test_accuracy * 100).toFixed(2) + '%';
        document.getElementById('gaPrecision').textContent = formatMetricValue(m.precision, 6);
        document.getElementById('gaRecall').textContent = formatMetricValue(m.recall, 6);
        document.getElementById('gaF1').textContent = formatMetricValue(m.f1_score, 6);
        document.getElementById('gaRocAuc').textContent = formatMetricValue(m.roc_auc, 6);
        document.getElementById('gaTime').textContent = formatMetricValue(m.time_taken_sec, 4);
        document.getElementById('gaConfusionMatrix').textContent = formatConfusionMatrix(m.confusion_matrix);
        document.getElementById('gaSelectedIndexes').textContent = formatList(getSelectedIndexes(m));
        document.getElementById('gaNotSelectedIndexes').textContent = formatList(getNotSelectedIndexes(m, totalFeatures));
    }
    
    // PSO Metrics
    if (results.pso_metrics) {
        const m = results.pso_metrics;
        document.getElementById('psoFeaturesCount').textContent = getSelectedCount(m);
        document.getElementById('psoAccuracy').textContent = (m.test_accuracy * 100).toFixed(2) + '%';
        document.getElementById('psoPrecision').textContent = formatMetricValue(m.precision, 6);
        document.getElementById('psoRecall').textContent = formatMetricValue(m.recall, 6);
        document.getElementById('psoF1').textContent = formatMetricValue(m.f1_score, 6);
        document.getElementById('psoRocAuc').textContent = formatMetricValue(m.roc_auc, 6);
        document.getElementById('psoTime').textContent = formatMetricValue(m.time_taken_sec, 4);
        document.getElementById('psoConfusionMatrix').textContent = formatConfusionMatrix(m.confusion_matrix);
        document.getElementById('psoSelectedIndexes').textContent = formatList(getSelectedIndexes(m));
        document.getElementById('psoNotSelectedIndexes').textContent = formatList(getNotSelectedIndexes(m, totalFeatures));
    }
    
    // DE Metrics
    if (results.de_metrics) {
        const m = results.de_metrics;
        document.getElementById('deFeaturesCount').textContent = getSelectedCount(m);
        document.getElementById('deAccuracy').textContent = (m.test_accuracy * 100).toFixed(2) + '%';
        document.getElementById('dePrecision').textContent = formatMetricValue(m.precision, 6);
        document.getElementById('deRecall').textContent = formatMetricValue(m.recall, 6);
        document.getElementById('deF1').textContent = formatMetricValue(m.f1_score, 6);
        document.getElementById('deRocAuc').textContent = formatMetricValue(m.roc_auc, 6);
        document.getElementById('deTime').textContent = formatMetricValue(m.time_taken_sec, 4);
        document.getElementById('deConfusionMatrix').textContent = formatConfusionMatrix(m.confusion_matrix);
        document.getElementById('deSelectedIndexes').textContent = formatList(getSelectedIndexes(m));
        document.getElementById('deNotSelectedIndexes').textContent = formatList(getNotSelectedIndexes(m, totalFeatures));
    }

    // Ensemble detailed card
    if (results.metrics) {
        const m = results.metrics;
        document.getElementById('ensembleFeaturesCount').textContent = getSelectedCount(m);
        document.getElementById('ensembleAccuracy').textContent = (m.test_accuracy * 100).toFixed(2) + '%';
        document.getElementById('ensemblePrecision').textContent = formatMetricValue(m.precision, 6);
        document.getElementById('ensembleRecall').textContent = formatMetricValue(m.recall, 6);
        document.getElementById('ensembleF1').textContent = formatMetricValue(m.f1_score, 6);
        document.getElementById('ensembleRocAuc').textContent = formatMetricValue(m.roc_auc, 6);
        document.getElementById('ensembleTime').textContent = formatMetricValue(m.time_taken_sec, 4);
        document.getElementById('ensembleConfusionMatrix').textContent = formatConfusionMatrix(m.confusion_matrix);
        document.getElementById('ensembleSelectedIndexes').textContent = formatList(getSelectedIndexes(m));
        document.getElementById('ensembleNotSelectedIndexes').textContent = formatList(getNotSelectedIndexes(m, totalFeatures));
    }

    displayMetricsComparisonTable(results);
}

function formatMetricValue(value, digits = 4) {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value !== 'number' || Number.isNaN(value)) return String(value);
    return value.toFixed(digits);
}

function getSelectedIndexes(metrics) {
    if (!metrics) return [];
    if (Array.isArray(metrics.feature_indexes)) return metrics.feature_indexes;
    return [];
}

function getSelectedCount(metrics) {
    if (!metrics) return 0;
    if (typeof metrics.selected_features === 'number') return metrics.selected_features;
    if (typeof metrics.selected_features_count === 'number') return metrics.selected_features_count;
    return getSelectedIndexes(metrics).length;
}

function getNotSelectedIndexes(metrics, totalCount) {
    if (!totalCount || totalCount <= 0) return [];
    const selected = new Set(getSelectedIndexes(metrics));
    const notSelected = [];
    for (let i = 0; i < totalCount; i++) {
        if (!selected.has(i)) {
            notSelected.push(i);
        }
    }
    return notSelected;
}

function inferTotalFeatureCount(results) {
    const metricBlocks = [
        results?.metrics,
        results?.ga_metrics,
        results?.pso_metrics,
        results?.de_metrics
    ];

    let maxIndex = -1;
    metricBlocks.forEach((block) => {
        const idx = getSelectedIndexes(block);
        idx.forEach((v) => {
            if (typeof v === 'number' && v > maxIndex) {
                maxIndex = v;
            }
        });
    });

    return maxIndex >= 0 ? maxIndex + 1 : 0;
}

function formatList(values) {
    if (!values || values.length === 0) return '[]';
    return `[${values.join(', ')}]`;
}

function formatConfusionMatrix(cm) {
    if (!Array.isArray(cm)) return 'N/A';
    return JSON.stringify(cm);
}

function displayMetricsComparisonTable(results) {
    const container = document.getElementById('metricsComparisonTable') || document.getElementById('detailedMetrics');
    if (!container) return;

    const totalFeatures = allFeatureNames.length || inferTotalFeatureCount(results);
    const columns = [
        { key: 'ga', title: 'GA', metrics: results.ga_metrics },
        { key: 'pso', title: 'PSO', metrics: results.pso_metrics },
        { key: 'de', title: 'DE', metrics: results.de_metrics },
        { key: 'ensemble', title: 'Ensemble', metrics: results.metrics }
    ];

    const rows = [
        {
            label: 'Accuracy',
            value: (m) => formatMetricValue(m?.test_accuracy, 6)
        },
        {
            label: 'Precision',
            value: (m) => formatMetricValue(m?.precision, 6)
        },
        {
            label: 'Recall',
            value: (m) => formatMetricValue(m?.recall, 6)
        },
        {
            label: 'F1 Score',
            value: (m) => formatMetricValue(m?.f1_score, 6)
        },
        {
            label: 'Confusion Matrix',
            value: (m) => `<span class="mono-list">${formatConfusionMatrix(m?.confusion_matrix)}</span>`
        },
        {
            label: 'Selected Features',
            value: (m) => String(getSelectedCount(m))
        },
        {
            label: 'Time Taken (sec)',
            value: (m) => formatMetricValue(m?.time_taken_sec, 4)
        },
        {
            label: 'ROC-AUC',
            value: (m) => formatMetricValue(m?.roc_auc, 6)
        }
    ];

    let tableHtml = `
        <table class="metrics-comparison-table">
            <thead>
                <tr>
                    <th class="metric-col">Metric</th>
                    ${columns.map(c => `<th>${c.title}</th>`).join('')}
                </tr>
            </thead>
            <tbody>
    `;

    rows.forEach(row => {
        tableHtml += `
            <tr>
                <td class="metric-col">${row.label}</td>
                ${columns.map(c => `<td class="comparison-value-cell">${row.value(c.metrics)}</td>`).join('')}
            </tr>
        `;
    });

    tableHtml += `
            </tbody>
        </table>
    `;

    container.innerHTML = tableHtml;
}

function displaySelectedFeatures(features, metrics) {
    const featuresList = document.getElementById('selectedFeaturesList');
    
    if (features.length === 0) {
        featuresList.innerHTML = '<p>No features selected</p>';
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
                backgroundColor: 'rgba(255, 99, 132, 0.15)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 5,
                pointHoverRadius: 7,
                pointBackgroundColor: '#ff6384',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            },
            {
                label: 'PSO Best Fitness',
                data: data.pso,
                borderColor: '#36a2eb',
                backgroundColor: 'rgba(54, 162, 235, 0.15)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 5,
                pointHoverRadius: 7,
                pointBackgroundColor: '#36a2eb',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            },
            {
                label: 'DE Best Fitness',
                data: data.de,
                borderColor: '#4bc0c0',
                backgroundColor: 'rgba(75, 192, 192, 0.15)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 5,
                pointHoverRadius: 7,
                pointBackgroundColor: '#4bc0c0',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            }
        ];
    } else if (type === 'ga') {
        datasets = [{
            label: 'GA Best Fitness',
            data: data.ga,
            borderColor: '#ff6384',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            borderWidth: 3,
            fill: true,
            tension: 0.4,
            pointRadius: 5,
            pointHoverRadius: 7,
            pointBackgroundColor: '#ff6384',
            pointBorderColor: '#fff',
            pointBorderWidth: 2
        }];
    } else if (type === 'pso') {
        datasets = [{
            label: 'PSO Best Fitness',
            data: data.pso,
            borderColor: '#36a2eb',
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            borderWidth: 3,
            fill: true,
            tension: 0.4,
            pointRadius: 5,
            pointHoverRadius: 7,
            pointBackgroundColor: '#36a2eb',
            pointBorderColor: '#fff',
            pointBorderWidth: 2
        }];
    } else if (type === 'de') {
        datasets = [{
            label: 'DE Best Fitness',
            data: data.de,
            borderColor: '#4bc0c0',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderWidth: 3,
            fill: true,
            tension: 0.4,
            pointRadius: 5,
            pointHoverRadius: 7,
            pointBackgroundColor: '#4bc0c0',
            pointBorderColor: '#fff',
            pointBorderWidth: 2
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
            maintainAspectRatio: false,
            devicePixelRatio: getChartPixelRatio(),
            resizeDelay: 100,
            plugins: {
                title: {
                    display: true,
                    text: `${type.toUpperCase()} Convergence Analysis`,
                    color: CHART_TEXT_COLOR,
                    font: { size: 24, weight: 'bold' },
                    padding: { bottom: 24 }
                },
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: CHART_TEXT_COLOR,
                        font: { size: 16, weight: '600' },
                        padding: 20,
                        usePointStyle: true,
                        boxWidth: 14,
                        boxHeight: 14
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleFont: { size: 15, weight: 'bold' },
                    bodyFont: { size: 14 },
                    padding: 12,
                    displayColors: true
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Generation',
                        color: CHART_TEXT_COLOR,
                        font: { size: 17, weight: 'bold' }
                    },
                    ticks: {
                        color: CHART_MUTED_TEXT_COLOR,
                        font: { size: 19, weight: '700' },
                        padding: 10
                    },
                    grid: {
                        drawBorder: true,
                        color: CHART_GRID_COLOR
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Best Fitness',
                        color: CHART_TEXT_COLOR,
                        font: { size: 17, weight: 'bold' }
                    },
                    ticks: {
                        color: CHART_MUTED_TEXT_COLOR,
                        font: { size: 19, weight: '700' },
                        padding: 10,
                        callback: function(value) {
                            return value.toFixed(2);
                        }
                    },
                    grid: {
                        color: CHART_GRID_COLOR
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
    await ensureFeatureNamesLoaded();
    
    try {
        // Fetch complete training results
        const response = await fetch(`${API_URL}/results`);
        const data = await response.json();
        
        // Display ensemble metrics
        displayMetrics(data.metrics);
        
        // Display individual + ensemble detailed metrics
        displayAlgorithmMetrics({
            metrics: data.metrics,
            ga_metrics: data.ga_metrics,
            pso_metrics: data.pso_metrics,
            de_metrics: data.de_metrics
        });
        
        // Display selected features
        if (data.selected_features) {
            displaySelectedFeatures(data.selected_features, data.metrics);
        }
        
        // Display confusion matrix
        if (data.metrics && data.metrics.confusion_matrix) {
            displayConfusionMatrix(data.metrics.confusion_matrix);
        }
        
        // Display convergence charts
        if (data.convergence) {
            convergenceData = data.convergence;
            displayConvergenceCharts(convergenceData);
        }
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

async function ensureFeatureNamesLoaded() {
    if (allFeatureNames.length > 0) return;
    try {
        const response = await fetch(`${API_URL}/features`);
        const data = await response.json();
        if (Array.isArray(data.features)) {
            allFeatureNames = data.features;
        }
    } catch (error) {
        console.error('Error loading feature names:', error);
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
                    backgroundColor: 'rgba(40, 167, 69, 0.8)',
                    borderColor: 'rgba(40, 167, 69, 1)',
                    borderWidth: 2
                },
                {
                    label: 'Class 1 (Positive)',
                    data: [result.voting_proba[1] * 100, result.nn_proba[1] * 100],
                    backgroundColor: 'rgba(220, 53, 69, 0.8)',
                    borderColor: 'rgba(220, 53, 69, 1)',
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            devicePixelRatio: getChartPixelRatio(),
            resizeDelay: 100,
            indexAxis: 'y',
            plugins: {
                title: {
                    display: true,
                    text: 'Model Probability Distribution',
                    color: CHART_TEXT_COLOR,
                    font: { size: 24, weight: 'bold' },
                    padding: { bottom: 24 }
                },
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: CHART_TEXT_COLOR,
                        font: { size: 16, weight: '600' },
                        padding: 20,
                        usePointStyle: true,
                        boxWidth: 14,
                        boxHeight: 14
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleFont: { size: 15, weight: 'bold' },
                    bodyFont: { size: 14 },
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.x.toFixed(1) + '%';
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Probability (%)',
                        color: CHART_TEXT_COLOR,
                        font: { size: 17, weight: 'bold' }
                    },
                    ticks: {
                        color: CHART_MUTED_TEXT_COLOR,
                        font: { size: 19, weight: '700' },
                        padding: 10,
                        callback: function(value) {
                            return value.toFixed(0) + '%';
                        }
                    },
                    grid: {
                        color: CHART_GRID_COLOR
                    }
                },
                y: {
                    ticks: {
                        color: CHART_MUTED_TEXT_COLOR,
                        font: { size: 19, weight: '700' },
                        padding: 10
                    },
                    grid: {
                        display: false
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
    
    let html = '<h4>Batch Prediction Results</h4><table class="batch-results-table">';
    html += '<tr><th>Sample</th><th>Prediction</th><th>Confidence</th></tr>';
    
    predictions.forEach((pred, idx) => {
        const predText = pred.prediction === 1 ? 'Positive' : 'Negative';
        const confidence = (pred.confidence * 100).toFixed(2);
        html += `<tr>
            <td class="batch-number-cell">${idx + 1}</td>
            <td class="batch-number-cell" style="color: ${pred.prediction === 1 ? '#dc3545' : '#28a745'}">${predText}</td>
            <td class="batch-number-cell">${confidence}%</td>
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
