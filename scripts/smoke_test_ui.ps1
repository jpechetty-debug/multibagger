# scripts/smoke_test_ui.ps1
# Smoke test for the Streamlit UI to ensure it boots without crashing

Write-Host "Starting Streamlit UI in background..."
Start-Process -FilePath "python" -ArgumentList "-m streamlit run app/streamlit_app.py --server.port 8501 --server.headless true" -NoNewWindow -PassThru -OutVariable stProc | Out-Null
$stId = $stProc.Id

Write-Host "Waiting for Streamlit to initialize (15 seconds)..."
Start-Sleep -Seconds 15

Write-Host "Checking health endpoint..."
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8501/_stcore/health" -UseBasicParsing -ErrorAction Stop
    $statusCode = $response.StatusCode
} catch {
    $statusCode = $_.Exception.Response.StatusCode.value__
    if ($null -eq $statusCode) {
        $statusCode = 0
    }
}

if ($statusCode -eq 200 -or $statusCode -eq 404) {
    if ($statusCode -eq 404) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8501/healthz" -UseBasicParsing -ErrorAction Stop
            $statusCode = $response.StatusCode
        } catch {
            $statusCode = $_.Exception.Response.StatusCode.value__
        }
    }
}

if ($statusCode -eq 200 -or $statusCode -eq 302) {
    Write-Host "SUCCESS: UI is running on port 8501." -ForegroundColor Green
    Stop-Process -Id $stId -Force
    exit 0
} else {
    Write-Host "ERROR: UI failed to respond correctly. HTTP Code: $statusCode" -ForegroundColor Red
    Stop-Process -Id $stId -Force
    exit 1
}
