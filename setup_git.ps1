# Script para inicializar el repositorio Git
# Ejecutar desde PowerShell: .\setup_git.ps1

Write-Host "🚀 Configurando repositorio Git..." -ForegroundColor Cyan

# Verificar si Git está instalado
try {
    $gitVersion = git --version
    Write-Host "✅ Git encontrado: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Git no está instalado o no está en el PATH" -ForegroundColor Red
    Write-Host "📥 Por favor instala Git desde: https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "   O ejecuta: choco install git (si tienes Chocolatey)" -ForegroundColor Yellow
    exit 1
}

# Verificar si ya existe un repositorio Git
if (Test-Path .git) {
    Write-Host "⚠️  Ya existe un repositorio Git en este directorio" -ForegroundColor Yellow
    $continue = Read-Host "¿Deseas continuar de todos modos? (s/n)"
    if ($continue -ne "s" -and $continue -ne "S") {
        exit 0
    }
}

# Inicializar repositorio
Write-Host "`n📦 Inicializando repositorio Git..." -ForegroundColor Cyan
git init

# Verificar configuración de usuario
Write-Host "`n👤 Verificando configuración de usuario..." -ForegroundColor Cyan
$userName = git config user.name
$userEmail = git config user.email

if (-not $userName -or -not $userEmail) {
    Write-Host "⚠️  No se encontró configuración de usuario" -ForegroundColor Yellow
    Write-Host "   Configurando con valores por defecto..." -ForegroundColor Yellow
    
    $defaultName = Read-Host "Ingresa tu nombre (o presiona Enter para omitir)"
    $defaultEmail = Read-Host "Ingresa tu email (o presiona Enter para omitir)"
    
    if ($defaultName) {
        git config user.name $defaultName
        Write-Host "✅ Nombre configurado: $defaultName" -ForegroundColor Green
    }
    
    if ($defaultEmail) {
        git config user.email $defaultEmail
        Write-Host "✅ Email configurado: $defaultEmail" -ForegroundColor Green
    }
} else {
    Write-Host "✅ Usuario configurado: $userName <$userEmail>" -ForegroundColor Green
}

# Mostrar estado
Write-Host "`n📊 Estado del repositorio:" -ForegroundColor Cyan
git status

# Preguntar si desea hacer el commit inicial
Write-Host "`n💾 ¿Deseas hacer el commit inicial ahora? (s/n)" -ForegroundColor Cyan
$doCommit = Read-Host

if ($doCommit -eq "s" -or $doCommit -eq "S") {
    Write-Host "`n➕ Agregando archivos al staging..." -ForegroundColor Cyan
    git add .
    
    Write-Host "📝 Creando commit inicial..." -ForegroundColor Cyan
    $commitMessage = @"
Initial commit: Proyecto de predicción SPY con RAG chatbot integrado

- Stacking de modelos para predicción del SPY
- Interfaz Streamlit con dos pantallas
- Integración de RAG chatbot con GCP
- Modelos: CNN, Transformer, Reconocimiento de Patrones
- Sistema de interpretabilidad con Gemini
"@
    
    git commit -m $commitMessage
    
    Write-Host "✅ Commit inicial creado exitosamente!" -ForegroundColor Green
    Write-Host "`n📋 Próximos pasos:" -ForegroundColor Cyan
    Write-Host "   1. Revisa el archivo README_GIT.md para más información" -ForegroundColor White
    Write-Host "   2. Si deseas subir a un repositorio remoto:" -ForegroundColor White
    Write-Host "      git remote add origin <URL_DEL_REPOSITORIO>" -ForegroundColor Gray
    Write-Host "      git push -u origin main" -ForegroundColor Gray
} else {
    Write-Host "`n📋 Para hacer commit más tarde, ejecuta:" -ForegroundColor Cyan
    Write-Host "   git add ." -ForegroundColor Gray
    Write-Host "   git commit -m 'Tu mensaje de commit'" -ForegroundColor Gray
}

Write-Host "`n✅ Configuración completada!" -ForegroundColor Green

