# Script para subir el proyecto a GitHub
# Repositorio: git@github.com:AndresMlz/trading_XAI.git

Write-Host "🚀 Script para subir proyecto a GitHub" -ForegroundColor Cyan
Write-Host "Repositorio: AndresMlz/trading_XAI" -ForegroundColor Cyan
Write-Host ""

# Verificar si Git está instalado
try {
    $gitVersion = git --version 2>&1
    Write-Host "✅ Git encontrado: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Git no está instalado" -ForegroundColor Red
    Write-Host "📥 Instala Git desde: https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "   O ejecuta: choco install git" -ForegroundColor Yellow
    exit 1
}

# Verificar si ya existe un repositorio
$isGitRepo = Test-Path .git
if (-not $isGitRepo) {
    Write-Host "📦 Inicializando repositorio Git..." -ForegroundColor Cyan
    git init
    Write-Host "✅ Repositorio inicializado" -ForegroundColor Green
} else {
    Write-Host "✅ Repositorio Git ya existe" -ForegroundColor Green
}

# Verificar configuración de usuario
$userName = git config user.name
$userEmail = git config user.email

if (-not $userName -or -not $userEmail) {
    Write-Host "`n⚠️  Configuración de usuario no encontrada" -ForegroundColor Yellow
    $userName = Read-Host "Ingresa tu nombre"
    $userEmail = Read-Host "Ingresa tu email"
    
    if ($userName) { git config user.name $userName }
    if ($userEmail) { git config user.email $userEmail }
    
    Write-Host "✅ Usuario configurado" -ForegroundColor Green
} else {
    Write-Host "✅ Usuario configurado: $userName <$userEmail>" -ForegroundColor Green
}

# Mostrar estado
Write-Host "`n📊 Estado actual del repositorio:" -ForegroundColor Cyan
git status --short

# Preguntar si desea agregar archivos
Write-Host "`n❓ ¿Deseas agregar todos los archivos al staging? (s/n)" -ForegroundColor Cyan
$addFiles = Read-Host

if ($addFiles -eq "s" -or $addFiles -eq "S") {
    Write-Host "`n➕ Agregando archivos..." -ForegroundColor Cyan
    git add .
    
    Write-Host "📋 Archivos en staging:" -ForegroundColor Cyan
    git status --short
    
    # Verificar si hay cambios para commitear
    $hasChanges = git diff --cached --quiet
    if (-not $hasChanges) {
        Write-Host "`n⚠️  No hay cambios nuevos para commitear" -ForegroundColor Yellow
    } else {
        Write-Host "`n❓ ¿Deseas hacer commit? (s/n)" -ForegroundColor Cyan
        $doCommit = Read-Host
        
        if ($doCommit -eq "s" -or $doCommit -eq "S") {
            $commitMsg = Read-Host "Mensaje del commit (o presiona Enter para usar el mensaje por defecto)"
            if (-not $commitMsg) {
                $commitMsg = "Initial commit: Proyecto de predicción SPY con RAG chatbot integrado"
            }
            
            git commit -m $commitMsg
            Write-Host "✅ Commit creado" -ForegroundColor Green
        }
    }
}

# Verificar si existe el remote
$remoteExists = git remote | Select-String -Pattern "origin"
if ($remoteExists) {
    Write-Host "`n⚠️  Ya existe un remote 'origin'" -ForegroundColor Yellow
    git remote -v
    $changeRemote = Read-Host "¿Deseas cambiarlo? (s/n)"
    
    if ($changeRemote -eq "s" -or $changeRemote -eq "S") {
        git remote remove origin
        Write-Host "✅ Remote 'origin' eliminado" -ForegroundColor Green
    } else {
        Write-Host "✅ Usando remote existente" -ForegroundColor Green
        $skipRemote = $true
    }
}

# Agregar remote si no existe o se eliminó
if (-not $skipRemote) {
    Write-Host "`n🌐 Configurando repositorio remoto..." -ForegroundColor Cyan
    Write-Host "Opciones:" -ForegroundColor Yellow
    Write-Host "  1. SSH: git@github.com:AndresMlz/trading_XAI.git" -ForegroundColor Gray
    Write-Host "  2. HTTPS: https://github.com/AndresMlz/trading_XAI.git" -ForegroundColor Gray
    
    $remoteType = Read-Host "Selecciona opción (1 o 2)"
    
    if ($remoteType -eq "1") {
        $remoteUrl = "git@github.com:AndresMlz/trading_XAI.git"
        Write-Host "⚠️  Asegúrate de tener SSH keys configuradas en GitHub" -ForegroundColor Yellow
    } else {
        $remoteUrl = "https://github.com/AndresMlz/trading_XAI.git"
    }
    
    git remote add origin $remoteUrl
    Write-Host "✅ Remote agregado: $remoteUrl" -ForegroundColor Green
    
    # Verificar conexión
    Write-Host "`n🔍 Verificando conexión..." -ForegroundColor Cyan
    git remote -v
}

# Preguntar si desea subir
Write-Host "`n❓ ¿Deseas subir el código a GitHub? (s/n)" -ForegroundColor Cyan
$doPush = Read-Host

if ($doPush -eq "s" -or $doPush -eq "S") {
    Write-Host "`n📤 Subiendo código a GitHub..." -ForegroundColor Cyan
    
    # Asegurar que la rama se llame 'main'
    $currentBranch = git branch --show-current
    if ($currentBranch -ne "main") {
        Write-Host "📝 Renombrando rama a 'main'..." -ForegroundColor Cyan
        git branch -M main
    }
    
    try {
        git push -u origin main
        Write-Host "`n✅ ¡Código subido exitosamente!" -ForegroundColor Green
        Write-Host "🌐 Ve a: https://github.com/AndresMlz/trading_XAI" -ForegroundColor Cyan
    } catch {
        Write-Host "`n❌ Error al subir el código" -ForegroundColor Red
        Write-Host "Posibles causas:" -ForegroundColor Yellow
        Write-Host "  - Si usas SSH: verifica que tu SSH key esté en GitHub" -ForegroundColor Gray
        Write-Host "  - Si usas HTTPS: usa un Personal Access Token como contraseña" -ForegroundColor Gray
        Write-Host "  - El repositorio remoto puede tener contenido diferente" -ForegroundColor Gray
        Write-Host "`nIntenta manualmente:" -ForegroundColor Yellow
        Write-Host "  git push -u origin main" -ForegroundColor Gray
    }
} else {
    Write-Host "`n📋 Para subir más tarde, ejecuta:" -ForegroundColor Cyan
    Write-Host "  git push -u origin main" -ForegroundColor Gray
}

Write-Host "`n✅ Proceso completado!" -ForegroundColor Green

