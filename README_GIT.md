# Guía de Configuración de Git para el Proyecto

## 📋 Requisitos Previos

### 1. Instalar Git (si no está instalado)

**Windows:**
- Descarga Git desde: https://git-scm.com/download/win
- O instala desde el instalador de Windows
- Verifica la instalación ejecutando: `git --version`

**Alternativa rápida (con Chocolatey):**
```powershell
choco install git
```

## 🚀 Configuración Inicial del Repositorio

### Paso 1: Inicializar el repositorio Git

Abre PowerShell o CMD en la carpeta del proyecto y ejecuta:

```powershell
# Inicializar repositorio
git init

# Configurar tu información (reemplaza con tus datos)
git config user.name "Tu Nombre"
git config user.email "tu.email@ejemplo.com"

# O configurar globalmente para todos los repositorios
git config --global user.name "Tu Nombre"
git config --global user.email "tu.email@ejemplo.com"
```

### Paso 2: Verificar archivos a agregar

```powershell
# Ver qué archivos se agregarán
git status

# Ver archivos que serán ignorados (según .gitignore)
git status --ignored
```

### Paso 3: Agregar archivos al staging

```powershell
# Agregar todos los archivos (excepto los del .gitignore)
git add .

# O agregar archivos específicos
git add interfaz_grafica/
git add interpretabilidad_gemini/
git add config/
# etc.
```

### Paso 4: Hacer el commit inicial

```powershell
# Commit inicial
git commit -m "Initial commit: Proyecto de predicción SPY con RAG chatbot integrado"

# O con un mensaje más descriptivo
git commit -m "Initial commit

- Stacking de modelos para predicción del SPY
- Interfaz Streamlit con dos pantallas
- Integración de RAG chatbot con GCP
- Modelos: CNN, Transformer, Reconocimiento de Patrones
- Sistema de interpretabilidad con Gemini"
```

## 📦 Configurar Repositorio Remoto (Opcional)

### Opción 1: GitHub

1. Crea un nuevo repositorio en GitHub (sin inicializar con README)
2. Conecta tu repositorio local:

```powershell
# Agregar el repositorio remoto (reemplaza con tu URL)
git remote add origin https://github.com/tu-usuario/tu-repositorio.git

# Verificar que se agregó correctamente
git remote -v

# Subir el código
git branch -M main
git push -u origin main
```

### Opción 2: GitLab

```powershell
git remote add origin https://gitlab.com/tu-usuario/tu-repositorio.git
git branch -M main
git push -u origin main
```

### Opción 3: Bitbucket

```powershell
git remote add origin https://bitbucket.org/tu-usuario/tu-repositorio.git
git branch -M main
git push -u origin main
```

## 🔄 Comandos Git Útiles

### Ver el estado del repositorio
```powershell
git status
```

### Ver el historial de commits
```powershell
git log
git log --oneline  # Versión compacta
git log --graph --oneline --all  # Con gráfico
```

### Hacer cambios y commitear
```powershell
# 1. Ver qué cambió
git status
git diff

# 2. Agregar cambios
git add .

# 3. Hacer commit
git commit -m "Descripción de los cambios"

# 4. Subir cambios (si hay remoto configurado)
git push
```

### Crear una rama para nuevas características
```powershell
# Crear y cambiar a nueva rama
git checkout -b feature/nombre-caracteristica

# O con el nuevo comando
git switch -c feature/nombre-caracteristica

# Trabajar en la rama, hacer commits, luego:
git push -u origin feature/nombre-caracteristica
```

### Ver diferencias antes de commitear
```powershell
git diff                    # Ver cambios no staged
git diff --staged           # Ver cambios staged
git diff HEAD               # Ver todos los cambios
```

## ⚠️ Archivos Importantes a NO Subir

El archivo `.gitignore` ya está configurado para ignorar:

- ✅ Credenciales (`config/token.json`, `config/credentials.json`)
- ✅ Entornos virtuales (`venv/`, `venv_bot/`)
- ✅ Archivos compilados (`__pycache__/`, `*.pyc`)
- ✅ Archivos del sistema (`.DS_Store`, `Thumbs.db`)

**IMPORTANTE:** Antes de hacer commit, verifica que:
- ❌ No hay credenciales de API en el código
- ❌ No hay tokens de acceso en archivos de configuración
- ❌ No hay información sensible en los commits

## 📝 Estructura del Proyecto

```
proyecto_exe/
├── config/              # Configuración y constantes
├── data_alpaca/         # Descarga de datos
├── enriquecimiento_datos/  # Procesamiento de datos
├── interfaz_grafica/    # Frontend Streamlit
├── interpretabilidad_gemini/  # RAG y explicaciones
├── modelos/             # Modelos ML (CNN, Transformer, etc.)
├── archivos_modelos/   # Modelos entrenados (puede estar en .gitignore)
├── outputs/             # Resultados de predicciones
└── inputs/              # Datos de entrada
```

## 🔐 Seguridad

**NUNCA subas al repositorio:**
- API keys
- Tokens de autenticación
- Credenciales de Google Cloud
- Contraseñas
- Archivos `.json` con información sensible

Si accidentalmente subiste información sensible:
1. Elimínala del historial: `git filter-branch` o `git filter-repo`
2. Cambia las credenciales comprometidas
3. Agrega los archivos al `.gitignore`

## 📚 Recursos Adicionales

- [Documentación oficial de Git](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)

