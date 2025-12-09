# 🚀 Guía para Subir el Proyecto a GitHub

## Repositorio: `git@github.com:AndresMlz/trading_XAI.git`

---

## 📋 Paso 1: Instalar Git (si no está instalado)

### Opción A: Descarga directa
1. Ve a: https://git-scm.com/download/win
2. Descarga e instala Git para Windows
3. **IMPORTANTE**: Durante la instalación, selecciona "Git from the command line and also from 3rd-party software"
4. Reinicia PowerShell/CMD después de instalar

### Opción B: Con Chocolatey (si lo tienes)
```powershell
choco install git
```

### Verificar instalación:
```powershell
git --version
```

---

## 📋 Paso 2: Configurar Git (solo la primera vez)

Abre PowerShell en la carpeta del proyecto y ejecuta:

```powershell
# Configurar tu nombre y email
git config --global user.name "Andres Matallana"
git config --global user.email "tu.email@ejemplo.com"

# Verificar configuración
git config --list
```

---

## 📋 Paso 3: Inicializar el Repositorio Local

```powershell
# Asegúrate de estar en la carpeta del proyecto
cd C:\Users\felip\Downloads\proyecto_exe

# Inicializar repositorio Git
git init

# Ver el estado (debería mostrar muchos archivos sin rastrear)
git status
```

---

## 📋 Paso 4: Agregar Archivos al Repositorio

```powershell
# Agregar todos los archivos (excepto los del .gitignore)
git add .

# Verificar qué se va a commitear
git status
```

**⚠️ IMPORTANTE**: Verifica que NO se estén agregando:
- `config/token.json`
- `config/credentials.json`
- Archivos con credenciales

Si ves alguno de estos archivos, elimínalos del staging:
```powershell
git reset HEAD config/token.json
```

---

## 📋 Paso 5: Hacer el Commit Inicial

```powershell
# Hacer el commit inicial
git commit -m "Initial commit: Proyecto de predicción SPY con RAG chatbot integrado

- Stacking de modelos para predicción del SPY
- Interfaz Streamlit con dos pantallas
- Integración de RAG chatbot con GCP
- Modelos: CNN, Transformer, Reconocimiento de Patrones
- Sistema de interpretabilidad con Gemini"
```

---

## 📋 Paso 6: Configurar SSH o usar HTTPS

### Opción A: Usar SSH (requiere configuración previa)

**Si ya tienes SSH keys configuradas en GitHub:**
```powershell
# Agregar el repositorio remoto
git remote add origin git@github.com:AndresMlz/trading_XAI.git

# Verificar que se agregó correctamente
git remote -v
```

**Si NO tienes SSH keys configuradas**, sigue estos pasos:

1. **Generar SSH key** (si no tienes una):
   ```powershell
   ssh-keygen -t ed25519 -C "tu.email@ejemplo.com"
   # Presiona Enter para aceptar la ubicación por defecto
   # Opcional: agrega una contraseña para mayor seguridad
   ```

2. **Copiar la clave pública**:
   ```powershell
   cat ~/.ssh/id_ed25519.pub
   # O en Windows:
   type C:\Users\felip\.ssh\id_ed25519.pub
   ```

3. **Agregar la clave a GitHub**:
   - Ve a GitHub.com → Settings → SSH and GPG keys
   - Click en "New SSH key"
   - Pega el contenido de `id_ed25519.pub`
   - Guarda

4. **Probar la conexión**:
   ```powershell
   ssh -T git@github.com
   ```

### Opción B: Usar HTTPS (más simple, no requiere SSH)

```powershell
# Agregar el repositorio remoto con HTTPS
git remote add origin https://github.com/AndresMlz/trading_XAI.git

# Verificar que se agregó correctamente
git remote -v
```

**Ventaja**: No necesitas configurar SSH keys, pero GitHub pedirá autenticación.

---

## 📋 Paso 7: Subir el Código a GitHub

```powershell
# Renombrar la rama principal a 'main' (si es necesario)
git branch -M main

# Subir el código
git push -u origin main
```

**Si usas HTTPS y te pide credenciales:**
- Usuario: tu nombre de usuario de GitHub
- Contraseña: usa un **Personal Access Token** (no tu contraseña normal)
  - Cómo crear un token: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token
  - Selecciona los scopes: `repo` (todos los permisos de repositorio)

**Si usas SSH y hay problemas de conexión:**
```powershell
# Probar conexión SSH
ssh -T git@github.com

# Si falla, verifica que tu clave esté en GitHub
```

---

## 📋 Paso 8: Verificar que se Subió Correctamente

1. Ve a: https://github.com/AndresMlz/trading_XAI
2. Deberías ver todos tus archivos
3. Verifica que NO estén subidos archivos sensibles como `token.json`

---

## 🔄 Para Futuros Cambios

Una vez configurado, para subir cambios futuros:

```powershell
# Ver qué cambió
git status

# Agregar cambios
git add .

# O agregar archivos específicos
git add interfaz_grafica/front_streamlit2.py

# Hacer commit
git commit -m "Descripción de los cambios"

# Subir cambios
git push
```

---

## ⚠️ Solución de Problemas

### Error: "remote origin already exists"
```powershell
# Eliminar el remote existente
git remote remove origin

# Agregar de nuevo
git remote add origin git@github.com:AndresMlz/trading_XAI.git
```

### Error: "Permission denied (publickey)"
- Verifica que tu SSH key esté en GitHub
- Prueba la conexión: `ssh -T git@github.com`
- O usa HTTPS en su lugar

### Error: "failed to push some refs"
```powershell
# Si el repositorio remoto tiene contenido, primero haz pull
git pull origin main --allow-unrelated-histories

# Luego intenta push de nuevo
git push -u origin main
```

### Error: "authentication failed" (HTTPS)
- Usa un Personal Access Token en lugar de tu contraseña
- O configura Git Credential Manager

---

## 📝 Resumen de Comandos (Copia y Pega)

```powershell
# 1. Inicializar
git init

# 2. Configurar (solo primera vez)
git config --global user.name "Andres Matallana"
git config --global user.email "tu.email@ejemplo.com"

# 3. Agregar archivos
git add .

# 4. Commit inicial
git commit -m "Initial commit: Proyecto de predicción SPY con RAG chatbot integrado"

# 5. Agregar remote (SSH)
git remote add origin git@github.com:AndresMlz/trading_XAI.git

# O usar HTTPS
# git remote add origin https://github.com/AndresMlz/trading_XAI.git

# 6. Subir
git branch -M main
git push -u origin main
```

---

## 🔐 Seguridad

**NUNCA subas:**
- ❌ `config/token.json`
- ❌ `config/credentials.json`
- ❌ API keys en el código
- ❌ Cualquier archivo con credenciales

El `.gitignore` ya está configurado para ignorar estos archivos, pero siempre verifica con `git status` antes de hacer commit.

