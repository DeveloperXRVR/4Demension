# 4Demension Worker Deployment Guide

## 🔄 Rebuild & Deploy Instructions

Your handler edits **won't automatically update** on RunPod and GitHub. Follow these steps:

### 📋 Prerequisites
1. **Install Docker Desktop** (if not already installed)
   - Download from: https://www.docker.com/products/docker-desktop
   - Start Docker Desktop after installation

2. **Docker Hub Account** (for pushing images)
   - Create account at: https://hub.docker.com
   - Login: `docker login` in terminal

### 🚀 Quick Deploy (Recommended)

Run the deployment script:
```bash
cd /Users/macos/Desktop/Developer/4Demension
./deploy-worker.sh
```

### 🔧 Manual Deploy Steps

#### 1. Build Docker Image
```bash
cd /Users/macos/Desktop/Developer/4Demension/runpod-worker
docker build -t 4demension-worker:latest .
```

#### 2. Tag for Registry
```bash
# Replace with your Docker Hub username
docker tag 4demension-worker:latest your-username/4demension-worker:latest
```

#### 3. Push to Docker Hub
```bash
docker push your-username/4demension-worker:latest
```

#### 4. Update RunPod Endpoint
1. Go to: https://www.runpod.io/console/serverless
2. Find your endpoint: `ika5cdpb25jjr5`
3. Click **Edit** or **Update**
4. Change **Docker Image** to: `your-username/4demension-worker:latest`
5. Click **Save**
6. Wait for workers to restart (2-3 minutes)

#### 5. Commit to GitHub
```bash
cd /Users/macos/Desktop/Developer/4Demension
git add .
git commit -m "Enhanced 4Demension: mesh generation + multi-format export"
git push origin main
```

## 🆕 What's New in This Update

### ✅ Enhanced Features
- **🎯 Increased Density**: 1x-3x point cloud density
- **🔷 Mesh Generation**: Poisson/Ball Pivot/Alpha methods
- **📁 Multi-Format Export**: OBJ, GLB, DAE, STL, PLY
- **🎨 Texture Mapping**: Automatic texture extraction
- **🖥️ Enhanced UI**: Format selection and mesh options

### 🔧 Technical Updates
- **Open3D Integration**: Advanced mesh processing
- **Trimesh Support**: Multi-format conversion
- **Enhanced Handler**: New parameters and processing
- **Updated Dependencies**: Open3D, Trimesh, NetworkX

## 🎯 Testing Your Update

After deployment, test the new features:

1. **Open 4Demension Web App**
   - Double-click the app icon or go to http://localhost:3000

2. **Upload a Video**
   - Select any quality setting

3. **Enable Mesh Generation**
   - Check "Создать 3D mesh (полигональную модель)"
   - Choose mesh method (Poisson recommended)

4. **Select Export Formats**
   - Choose additional formats beyond Splat
   - OBJ for 3D modeling, GLB for web, STL for 3D printing

5. **Process & Download**
   - Wait for completion
   - Download multiple format files

## 🐛 Troubleshooting

### Docker Issues
```bash
# Docker not found
# Install Docker Desktop from docker.com

# Permission denied
# Make sure Docker Desktop is running
# Try: sudo docker build ...
```

### RunPod Issues
- **Endpoint not updating**: Wait 2-3 minutes, then refresh
- **Build errors**: Check Docker logs for dependency issues
- **Memory issues**: Try reducing `max_frames` in webapp

### Format Issues
- **Mesh generation fails**: Try "Ball Pivot" method instead of Poisson
- **Large files**: Reduce quality setting or disable mesh generation
- **Texture missing**: Ensure video has good lighting and contrast

## 📞 Support

If you encounter issues:
1. Check the deployment script output
2. Review RunPod endpoint logs
3. Test with a simple video first
4. Enable mesh generation only after successful basic reconstruction

**Your enhanced 4Demension will be ready after these steps!** 🎉
