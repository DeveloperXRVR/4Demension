# Quick Docker Hub Setup Guide

## 🚀 **Option 1: Create Docker Hub Account (5 Minutes)**

### **Step 1: Sign Up**
1. Go to: https://hub.docker.com/signup
2. Fill out the form:
   - Username: `yourname4d` (or any available name)
   - Email: your email
   - Password: create strong password
3. Check your email for verification
4. Click the verification link

### **Step 2: Create Access Token**
1. Sign in to Docker Hub
2. Go to: https://hub.docker.com/settings/security
3. Click **"New Access Token"**
4. Fill in:
   - Token name: `4demension-deploy`
   - Description: `For GitHub Actions`
   - Permissions: ✅ Read, Write, Delete
5. Click **"Generate"**
6. **IMPORTANT**: Copy the token immediately (you won't see it again!)

### **Step 3: Add to GitHub**
1. Go to: https://github.com/DeveloperXRVR/4Demension/settings/secrets
2. Click **"New repository secret"**
3. First secret:
   - Name: `DOCKER_USERNAME`
   - Value: your Docker Hub username (e.g., `yourname4d`)
4. Second secret:
   - Name: `DOCKER_PASSWORD`
   - Value: the access token you copied
5. Click **"Add secret"** for each

## 🎯 **Option 2: Use Public Repository (No Account Needed)**

If you don't want to create an account, I can push to a public repository for you.

Just let me know and I'll:
1. Build the image myself
2. Push to a public Docker Hub repository
3. Give you the image name to use in RunPod

## 📋 **What You'll Have**

After setup, you'll get:
- **Username**: Something like `yourname4d`
- **Password/Token**: Long random string like `dckr_pat_xxxxx...`
- **Repository**: `yourname4d/4demension-worker:latest`

## 🔧 **Test Your Setup**

After adding secrets, test by:
1. Going to: https://github.com/DeveloperXRVR/4Demension/actions
2. Click **"Deploy to RunPod"** workflow
3. Click **"Run workflow"** → **"Run workflow"**
4. Watch it build your image!

## 🚨 **Important Notes**

- **Use Access Token**: More secure than password
- **Copy Token Immediately**: You can't see it again
- **Keep Secrets Private**: Never share your tokens
- **Free Tier**: Docker Hub has generous free limits

## 🎉 **Next Steps**

Once you have the credentials:
1. Add them to GitHub secrets
2. The workflow will automatically build
3. Update RunPod with the new image name
4. Test your new 4Demension features!

**Choose Option 1 for full control, or Option 2 for quick setup!** 🚀
