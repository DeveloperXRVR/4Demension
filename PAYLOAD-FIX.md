# RunPod 10MB Payload Limit Fix

## 🚨 **Problem: HTTP 400 - Exceeded 10MiB Body Size**

The enhanced density is generating too much data for RunPod's 10MB limit:
- Even with conservative limits, the video data + parameters exceed 10MB
- RunPod serverless has a strict 10MiB request body limit

## ✅ **Solution: Optimize Payload Size**

### **Option 1: Reduce Video Quality (Quick Fix)**
- Compress video more aggressively
- Reduce max_frames parameter
- Lower resolution before upload

### **Option 2: Split Processing (Recommended)**
- Step 1: Upload video separately
- Step 2: Process with density parameters
- Step 3: Download results

### **Option 3: Use RunPod Volume Storage**
- Upload video to RunPod volume
- Pass only parameters in request
- Reference video file from volume

## 🔧 **Immediate Fix: Reduce Payload Size**

Let me implement Option 1 first to get it working:
