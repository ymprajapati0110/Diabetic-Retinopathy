# Retinal Diagnostic Portal: Deep Learning-based Diabetic Retinopathy Detection

An advanced clinical screening suite for detecting Diabetic Retinopathy (DR) using a ConvNeXt-V2 deep learning architecture. The system processes fundus photographs, crops boundaries, applies contrast adjustment (CLAHE), predicts DR severity, and highlights pathology regions using Grad-CAM attention visualization.

---

## 🌟 Demo & Showcase
The frontend of this clinical portal is optimized for serverless demonstration and is hosted on Vercel:

🔗 **[Live Showcase Link (Vercel) Placeholder]** *(Paste your Vercel deployment link here)*

### 🔑 Test User Credentials
To log in and test the system on Vercel (or locally), use the following authorized credentials:
* **Clinical Identifier (Email)**: `test@example.com` (or any valid email address)
* **Access Key (Password)**: `1234`

*Note: In the live Vercel demo, uploading any fundus image will simulate the AI inference process and return a diagnostic result of **Proliferative DR (Level 4, Score 4.01)** with the custom Grad-CAM lesion attention map overlay.*

---

## 🛠️ Architecture

* **Frontend**: Next.js (App Router), React, TailwindCSS, Lucide Icons, and Axios.
* **Backend**: FastAPI, PyTorch, Albumentations, Timm (ConvNeXt-V2 Large), SQLite / SQLAlchemy.

---

## 🚀 How to Run Locally

### 1. Backend Setup
1. Open a terminal and navigate to the `backend` folder:
   ```bash
   cd backend
   ```
2. Recreate the Python virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   * **Windows (PowerShell)**: `.\venv\Scripts\Activate.ps1`
   * **Linux/macOS**: `source venv/bin/activate`
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Place the model weights (`convnextv2_large_epoch_25_ema.pth`) in the project root directory.
6. Start the backend:
   ```bash
   python -m uvicorn main:app --reload --port 8000
   ```

### 2. Frontend Setup
1. Open a terminal and navigate to the `frontend` folder:
   ```bash
   cd frontend
   ```
2. Install the required Node packages:
   ```bash
   npm install
   ```
3. Start the Next.js development server:
   ```bash
   npm run dev
   ```

### 3. Launching with Batch Script
If you are on Windows, you can double-click the `start_app.bat` file in the root directory to automatically launch both servers once you've reinstalled dependencies:
```bash
.\start_app.bat
```

---

## ☁️ Deploying the Frontend to Vercel

To deploy this project's frontend to Vercel:
1. Push this project to your GitHub repository.
2. Sign in to your [Vercel account](https://vercel.com/) and click **Add New** -> **Project**.
3. Import your GitHub repository.
4. Set the **Root Directory** option to `frontend`.
5. Keep other settings default and click **Deploy**.
6. Vercel will build and serve your Next.js app on a public URL.

---

## 🛡️ License and Disclaimer
This portal is developed as a clinical diagnostic research tool. All uploaded data is processed locally.
