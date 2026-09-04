"use client";

import { useState, useRef } from 'react';
import { UploadCloud, FileImage, Image as ImageIcon, CheckCircle, AlertCircle, Percent, Ruler, X, Eye } from 'lucide-react';
import api from '@/lib/api';

export default function ClinicalViewer({ patientId, onScanComplete }: { patientId: string | number, onScanComplete: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [eyeSide, setEyeSide] = useState('left');
  const [isUploading, setIsUploading] = useState(false);
  const [scanResult, setScanResult] = useState<any>(null);
  const [opacity, setOpacity] = useState(50);
  const [error, setError] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (selectedFile: File) => {
    if (!selectedFile.type.startsWith('image/')) {
      setError("Please upload an image file.");
      return;
    }
    setFile(selectedFile);
    setError('');
    setScanResult(null);
    const url = URL.createObjectURL(selectedFile);
    setPreview(url);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleClear = () => {
    setFile(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setScanResult(null);
    setError('');
    if (inputRef.current) inputRef.current.value = '';
  };

  const handleUpload = async () => {
    if (!file) return;
    setIsUploading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('patient_id', patientId.toString());
    formData.append('eye_side', eyeSide);

    try {
      const res = await api.post('/scans/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const scanId = res.data.id;

      // Poll for completion every 2 seconds
      const poll = setInterval(async () => {
        try {
          const statusRes = await api.get(`/scans/${scanId}`);
          if (statusRes.data.status === 'completed') {
            clearInterval(poll);
            setScanResult(statusRes.data);
            setIsUploading(false);
            onScanComplete();
          }
        } catch (err) {
          clearInterval(poll);
          setIsUploading(false);
          setError("Failed to fetch scan status. Check backend connection.");
        }
      }, 2000);

    } catch (err: any) {
      setError(err.response?.data?.detail || "Upload failed. Ensure backend is running.");
      setIsUploading(false);
    }
  };

  const severityLabels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"];
  const severityColors = ["text-green-600 bg-green-50 border-green-200", "text-yellow-600 bg-yellow-50 border-yellow-200", "text-orange-500 bg-orange-50 border-orange-200", "text-red-500 bg-red-50 border-red-200", "text-red-700 bg-red-100 border-red-300"];
  const severityBg = ["from-green-500/10", "from-yellow-500/10", "from-orange-500/10", "from-red-500/10", "from-red-700/20"];

  return (
    <div className="bg-white rounded-3xl p-6 lg:p-8 shadow-sm border border-slate-200">
      <div className="flex items-center justify-between mb-8">
        <div>
           <h3 className="text-xl font-bold text-slate-900 tracking-tight">AI Diagnostic Suite</h3>
           <p className="text-sm font-medium text-slate-500 mt-1">Upload fundus images for automated GradCAM analysis.</p>
        </div>
        <div className="hidden sm:flex items-center gap-3 bg-slate-50 px-4 py-2 rounded-xl border border-slate-100">
          <label className="text-sm font-bold text-slate-600 uppercase tracking-wider">Eye:</label>
          <select
            className="bg-white border-none text-sm font-semibold text-indigo-700 focus:ring-0 rounded-lg py-1 px-2 cursor-pointer outline-none shadow-sm"
            value={eyeSide}
            onChange={e => setEyeSide(e.target.value)}
          >
            <option value="left">Left Eye (OS)</option>
            <option value="right">Right Eye (OD)</option>
          </select>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="mb-6 flex items-center gap-3 bg-red-50 border border-red-200 text-red-700 px-5 py-4 rounded-2xl">
          <AlertCircle className="shrink-0" size={20} />
          <p className="text-sm font-semibold">{error}</p>
          <button onClick={() => setError('')} className="ml-auto text-red-400 hover:text-red-600"><X size={16} /></button>
        </div>
      )}

      {/* Upload Zone */}
      {!scanResult && (
        <div>
          {!file ? (
            <div
              className="border-2 border-dashed border-slate-200 hover:border-indigo-400 rounded-3xl p-16 text-center transition-all duration-300 group cursor-pointer hover:bg-indigo-50/30"
              onDragOver={e => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
            >
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={e => { if (e.target.files && e.target.files[0]) handleFileSelect(e.target.files[0]); }}
              />
              <div className="flex flex-col items-center">
                <div className="w-20 h-20 bg-indigo-50 text-indigo-500 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 group-hover:bg-indigo-100 transition-all shadow-sm border border-indigo-100">
                   <UploadCloud size={36} />
                </div>
                <p className="text-xl font-bold text-slate-800 mb-2 tracking-tight">Drag & drop fundus image here</p>
                <p className="text-sm text-slate-500 font-medium max-w-sm">Accepts PNG, JPG, TIFF — high-resolution images recommended</p>
                <div className="mt-8 bg-white shadow-sm border border-slate-200 text-slate-700 px-6 py-2.5 rounded-xl font-semibold hover:bg-slate-50 transition-colors inline-block">
                  Browse Files
                </div>
              </div>
            </div>
          ) : (
            /* File Selected — Preview + Controls */
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Image Preview */}
              <div className="bg-slate-950 rounded-3xl overflow-hidden relative shadow-2xl border border-slate-800 flex flex-col">
                <div className="h-12 bg-slate-900/80 border-b border-slate-800 flex items-center justify-between px-5 shrink-0">
                  <div className="flex items-center gap-2 text-slate-300 font-mono text-xs">
                    <Eye size={14} className="text-indigo-400" />
                    <span className="truncate max-w-[200px]">{file.name}</span>
                  </div>
                  <button onClick={handleClear} className="text-slate-500 hover:text-white transition-colors p-1">
                    <X size={16} />
                  </button>
                </div>
                <div className="relative flex-1 min-h-[260px] flex items-center justify-center bg-slate-950">
                  {preview && (
                    <img
                      src={preview}
                      alt="Fundus Preview"
                      className="max-w-full max-h-64 object-contain"
                    />
                  )}
                  <div className="absolute bottom-4 left-4 bg-slate-900/90 backdrop-blur text-white px-3 py-1.5 rounded-lg text-xs font-mono font-bold border border-slate-700">
                    {(file.size / 1024 / 1024).toFixed(2)} MB • {eyeSide.toUpperCase()} EYE
                  </div>
                </div>
              </div>

              {/* Upload Panel */}
              <div className="flex flex-col justify-between gap-4">
                <div className="bg-indigo-50 rounded-2xl p-6 border border-indigo-100">
                  <h4 className="font-bold text-indigo-900 mb-1">Ready for Inference</h4>
                  <p className="text-sm text-indigo-700/80 font-medium leading-relaxed">
                    Image queued for <strong>ConvNeXt-V2</strong> analysis. The AI will evaluate DR severity (0–4) and generate a GradCAM activation heatmap.
                  </p>
                  <div className="mt-4 flex items-center gap-3">
                    <label className="text-xs font-bold text-indigo-600 uppercase tracking-wider">Eye Side:</label>
                    <select
                      className="bg-white border border-indigo-200 text-sm font-semibold text-indigo-700 rounded-lg py-1.5 px-3 outline-none cursor-pointer"
                      value={eyeSide}
                      onChange={e => setEyeSide(e.target.value)}
                    >
                      <option value="left">Left Eye (OS)</option>
                      <option value="right">Right Eye (OD)</option>
                    </select>
                  </div>
                </div>

                <button
                  onClick={handleUpload}
                  disabled={isUploading}
                  className="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-4 rounded-2xl font-bold transition-all shadow-lg shadow-indigo-600/30 flex items-center justify-center gap-3 disabled:opacity-70 disabled:cursor-not-allowed relative overflow-hidden"
                >
                  {isUploading ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Analyzing with AI...
                    </>
                  ) : (
                    <>
                      <CheckCircle size={20} />
                      Run AI Diagnostic
                    </>
                  )}
                </button>

                <button
                  onClick={handleClear}
                  className="w-full text-sm text-slate-500 hover:text-red-600 font-semibold py-2 transition-colors"
                >
                  Cancel & Clear
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Results View */}
      {scanResult && (
        <div className="space-y-6 animate-fade-in">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
             {/* Main Viewer */}
             <div className="lg:col-span-2 bg-slate-950 rounded-3xl overflow-hidden relative shadow-2xl border border-slate-800 flex flex-col">

               {/* Viewer Header */}
               <div className="h-14 bg-slate-900/80 backdrop-blur-md border-b border-slate-800 flex items-center justify-between px-6 shrink-0">
                  <div className="flex items-center gap-2 text-slate-300 font-mono text-xs">
                    <ImageIcon size={14} className="text-indigo-400" />
                    <span>Scan_{String(scanResult.id)}</span>
                    <span className="mx-2 opacity-30">|</span>
                    <span className="text-emerald-400">AI Verified</span>
                  </div>
               </div>

               {/* Image Overlay Engine */}
               <div className="relative flex-1 min-h-[300px] flex items-center justify-center bg-slate-950 group">
                 <img
                   src={scanResult.raw_image_s3_url}
                   alt="Raw Fundus"
                   className="absolute max-w-full max-h-[320px] object-contain pointer-events-none"
                 />
                 {scanResult.gradcam_image_s3_url && (
                   <img
                     src={scanResult.gradcam_image_s3_url}
                     alt="GradCAM Heatmap"
                     className="absolute max-w-full max-h-[320px] object-contain pointer-events-none transition-opacity mix-blend-screen filter contrast-125 saturate-150"
                     style={{ opacity: opacity / 100 }}
                   />
                 )}
                 {/* High Activation badge */}
                 <div className="absolute bottom-6 right-6 bg-slate-900/80 backdrop-blur text-white px-3 py-1.5 rounded-lg text-xs font-mono font-bold flex items-center gap-2 border border-slate-700 opacity-0 group-hover:opacity-100 transition-opacity">
                    <div className="w-2 h-2 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)]"></div>
                    High Activation Area
                 </div>
               </div>

               {/* Opacity Slider */}
               <div className="bg-slate-900 border-t border-slate-800 p-5 flex items-center gap-6 shrink-0">
                 <span className="text-sm font-semibold text-slate-400 whitespace-nowrap hidden sm:block">GradCAM Overlay</span>
                 <input
                   type="range"
                   min="0" max="100"
                   value={opacity}
                   onChange={(e) => setOpacity(Number(e.target.value))}
                   className="w-full h-2 bg-slate-800 rounded-full appearance-none cursor-pointer accent-indigo-500 outline-none"
                 />
                 <span className="text-sm font-mono font-bold text-indigo-400 w-12 text-right">{opacity}%</span>
               </div>
            </div>

             {/* Analytics Side Panel */}
             <div className="space-y-5 flex flex-col">
               {/* Severity Card */}
               <div className={`bg-gradient-to-br ${severityBg[scanResult.dr_prediction_level ?? 0]} to-white border text-center rounded-3xl p-6 shadow-sm flex flex-col items-center justify-center relative overflow-hidden group ${severityColors[scanResult.dr_prediction_level ?? 0]}`}>
                  <div className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">AI Prediction</div>

                  <div className={`px-6 py-2.5 rounded-2xl font-black text-2xl tracking-tight mb-6 shadow-sm border ${severityColors[scanResult.dr_prediction_level ?? 0]}`}>
                    {severityLabels[scanResult.dr_prediction_level ?? 0]}
                  </div>

                  <div className="w-full px-2">
                     <div className="flex justify-between text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2">
                       <span>No DR</span>
                       <span>Proliferative</span>
                     </div>
                     <div className="h-3 w-full bg-slate-100 rounded-full overflow-hidden relative">
                       <div
                         style={{ width: `${((scanResult.dr_prediction_level ?? 0) / 4) * 100}%` }}
                         className="absolute top-0 left-0 h-full rounded-full bg-gradient-to-r from-green-500 via-yellow-400 to-red-600 transition-all duration-1000"
                       />
                     </div>
                     <div className="mt-3 flex items-center justify-center gap-1.5 text-xs text-slate-500 font-semibold">
                       <Percent size={13} className="text-indigo-500" />
                       Confidence Score: {(scanResult.regression_score ?? 0).toFixed(3)}
                     </div>
                  </div>
               </div>

               {/* Disclaimer */}
               <div className="bg-amber-50 rounded-2xl p-5 border border-amber-200 flex-1 flex flex-col justify-center shadow-sm relative overflow-hidden">
                 <div className="absolute -right-4 -top-4 text-amber-500/10"><AlertCircle size={80}/></div>
                 <div className="relative z-10 flex items-start gap-4">
                   <AlertCircle className="text-amber-600 shrink-0 mt-0.5" size={22} />
                   <div>
                     <h4 className="font-bold text-amber-900 mb-1">Clinical Disclaimer</h4>
                     <p className="text-xs text-amber-800/80 font-medium leading-relaxed">
                       AI diagnostic provides CNN activation heatmaps as an <strong className="text-amber-900">assistive tool only</strong>. Final determination must be made by a verified ophthalmologist.
                     </p>
                   </div>
                 </div>
               </div>

               <button
                 onClick={handleClear}
                 className="w-full bg-slate-900 hover:bg-slate-800 text-white font-bold py-4 rounded-xl shadow-lg transition-colors flex items-center justify-center gap-2 shrink-0 group"
               >
                 <UploadCloud size={18} className="group-hover:-translate-y-1 transition-transform" /> Process New Scan
               </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
