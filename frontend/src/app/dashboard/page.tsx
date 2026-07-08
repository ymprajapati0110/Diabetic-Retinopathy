"use client";

import { useState, useEffect, useRef } from 'react';
import { UploadCloud, FileImage, CheckCircle, AlertCircle, Eye, History, Trash2, Calendar, ClipboardCheck } from 'lucide-react';
import api from '@/lib/api';

export default function DashboardPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [scanResult, setScanResult] = useState<any>(null);
  const [opacity, setOpacity] = useState(50);
  const [error, setError] = useState('');
  const [scans, setScans] = useState<any[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const inputRef = useRef<HTMLInputElement>(null);

  // Fetch scan history from the backend
  const fetchHistory = async () => {
    try {
      const res = await api.get('/scans/');
      setScans(res.data);
    } catch (err) {
      console.error("Failed to fetch scan history", err);
    } finally {
      setLoadingHistory(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleFileSelect = (selectedFile: File) => {
    if (!selectedFile.type.startsWith('image/')) {
      setError("Please upload a valid fundus image file.");
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

  const handleDeleteScan = async (scanId: number, e: React.MouseEvent) => {
    e.stopPropagation(); // Avoid loading the scan in the viewer
    if (!confirm("Are you sure you want to permanently delete this scan?")) return;
    
    try {
      await api.delete(`/scans/${scanId}`);
      // Refresh scan list
      fetchHistory();
      // If the currently viewed scan was the deleted one, clear the viewer
      if (scanResult && scanResult.id === scanId) {
        handleClear();
      }
    } catch (err) {
      console.error("Failed to delete scan", err);
      alert("Failed to delete scan. Ensure connection is active.");
    }
  };

  const handleUploadAndDiagnose = async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);
    // Note: patient_id and eye_side are omitted; backend handles them automatically!

    try {
      const res = await api.post('/scans/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const scanId = res.data.id;

      // Poll for background task completion every 2 seconds
      const poll = setInterval(async () => {
        try {
          const statusRes = await api.get(`/scans/${scanId}`);
          if (statusRes.data.status === 'completed') {
            clearInterval(poll);
            setScanResult(statusRes.data);
            setIsAnalyzing(false);
            fetchHistory(); // Refresh history panel
          } else if (statusRes.data.status === 'failed') {
            clearInterval(poll);
            setIsAnalyzing(false);
            setError("AI Analysis failed. Please try a different scan.");
          }
        } catch (err) {
          clearInterval(poll);
          setIsAnalyzing(false);
          setError("Failed to monitor analysis progress.");
        }
      }, 2000);

    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to upload file. Ensure backend is active.");
      setIsAnalyzing(false);
    }
  };

  const loadScanFromHistory = (scan: any) => {
    setError('');
    setFile(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setScanResult(scan);
  };

  // Diagnostic presentation helpers
  const severityLabels = ["No DR Detected", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"];
  const severityColors = [
    "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
    "text-amber-400 bg-amber-500/10 border-amber-500/20",
    "text-orange-400 bg-orange-500/10 border-orange-500/20",
    "text-rose-400 bg-rose-500/10 border-rose-500/20",
    "text-red-500 bg-red-500/10 border-red-500/20"
  ];
  const severityGlows = [
    "shadow-emerald-500/5 border-emerald-500/20",
    "shadow-amber-500/5 border-amber-500/20",
    "shadow-orange-500/5 border-orange-500/20",
    "shadow-rose-500/5 border-rose-500/20",
    "shadow-red-500/10 border-red-500/20"
  ];
  const progressColors = [
    "bg-emerald-500",
    "bg-amber-500",
    "bg-orange-500",
    "bg-rose-500",
    "bg-red-500"
  ];

  const formatDate = (dateString: string) => {
    try {
      // Append 'Z' to treat as UTC or parse directly
      const d = new Date(dateString);
      return d.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return dateString;
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
      {/* Left Area: Main Workspace (Scanner or Visualizer) */}
      <div className="lg:col-span-2 space-y-6">
        
        {/* Error Banner */}
        {error && (
          <div className="flex items-start gap-4 bg-red-500/10 border border-red-500/20 text-red-400 p-5 rounded-2xl animate-fade-in">
            <AlertCircle className="shrink-0 mt-0.5" size={20} />
            <div className="flex-1">
              <h4 className="font-bold text-sm text-red-200">Analysis Error</h4>
              <p className="text-xs font-semibold leading-relaxed mt-1">{error}</p>
            </div>
            <button onClick={() => setError('')} className="text-red-500 hover:text-red-300 font-bold text-xs p-1">Dismiss</button>
          </div>
        )}

        {/* Diagnostic Panel */}
        {!scanResult ? (
          <div className="bg-slate-900/40 border border-slate-900 rounded-3xl p-6 md:p-8 shadow-2xl relative">
            {!file ? (
              /* Drag and Drop Zone */
              <div
                className="border-2 border-dashed border-slate-800 hover:border-indigo-500/50 rounded-2xl p-20 text-center transition-all duration-300 group cursor-pointer hover:bg-indigo-950/5 relative overflow-hidden"
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
                  <div className="w-16 h-16 bg-indigo-950/40 text-indigo-400 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-105 transition-transform border border-indigo-500/10 shadow-md">
                    <UploadCloud size={28} />
                  </div>
                  <p className="text-lg font-bold text-white mb-2 tracking-tight">Select Patient Retinal Scan</p>
                  <p className="text-xs text-slate-500 font-medium max-w-xs leading-relaxed">
                    Drag and drop fundus photograph or browse local directories
                  </p>
                  <div className="mt-8 bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-800 px-5 py-2.5 rounded-xl font-bold transition-colors inline-block text-xs uppercase tracking-wider cursor-pointer">
                    Browse File System
                  </div>
                </div>
              </div>
            ) : (
              /* Staged File Preview and Run Action */
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-stretch">
                {/* Preview Thumbnail */}
                <div className="bg-slate-950 rounded-2xl overflow-hidden border border-slate-900 flex flex-col shadow-inner">
                  <div className="h-10 bg-slate-900/60 border-b border-slate-900 flex items-center justify-between px-4">
                    <span className="truncate max-w-[180px] font-mono text-[10px] text-slate-500">{file.name}</span>
                    <button onClick={handleClear} className="text-slate-500 hover:text-red-400 transition-colors text-[10px] font-bold uppercase tracking-wider">
                      Remove
                    </button>
                  </div>
                  <div className="flex-1 min-h-[260px] flex items-center justify-center p-6 bg-slate-950 relative">
                    {preview && (
                      <img
                        src={preview}
                        alt="Staged Retina"
                        className="max-w-full max-h-60 object-contain rounded-xl shadow-lg"
                      />
                    )}
                    <div className="absolute bottom-4 left-4 bg-slate-900/90 backdrop-blur text-slate-400 px-3 py-1 rounded-lg text-[10px] font-mono font-bold border border-slate-800">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </div>
                  </div>
                </div>

                {/* Inference Setup */}
                <div className="flex flex-col justify-between py-2">
                  <div className="space-y-4">
                    <h3 className="text-lg font-bold text-white tracking-tight">Image Staged</h3>
                    <p className="text-xs text-slate-400 font-medium leading-relaxed">
                      The scan will be preprocessed (black boundaries cropped, CLAHE contrast adjusted) and evaluated by the diagnostic neural model.
                    </p>
                    <div className="bg-indigo-950/20 rounded-xl p-4 border border-indigo-500/10">
                      <p className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest mb-1">AI Automation</p>
                      <p className="text-xs text-slate-400 leading-relaxed font-medium">
                        The neural network will automatically detect which eye side is scanned by locating the optic disc, saving manual configuration time.
                      </p>
                    </div>
                  </div>

                  <div className="space-y-3 pt-6">
                    <button
                      onClick={handleUploadAndDiagnose}
                      disabled={isAnalyzing}
                      className="w-full bg-indigo-600 hover:bg-indigo-500 text-white py-3.5 rounded-xl font-bold transition-all shadow-lg shadow-indigo-500/10 flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed text-xs uppercase tracking-wider cursor-pointer"
                    >
                      {isAnalyzing ? (
                        <>
                          <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                          Evaluating Retinal Architecture...
                        </>
                      ) : (
                        <>
                          <ClipboardCheck size={16} />
                          Analyze & Save Scan
                        </>
                      )}
                    </button>
                    <button
                      onClick={handleClear}
                      className="w-full text-[10px] text-slate-500 hover:text-red-400 font-bold py-2 transition-colors uppercase tracking-widest"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Analysis Result Blended Viewer */
          <div className="space-y-6 animate-fade-in">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-stretch">
              
              {/* Blender Screen */}
              <div className="md:col-span-2 bg-slate-950 rounded-3xl overflow-hidden border border-slate-900 flex flex-col shadow-2xl relative">
                <div className="h-12 bg-slate-900/60 border-b border-slate-900 flex items-center justify-between px-5 shrink-0">
                  <div className="flex items-center gap-2 text-slate-500 font-mono text-[9px] uppercase font-bold tracking-widest">
                    <span>Retinal Diagnostics Map</span>
                    <span className="opacity-20">|</span>
                    <span className="text-indigo-400">Side: {scanResult.eye_side?.toUpperCase()} EYE</span>
                  </div>
                </div>

                <div className="relative flex-1 min-h-[340px] flex items-center justify-center bg-slate-950 p-6">
                  <img
                    src={scanResult.raw_image_s3_url}
                    alt="Original Fundus"
                    className="absolute max-w-full max-h-[300px] object-contain rounded-xl select-none"
                  />
                  {scanResult.gradcam_image_s3_url && (
                    <img
                      src={scanResult.gradcam_image_s3_url}
                      alt="Grad-CAM Overlay"
                      className="absolute max-w-full max-h-[300px] object-contain rounded-xl select-none transition-opacity"
                      style={{ opacity: opacity / 100 }}
                    />
                  )}
                </div>

                {/* Opacity Control */}
                <div className="bg-slate-900/60 border-t border-slate-900 p-4 flex items-center gap-5 shrink-0">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest whitespace-nowrap">Lesion Attention</span>
                  <input
                    type="range"
                    min="0" max="100"
                    value={opacity}
                    onChange={(e) => setOpacity(Number(e.target.value))}
                    className="w-full h-1 bg-slate-800 rounded-full appearance-none cursor-pointer accent-indigo-500 outline-none"
                  />
                  <span className="text-xs font-mono font-bold text-indigo-400 w-10 text-right">{opacity}%</span>
                </div>
              </div>

              {/* Severity Presentation Panel */}
              <div className="flex flex-col gap-5 justify-between">
                
                {/* Result Card */}
                <div className={`bg-gradient-to-b from-slate-900/80 to-slate-950 border rounded-3xl p-6 shadow-xl flex flex-col items-center justify-center relative overflow-hidden group ${severityGlows[scanResult.dr_prediction_level ?? 0]}`}>
                  <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-4">Diagnostic Conclusion</span>

                  <div className={`px-5 py-2.5 rounded-2xl font-black text-base tracking-tight mb-4 border ${severityColors[scanResult.dr_prediction_level ?? 0]}`}>
                    {severityLabels[scanResult.dr_prediction_level ?? 0]}
                  </div>

                  <div className="w-full mt-4">
                    <div className="flex justify-between text-[8px] font-bold text-slate-500 uppercase tracking-widest mb-2">
                      <span>No Lesions</span>
                      <span>Proliferative</span>
                    </div>
                    <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden relative border border-slate-900">
                      <div
                        style={{ width: `${((scanResult.dr_prediction_level ?? 0) / 4) * 100}%` }}
                        className={`absolute top-0 left-0 h-full rounded-full ${progressColors[scanResult.dr_prediction_level ?? 0]} transition-all duration-1000`}
                      />
                    </div>
                    
                    <div className="mt-6 flex flex-col gap-2">
                      <div className="flex justify-between items-center text-xs font-semibold py-1.5 border-b border-slate-900/50">
                        <span className="text-slate-500 uppercase text-[9px] tracking-wider">Severity Score</span>
                        <span className="text-slate-200 font-mono">{(scanResult.regression_score ?? 0).toFixed(2)} / 4.00</span>
                      </div>
                      <div className="flex justify-between items-center text-xs font-semibold py-1.5 border-b border-slate-900/50">
                        <span className="text-slate-500 uppercase text-[9px] tracking-wider">Eye Side</span>
                        <span className="text-slate-200 uppercase">{scanResult.eye_side === 'right' ? 'Right Eye (OD)' : 'Left Eye (OS)'}</span>
                      </div>
                      <div className="flex justify-between items-center text-xs font-semibold py-1.5">
                        <span className="text-slate-500 uppercase text-[9px] tracking-wider">Scan Time</span>
                        <span className="text-slate-200 text-right">{formatDate(scanResult.created_at)}</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Reset Uploader Button */}
                <button
                  onClick={handleClear}
                  className="w-full bg-slate-900 hover:bg-slate-800 text-white font-bold py-4 rounded-xl border border-slate-800 shadow-lg transition-colors flex items-center justify-center gap-2 text-xs uppercase tracking-wider cursor-pointer"
                >
                  Analyze New Scan
                </button>
              </div>

            </div>
          </div>
        )}
      </div>

      {/* Right Area: Scan History Panel */}
      <div className="bg-slate-900/40 border border-slate-900 rounded-3xl p-6 shadow-2xl flex flex-col h-[600px]">
        <div className="flex items-center gap-2 mb-4 shrink-0 pb-3 border-b border-slate-900">
          <History className="text-indigo-400" size={18} />
          <h2 className="text-base font-bold text-white tracking-tight">Clinical Scan Registry</h2>
        </div>

        {/* Scan List */}
        <div className="flex-1 overflow-y-auto space-y-3 pr-1 scrollbar-hide">
          {loadingHistory ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-500 py-10">
              <div className="w-5 h-5 border-2 border-slate-700 border-t-indigo-500 rounded-full animate-spin mb-3" />
              <p className="text-[10px] uppercase font-bold tracking-wider">Retrieving scan registry...</p>
            </div>
          ) : scans.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-500 py-10 text-center">
              <Calendar className="text-slate-700 mb-3" size={24} />
              <p className="text-xs font-bold text-slate-400">No Scans Recorded</p>
              <p className="text-[10px] text-slate-500 max-w-[180px] mt-1 font-medium leading-relaxed">
                Upload and run retinal photographs to record cases.
              </p>
            </div>
          ) : (
            scans.map((scan) => {
              const dateLabel = formatDate(scan.created_at);
              const colorClass = severityColors[scan.dr_prediction_level ?? 0];
              return (
                <div
                  key={scan.id}
                  onClick={() => loadScanFromHistory(scan)}
                  className={`p-4 bg-slate-950/40 hover:bg-slate-950 border border-slate-900 rounded-2xl cursor-pointer transition-all flex justify-between items-center gap-3 group relative overflow-hidden ${scanResult?.id === scan.id ? 'border-indigo-500/50 bg-slate-950' : ''}`}
                >
                  {scanResult?.id === scan.id && (
                    <div className="absolute left-0 top-0 bottom-0 w-1 bg-indigo-500"></div>
                  )}
                  <div className="space-y-1 min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-0.5 rounded text-[8px] font-extrabold uppercase border ${colorClass}`}>
                        {severityLabels[scan.dr_prediction_level ?? 0]}
                      </span>
                      <span className="text-[9px] font-bold text-slate-500 uppercase">{scan.eye_side === 'right' ? 'OD' : 'OS'}</span>
                    </div>
                    <div className="text-[9px] font-bold text-slate-500 flex items-center gap-1.5">
                      <Calendar size={10} />
                      {dateLabel}
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-3 shrink-0">
                    <div className="font-mono text-xs font-bold text-slate-300">
                      S: {(scan.regression_score ?? 0).toFixed(2)}
                    </div>
                    <button
                      onClick={(e) => handleDeleteScan(scan.id, e)}
                      className="p-1.5 bg-slate-900 hover:bg-red-500/20 text-slate-500 hover:text-red-400 rounded-lg border border-slate-800 hover:border-red-500/20 transition-all opacity-0 group-hover:opacity-100 cursor-pointer"
                      title="Delete Scan"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
