"use client";

import { useState, useEffect, useRef } from 'react';
import { 
  UploadCloud, 
  FileImage, 
  AlertCircle, 
  History, 
  Trash2, 
  Calendar, 
  ClipboardCheck, 
  Columns, 
  ZoomIn, 
  ZoomOut, 
  ShieldCheck, 
  Activity, 
  Sparkles, 
  Search,
  RefreshCw,
  ExternalLink,
  Info
} from 'lucide-react';
import api from '@/lib/api';

export default function DashboardPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStep, setAnalysisStep] = useState(0);
  const [scanResult, setScanResult] = useState<any>(null);
  const [viewMode, setViewMode] = useState<'split' | 'heatmap' | 'raw'>('split');
  const [zoomLevel, setZoomLevel] = useState(1);
  const [error, setError] = useState('');
  const [scans, setScans] = useState<any[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [searchFilter, setSearchFilter] = useState('');
  const [gradeFilter, setGradeFilter] = useState<number | 'all'>('all');
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

  // Analysis step animation during scanning
  useEffect(() => {
    let interval: any;
    if (isAnalyzing) {
      setAnalysisStep(0);
      interval = setInterval(() => {
        setAnalysisStep((prev) => (prev < 3 ? prev + 1 : prev));
      }, 1200);
    } else {
      setAnalysisStep(0);
    }
    return () => clearInterval(interval);
  }, [isAnalyzing]);

  const handleFileSelect = (selectedFile: File) => {
    if (!selectedFile.type.startsWith('image/')) {
      setError("Please upload a valid fundus retinal image (JPEG, PNG).");
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
    setZoomLevel(1);
    if (inputRef.current) inputRef.current.value = '';
  };

  const handleDeleteScan = async (scanId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm("Are you sure you want to permanently delete this scan record?")) return;
    
    try {
      await api.delete(`/scans/${scanId}`);
      fetchHistory();
      if (scanResult && scanResult.id === scanId) {
        handleClear();
      }
    } catch (err) {
      console.error("Failed to delete scan", err);
      alert("Failed to delete scan. Check network connection.");
    }
  };

  const handleUploadAndDiagnose = async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await api.post('/scans/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const scanId = res.data.id;

      // Poll every 1.5 seconds for fast completion
      const poll = setInterval(async () => {
        try {
          const statusRes = await api.get(`/scans/${scanId}`);
          if (statusRes.data.status === 'completed') {
            clearInterval(poll);
            setScanResult(statusRes.data);
            setIsAnalyzing(false);
            fetchHistory();
          } else if (statusRes.data.status === 'failed') {
            clearInterval(poll);
            setIsAnalyzing(false);
            setError("AI Diagnostic processing failed. Please try a different scan.");
          }
        } catch (err) {
          clearInterval(poll);
          setIsAnalyzing(false);
          setError("Failed to monitor analysis progress.");
        }
      }, 1500);

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
    setZoomLevel(1);
  };

  // Diagnostic metadata definitions
  const severityMetadata = [
    {
      title: "No Apparent Retinopathy",
      grade: "Grade 0",
      code: "NO_DR",
      color: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30",
      glow: "shadow-emerald-500/10 border-emerald-500/30",
      barColor: "bg-emerald-500",
      referral: "Non-Referable",
      referralColor: "text-emerald-400 bg-emerald-950/60 border-emerald-500/20",
      recommendation: "Routine annual diabetic retinal rescreening recommended. Maintain optimal glycemic (HbA1c) and blood pressure control.",
      riskLevel: "Low Risk"
    },
    {
      title: "Mild Non-Proliferative DR",
      grade: "Grade 1",
      code: "MILD_NPDR",
      color: "text-amber-400 bg-amber-500/10 border-amber-500/30",
      glow: "shadow-amber-500/10 border-amber-500/30",
      barColor: "bg-amber-500",
      referral: "Non-Referable",
      referralColor: "text-amber-400 bg-amber-950/60 border-amber-500/20",
      recommendation: "Microaneurysms detected. Rescreen in 6–12 months. Reinforce medical management of diabetes, lipids, and hypertension.",
      riskLevel: "Moderate Early Stage"
    },
    {
      title: "Moderate Non-Proliferative DR",
      grade: "Grade 2",
      code: "MOD_NPDR",
      color: "text-orange-400 bg-orange-500/10 border-orange-500/30",
      glow: "shadow-orange-500/10 border-orange-500/30",
      barColor: "bg-orange-500",
      referral: "Referable DR",
      referralColor: "text-orange-400 bg-orange-950/60 border-orange-500/20",
      recommendation: "Multiple microaneurysms and dot hemorrhages identified. Comprehensive ophthalmology referral within 2–4 months.",
      riskLevel: "High Risk (Referral Indicated)"
    },
    {
      title: "Severe Non-Proliferative DR",
      grade: "Grade 3",
      code: "SEV_NPDR",
      color: "text-rose-400 bg-rose-500/10 border-rose-500/30",
      glow: "shadow-rose-500/10 border-rose-500/30",
      barColor: "bg-rose-500",
      referral: "Urgent Referral",
      referralColor: "text-rose-400 bg-rose-950/60 border-rose-500/20",
      recommendation: "Severe hemorrhages, venous beading, or IRMA present. Urgent retinal specialist referral within 2–4 weeks. High risk of proliferative conversion.",
      riskLevel: "Severe Clinical Risk"
    },
    {
      title: "Proliferative Diabetic Retinopathy",
      grade: "Grade 4",
      code: "PDR",
      color: "text-red-500 bg-red-500/10 border-red-500/30",
      glow: "shadow-red-500/15 border-red-500/30",
      barColor: "bg-red-500",
      referral: "Critical Referral",
      referralColor: "text-red-500 bg-red-950/60 border-red-500/20",
      recommendation: "Active neovascularization or vitreous hemorrhage detected. Immediate vitreoretinal evaluation. Urgent panretinal photocoagulation or anti-VEGF therapy.",
      riskLevel: "Vision-Threatening Emergency"
    }
  ];

  const currentMeta = severityMetadata[scanResult?.dr_prediction_level ?? 0] || severityMetadata[0];

  const formatDate = (dateString: string) => {
    try {
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

  // Filter scans
  const filteredScans = scans.filter(scan => {
    const matchesSearch = !searchFilter || 
      (scan.eye_side?.toLowerCase().includes(searchFilter.toLowerCase())) ||
      (scan.id?.toString().includes(searchFilter));
    const matchesGrade = gradeFilter === 'all' || scan.dr_prediction_level === gradeFilter;
    return matchesSearch && matchesGrade;
  });

  const stepLabels = [
    "Preprocessing Retinal Image (YUV CLAHE Contrast)",
    "Running ConvNeXt-V2 Large Deep Feature Extraction",
    "Synthesizing High-Resolution Lesion Grad-CAM Map",
    "Finalizing Multitask Diagnostic Grade & Summary"
  ];

  return (
    <div className="space-y-8">
      {/* ─── Clean Engine Status Header ─── */}
      <div className="bg-slate-900/40 backdrop-blur-xl border border-slate-800/80 rounded-2xl p-4 sm:p-5 flex flex-wrap items-center justify-between gap-4 shadow-xl">
        <div className="flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-emerald-400 animate-pulse shadow-lg shadow-emerald-400/50"></div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-white tracking-wide uppercase">ConvNeXt-V2 Large Diagnostic Engine</span>
            </div>
            <p className="text-[10px] text-slate-400 mt-0.5">
              Authentic Grad-CAM Lesion Interpretability • Automated Multi-Task Ordinal Regression (CORN)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="text-slate-400">
            Total Scans: <span className="text-slate-100 font-bold">{scans.length}</span>
          </div>
          <div className="w-px h-4 bg-slate-800"></div>
          <button 
            onClick={fetchHistory}
            className="text-slate-400 hover:text-indigo-400 flex items-center gap-1.5 transition-colors cursor-pointer text-[11px]"
            title="Refresh Scan List"
          >
            <RefreshCw size={13} />
            Sync
          </button>
        </div>
      </div>

      {/* ─── Error Notification ─── */}
      {error && (
        <div className="flex items-start gap-4 bg-rose-500/10 border border-rose-500/30 text-rose-300 p-4 sm:p-5 rounded-2xl shadow-lg animate-fade-in">
          <AlertCircle className="shrink-0 mt-0.5 text-rose-400" size={20} />
          <div className="flex-1">
            <h4 className="font-bold text-sm text-rose-200">Diagnostic Error</h4>
            <p className="text-xs font-medium leading-relaxed mt-1 text-rose-300/90">{error}</p>
          </div>
          <button onClick={() => setError('')} className="text-rose-400 hover:text-rose-200 font-bold text-xs p-1">
            Dismiss
          </button>
        </div>
      )}

      {/* ─── Main Grid Workspace ─── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
        
        {/* Left Area (2 cols): Stager, Scanner & Side-by-Side Visualizer */}
        <div className="lg:col-span-2 space-y-6">

          {!scanResult ? (
            /* Upload & Staging Workspace */
            <div className="bg-slate-900/50 backdrop-blur-xl border border-slate-800/90 rounded-3xl p-6 sm:p-8 shadow-2xl relative overflow-hidden">
              
              {!file ? (
                /* Drag and Drop Zone */
                <div
                  className="border-2 border-dashed border-slate-800 hover:border-indigo-500/60 rounded-2xl p-12 sm:p-16 text-center transition-all duration-300 group cursor-pointer hover:bg-indigo-950/10 relative overflow-hidden"
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
                    <div className="w-16 h-16 bg-gradient-to-tr from-indigo-950/80 to-indigo-800/30 text-indigo-400 rounded-2xl flex items-center justify-center mb-5 group-hover:scale-110 transition-transform border border-indigo-500/20 shadow-lg shadow-indigo-500/10">
                      <UploadCloud size={30} />
                    </div>
                    <h3 className="text-lg font-bold text-white mb-2 tracking-tight">Upload Retinal Fundus Scan</h3>
                    <p className="text-xs text-slate-400 font-medium max-w-sm leading-relaxed mb-6">
                      Drag and drop high-resolution fundus photograph or click to browse. Supports JPG, PNG, and TIFF formats.
                    </p>
                    <div className="bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-700/80 px-5 py-2.5 rounded-xl font-bold transition-all text-xs uppercase tracking-wider cursor-pointer shadow-md">
                      Browse Files
                    </div>
                  </div>
                </div>
              ) : (
                /* Staged Image Confirmation & Scanning View */
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-stretch">
                  {/* Left: Thumbnail & Scan Beam */}
                  <div className="bg-slate-950 rounded-2xl overflow-hidden border border-slate-800/80 flex flex-col shadow-inner relative">
                    <div className="h-10 bg-slate-900/80 border-b border-slate-800/80 flex items-center justify-between px-4">
                      <span className="truncate max-w-[180px] font-mono text-[11px] text-slate-400">{file.name}</span>
                      <button 
                        onClick={handleClear} 
                        disabled={isAnalyzing}
                        className="text-slate-500 hover:text-rose-400 transition-colors text-[10px] font-bold uppercase tracking-wider disabled:opacity-30"
                      >
                        Change
                      </button>
                    </div>

                    <div className="flex-1 min-h-[280px] flex items-center justify-center p-6 bg-slate-950 relative overflow-hidden">
                      {preview && (
                        <img
                          src={preview}
                          alt="Staged Fundus"
                          className="max-w-full max-h-64 object-contain rounded-xl shadow-lg"
                        />
                      )}

                      {/* Animated Scanning Beam during inference */}
                      {isAnalyzing && (
                        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-indigo-500/20 to-transparent animate-scan pointer-events-none border-b-2 border-indigo-400 shadow-[0_0_20px_rgba(99,102,241,0.5)]"></div>
                      )}

                      <div className="absolute bottom-3 left-3 bg-slate-900/90 backdrop-blur text-slate-300 px-3 py-1 rounded-lg text-[10px] font-mono font-bold border border-slate-800 shadow">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </div>
                    </div>
                  </div>

                  {/* Right: AI Pipeline Configuration & Run Action */}
                  <div className="flex flex-col justify-between py-1">
                    <div className="space-y-4">
                      <div className="flex items-center gap-2">
                        <Sparkles className="text-indigo-400" size={18} />
                        <h3 className="text-lg font-bold text-white tracking-tight">Retinal Analysis Ready</h3>
                      </div>
                      <p className="text-xs text-slate-400 font-medium leading-relaxed">
                        The neural pipeline will preprocess the fundus scan with YUV CLAHE contrast equalization, evaluate multi-task severity, and synthesize an authentic Grad-CAM lesion heatmap.
                      </p>

                      {/* Progress Stages during analysis */}
                      {isAnalyzing ? (
                        <div className="bg-indigo-950/30 border border-indigo-500/30 rounded-2xl p-4 space-y-3">
                          <div className="flex items-center justify-between text-[11px] font-bold text-indigo-300">
                            <span>Processing Pipeline</span>
                            <span className="font-mono">{analysisStep + 1} / 4</span>
                          </div>
                          <div className="h-1.5 w-full bg-slate-900 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-indigo-500 transition-all duration-700 rounded-full"
                              style={{ width: `${((analysisStep + 1) / 4) * 100}%` }}
                            ></div>
                          </div>
                          <p className="text-xs text-slate-300 font-medium animate-pulse">
                            {stepLabels[analysisStep]}
                          </p>
                        </div>
                      ) : (
                        <div className="bg-slate-950/60 rounded-xl p-4 border border-slate-800/80 space-y-2 text-xs text-slate-400">
                          <div className="flex items-center gap-2 font-bold text-indigo-400 uppercase text-[10px] tracking-wider">
                            <ShieldCheck size={14} />
                            Automated Anatomical Localization
                          </div>
                          <p className="leading-relaxed text-[11px]">
                            Optic disc position is automatically recognized to assign Left Eye (OS) or Right Eye (OD) laterality without manual tagging.
                          </p>
                        </div>
                      )}
                    </div>

                    <div className="space-y-3 pt-6">
                      <button
                        onClick={handleUploadAndDiagnose}
                        disabled={isAnalyzing}
                        className="w-full bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 text-white py-4 rounded-xl font-bold transition-all shadow-xl shadow-indigo-600/20 flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed text-xs uppercase tracking-wider cursor-pointer"
                      >
                        {isAnalyzing ? (
                          <>
                            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                            Evaluating Retinal Scan...
                          </>
                        ) : (
                          <>
                            <ClipboardCheck size={18} />
                            Execute Diagnostic Scan
                          </>
                        )}
                      </button>

                      <button
                        onClick={handleClear}
                        disabled={isAnalyzing}
                        className="w-full text-[11px] text-slate-500 hover:text-rose-400 font-bold py-2 transition-colors uppercase tracking-widest disabled:opacity-20 cursor-pointer"
                      >
                        Cancel Selection
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            /* Analysis Result Multi-Mode Visualizer (Side-by-Side Default) */
            <div className="space-y-6 animate-fade-in">
              
              {/* Visualizer Container */}
              <div className="bg-slate-900/50 backdrop-blur-xl rounded-3xl overflow-hidden border border-slate-800/90 flex flex-col shadow-2xl">
                
                {/* Visualizer Top Bar with View Mode Switcher */}
                <div className="h-14 bg-slate-950/80 border-b border-slate-800/80 flex flex-wrap items-center justify-between px-5 gap-3 shrink-0">
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2 text-slate-400 font-mono text-[10px] uppercase font-bold tracking-wider">
                      <span className="text-white font-bold">Scan #{scanResult.id}</span>
                      <span className="opacity-30">•</span>
                      <span className="text-indigo-400">
                        {scanResult.eye_side === 'right' ? 'Right Eye (OD)' : 'Left Eye (OS)'}
                      </span>
                    </div>
                  </div>

                  {/* Mode Buttons */}
                  <div className="flex items-center bg-slate-900/90 rounded-xl p-1 border border-slate-800">
                    <button
                      onClick={() => setViewMode('split')}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 cursor-pointer ${viewMode === 'split' ? 'bg-indigo-600 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'}`}
                    >
                      <Columns size={13} />
                      Side-by-Side
                    </button>
                    <button
                      onClick={() => setViewMode('heatmap')}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 cursor-pointer ${viewMode === 'heatmap' ? 'bg-indigo-600 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'}`}
                    >
                      <Activity size={13} />
                      Grad-CAM Map
                    </button>
                    <button
                      onClick={() => setViewMode('raw')}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 cursor-pointer ${viewMode === 'raw' ? 'bg-indigo-600 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'}`}
                    >
                      <FileImage size={13} />
                      Fundus Scan
                    </button>
                  </div>
                </div>

                {/* Viewport Canvas */}
                <div className="relative min-h-[380px] sm:min-h-[420px] flex items-center justify-center bg-slate-950 p-6 overflow-hidden">
                  
                  {viewMode === 'split' && (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 w-full max-w-4xl items-center">
                      <div className="bg-slate-900/60 rounded-2xl p-3 border border-slate-800/80 flex flex-col items-center">
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2">Preprocessed Fundus</span>
                        <img
                          src={scanResult.raw_image_s3_url}
                          alt="Original Fundus"
                          style={{ transform: `scale(${zoomLevel})` }}
                          className="max-h-[300px] object-contain rounded-xl select-none transition-transform"
                        />
                      </div>
                      <div className="bg-slate-900/60 rounded-2xl p-3 border border-slate-800/80 flex flex-col items-center">
                        <span className="text-[10px] font-bold text-indigo-400 uppercase tracking-wider mb-2">Grad-CAM Lesion Heatmap</span>
                        <img
                          src={scanResult.gradcam_image_s3_url || scanResult.raw_image_s3_url}
                          alt="GradCAM Map"
                          style={{ transform: `scale(${zoomLevel})` }}
                          className="max-h-[300px] object-contain rounded-xl select-none transition-transform"
                        />
                      </div>
                    </div>
                  )}

                  {viewMode === 'heatmap' && (
                    <div className="relative w-full h-full flex items-center justify-center">
                      <img
                        src={scanResult.gradcam_image_s3_url || scanResult.raw_image_s3_url}
                        alt="Grad-CAM Heatmap"
                        style={{ transform: `scale(${zoomLevel})` }}
                        className="max-w-full max-h-[360px] object-contain rounded-xl select-none transition-transform duration-200"
                      />
                    </div>
                  )}

                  {viewMode === 'raw' && (
                    <div className="relative w-full h-full flex items-center justify-center">
                      <img
                        src={scanResult.raw_image_s3_url}
                        alt="Preprocessed Fundus"
                        style={{ transform: `scale(${zoomLevel})` }}
                        className="max-w-full max-h-[360px] object-contain rounded-xl select-none transition-transform duration-200"
                      />
                    </div>
                  )}

                  {/* Canvas Zoom Controls Overlay */}
                  <div className="absolute top-4 right-4 flex items-center gap-1.5 bg-slate-900/80 backdrop-blur rounded-xl p-1 border border-slate-800">
                    <button
                      onClick={() => setZoomLevel(prev => Math.min(prev + 0.25, 2.5))}
                      className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition-colors"
                      title="Zoom In"
                    >
                      <ZoomIn size={14} />
                    </button>
                    <button
                      onClick={() => setZoomLevel(prev => Math.max(prev - 0.25, 0.75))}
                      className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition-colors"
                      title="Zoom Out"
                    >
                      <ZoomOut size={14} />
                    </button>
                    <button
                      onClick={() => setZoomLevel(1)}
                      className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition-colors text-[10px] font-mono px-2"
                      title="Reset Zoom"
                    >
                      {Math.round(zoomLevel * 100)}%
                    </button>
                  </div>
                </div>

              </div>

              {/* Action Toolbar */}
              <div className="flex flex-wrap items-center justify-between gap-4">
                <button
                  onClick={handleClear}
                  className="bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-3.5 px-6 rounded-xl shadow-lg shadow-indigo-600/20 transition-all flex items-center gap-2 text-xs uppercase tracking-wider cursor-pointer"
                >
                  <UploadCloud size={16} />
                  Analyze New Scan
                </button>

                <div className="flex items-center gap-3">
                  <a
                    href={scanResult.gradcam_image_s3_url || scanResult.raw_image_s3_url}
                    download={`retinal_diagnosis_${scanResult.id}.jpg`}
                    target="_blank"
                    rel="noreferrer"
                    className="bg-slate-900 hover:bg-slate-800 text-slate-200 font-bold py-3.5 px-4 rounded-xl border border-slate-800 shadow transition-all flex items-center gap-2 text-xs uppercase tracking-wider cursor-pointer"
                    title="Open Full Resolution"
                  >
                    <ExternalLink size={15} />
                    <span>View High-Res</span>
                  </a>
                </div>
              </div>

            </div>
          )}
        </div>

        {/* Right Area (1 col): Clinical Diagnostic Findings or Registry */}
        <div className="space-y-6">

          {/* If Result Active: Clinical Diagnosis Summary Card */}
          {scanResult && (
            <div className={`bg-gradient-to-b from-slate-900/90 to-slate-950 border rounded-3xl p-6 shadow-2xl relative overflow-hidden transition-all ${currentMeta.glow}`}>
              <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-800/80">
                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest flex items-center gap-1.5">
                  <ShieldCheck size={14} className="text-indigo-400" />
                  Clinical Diagnostic Report
                </span>
                <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-extrabold uppercase border ${currentMeta.referralColor}`}>
                  {currentMeta.referral}
                </span>
              </div>

              {/* Main Severity Badge */}
              <div className="text-center my-4 space-y-2">
                <div className="text-xs font-mono font-bold text-slate-400 uppercase tracking-wider">
                  {currentMeta.grade}
                </div>
                <div className={`px-4 py-3 rounded-2xl font-black text-lg sm:text-xl tracking-tight border shadow-lg ${currentMeta.color}`}>
                  {currentMeta.title}
                </div>
              </div>

              {/* Severity Gauge */}
              <div className="mt-6 space-y-2">
                <div className="flex justify-between text-[9px] font-bold text-slate-400 uppercase tracking-wider">
                  <span>Continuous Severity Index</span>
                  <span className="font-mono text-slate-200">
                    {(scanResult.regression_score ?? 0).toFixed(2)} / 4.00
                  </span>
                </div>
                <div className="h-2 w-full bg-slate-950 rounded-full overflow-hidden relative border border-slate-800">
                  <div
                    style={{ width: `${Math.min(Math.max(((scanResult.regression_score ?? 0) / 4) * 100, 4), 100)}%` }}
                    className={`h-full rounded-full ${currentMeta.barColor} transition-all duration-1000`}
                  />
                </div>
              </div>

              {/* Clinical Guidelines & Action Plan */}
              <div className="mt-6 bg-slate-950/80 rounded-2xl p-4 border border-slate-800/90 space-y-2">
                <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider text-slate-300">
                  <Info size={13} className="text-indigo-400" />
                  Management & Referral Protocol
                </div>
                <p className="text-xs leading-relaxed text-slate-400 font-medium">
                  {currentMeta.recommendation}
                </p>
              </div>

              {/* Scan Metadata Table */}
              <div className="mt-6 space-y-2 text-xs">
                <div className="flex justify-between items-center py-2 border-b border-slate-800/60 font-medium">
                  <span className="text-slate-500 uppercase text-[10px] tracking-wider">Laterality</span>
                  <span className="text-slate-200 font-bold uppercase">
                    {scanResult.eye_side === 'right' ? 'Right Eye (OD)' : 'Left Eye (OS)'}
                  </span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-slate-800/60 font-medium">
                  <span className="text-slate-500 uppercase text-[10px] tracking-wider">Risk Category</span>
                  <span className="text-slate-200 font-bold">{currentMeta.riskLevel}</span>
                </div>
                <div className="flex justify-between items-center py-2 font-medium">
                  <span className="text-slate-500 uppercase text-[10px] tracking-wider">Timestamp</span>
                  <span className="text-slate-300 text-right">{formatDate(scanResult.created_at)}</span>
                </div>
              </div>
            </div>
          )}

          {/* Clinical Scan Registry (History Sidebar) */}
          <div className="bg-slate-900/50 backdrop-blur-xl border border-slate-800/90 rounded-3xl p-6 shadow-2xl flex flex-col h-[520px]">
            <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-800/80 shrink-0">
              <div className="flex items-center gap-2">
                <History className="text-indigo-400" size={17} />
                <h2 className="text-sm font-bold text-white tracking-tight">Scan Registry</h2>
              </div>
              <span className="text-[10px] font-mono text-slate-400 font-bold bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                {filteredScans.length} records
              </span>
            </div>

            {/* Filter Bar */}
            <div className="mb-3 space-y-2 shrink-0">
              <div className="relative">
                <Search size={12} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                <input
                  type="text"
                  placeholder="Filter by side or ID..."
                  value={searchFilter}
                  onChange={(e) => setSearchFilter(e.target.value)}
                  className="w-full bg-slate-950 text-xs text-slate-200 pl-8 pr-3 py-1.5 rounded-xl border border-slate-800 focus:outline-none focus:border-indigo-500 placeholder:text-slate-600 font-medium"
                />
              </div>

              {/* Grade Filter Pills */}
              <div className="flex items-center gap-1 overflow-x-auto pb-1 scrollbar-hide text-[9px] font-bold font-mono">
                <button
                  onClick={() => setGradeFilter('all')}
                  className={`px-2 py-1 rounded-md transition-all cursor-pointer ${gradeFilter === 'all' ? 'bg-indigo-600 text-white' : 'bg-slate-950 text-slate-400 border border-slate-800 hover:text-slate-200'}`}
                >
                  ALL
                </button>
                {[0, 1, 2, 3, 4].map(g => (
                  <button
                    key={g}
                    onClick={() => setGradeFilter(g)}
                    className={`px-2 py-1 rounded-md transition-all cursor-pointer ${gradeFilter === g ? 'bg-indigo-600 text-white' : 'bg-slate-950 text-slate-400 border border-slate-800 hover:text-slate-200'}`}
                  >
                    G{g}
                  </button>
                ))}
              </div>
            </div>

            {/* Registry List */}
            <div className="flex-1 overflow-y-auto space-y-2.5 pr-1 scrollbar-hide">
              {loadingHistory ? (
                <div className="h-full flex flex-col items-center justify-center text-slate-500 py-10">
                  <div className="w-5 h-5 border-2 border-slate-700 border-t-indigo-500 rounded-full animate-spin mb-3" />
                  <p className="text-[10px] uppercase font-bold tracking-wider">Syncing scan registry...</p>
                </div>
              ) : filteredScans.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-slate-500 py-10 text-center">
                  <Calendar className="text-slate-700 mb-2" size={24} />
                  <p className="text-xs font-bold text-slate-400">No Scans Match</p>
                  <p className="text-[10px] text-slate-500 max-w-[180px] mt-1 font-medium">
                    Upload new fundus scans or adjust filters.
                  </p>
                </div>
              ) : (
                filteredScans.map((scan) => {
                  const meta = severityMetadata[scan.dr_prediction_level ?? 0] || severityMetadata[0];
                  const isSelected = scanResult?.id === scan.id;
                  return (
                    <div
                      key={scan.id}
                      onClick={() => loadScanFromHistory(scan)}
                      className={`p-3 bg-slate-950/60 hover:bg-slate-950 border border-slate-800/80 hover:border-slate-700 rounded-2xl cursor-pointer transition-all flex justify-between items-center gap-3 group relative overflow-hidden ${isSelected ? 'border-indigo-500/80 bg-slate-950 shadow-md' : ''}`}
                    >
                      {isSelected && (
                        <div className="absolute left-0 top-0 bottom-0 w-1 bg-indigo-500"></div>
                      )}
                      
                      <div className="space-y-1 min-w-0 flex-1 pl-1">
                        <div className="flex items-center gap-1.5">
                          <span className={`px-2 py-0.5 rounded text-[8px] font-extrabold uppercase border ${meta.color}`}>
                            {meta.grade}
                          </span>
                          <span className="text-[9px] font-bold text-slate-400 uppercase font-mono">
                            {scan.eye_side === 'right' ? 'OD' : 'OS'}
                          </span>
                        </div>
                        <div className="text-[9px] font-bold text-slate-500 flex items-center gap-1">
                          <Calendar size={10} />
                          {formatDate(scan.created_at)}
                        </div>
                      </div>

                      <div className="flex items-center gap-2.5 shrink-0">
                        <div className="font-mono text-[11px] font-bold text-slate-300">
                          {(scan.regression_score ?? 0).toFixed(2)}
                        </div>
                        <button
                          onClick={(e) => handleDeleteScan(scan.id, e)}
                          className="p-1.5 bg-slate-900 hover:bg-rose-500/20 text-slate-500 hover:text-rose-400 rounded-lg border border-slate-800 hover:border-rose-500/30 transition-all opacity-0 group-hover:opacity-100 cursor-pointer"
                          title="Delete Scan Record"
                        >
                          <Trash2 size={12} />
                        </button>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>

        </div>

      </div>
    </div>
  );
}
