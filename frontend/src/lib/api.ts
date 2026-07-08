import axios from 'axios';

// Original axios instance for local connection
const apiInstance = axios.create({
  baseURL: 'http://localhost:8000/api', // FastAPI default port
  headers: {
    'Content-Type': 'application/json',
  },
});

apiInstance.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token && config.headers) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
}, (error) => Promise.reject(error));

// Check if we should use demo/mock mode
// We use demo mode if we are running in the browser and NOT on localhost/127.0.0.1
const isDemoMode = typeof window !== 'undefined' && 
  window.location.hostname !== 'localhost' && 
  window.location.hostname !== '127.0.0.1' &&
  window.location.hostname !== '::1';

// Mock database in localStorage
const MOCK_SCANS_KEY = 'retina_mock_scans';

const getInitialScans = () => {
  return [
    {
      id: 101,
      dr_prediction_level: 0,
      regression_score: 0.15,
      eye_side: 'right',
      raw_image_s3_url: 'https://images.unsplash.com/photo-1579154204601-01588f351167?w=500&auto=format&fit=crop&q=60',
      gradcam_image_s3_url: null,
      status: 'completed',
      created_at: new Date(Date.now() - 24 * 60 * 60 * 1000 * 2).toISOString(), // 2 days ago
    },
    {
      id: 102,
      dr_prediction_level: 2,
      regression_score: 2.10,
      eye_side: 'right',
      raw_image_s3_url: 'https://images.unsplash.com/photo-1579154204601-01588f351167?w=500&auto=format&fit=crop&q=60',
      gradcam_image_s3_url: null,
      status: 'completed',
      created_at: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(), // 1 day ago
    }
  ];
};

const getScansFromStorage = () => {
  if (typeof window === 'undefined') return [];
  const stored = localStorage.getItem(MOCK_SCANS_KEY);
  if (!stored) {
    const initial = getInitialScans();
    localStorage.setItem(MOCK_SCANS_KEY, JSON.stringify(initial));
    return initial;
  }
  return JSON.parse(stored);
};

const saveScansToStorage = (scans: any[]) => {
  if (typeof window === 'undefined') return;
  localStorage.setItem(MOCK_SCANS_KEY, JSON.stringify(scans));
};

// Wrapper api object to support dual modes: real backend and offline Vercel demo
const api = {
  get: async (url: string, config?: any) => {
    if (isDemoMode) {
      console.log(`[API Mock GET] ${url}`);
      if (url === '/scans/') {
        return { data: getScansFromStorage() };
      }
      if (url.startsWith('/scans/')) {
        const idStr = url.split('/').filter(Boolean).pop();
        const scans = getScansFromStorage();
        const scan = scans.find((s: any) => s.id === Number(idStr));
        if (scan) {
          if (scan.status === 'pending') {
            scan.status = 'completed';
            saveScansToStorage(scans);
          }
          return { data: scan };
        }
        throw { response: { data: { detail: 'Scan not found' } } };
      }
    }
    return apiInstance.get(url, config);
  },

  post: async (url: string, data?: any, config?: any) => {
    if (isDemoMode) {
      console.log(`[API Mock POST] ${url}`, data);
      if (url === '/auth/login') {
        let username = '';
        let password = '';
        if (data instanceof URLSearchParams) {
          username = data.get('username') || '';
          password = data.get('password') || '';
        } else if (typeof data === 'object') {
          username = data.username || '';
          password = data.password || '';
        }

        // Allow '1234' as password for any email as requested
        if (password === '1234') {
          return { data: { access_token: 'mock-access-token', token_type: 'bearer' } };
        } else {
          throw { response: { data: { detail: 'Incorrect email or password' } } };
        }
      }
      if (url === '/auth/register') {
        return { data: { email: data?.email, name: data?.name } };
      }
      if (url === '/scans/upload') {
        const file = data ? (data as FormData).get('file') : null;
        let fileUrl = 'https://images.unsplash.com/photo-1579154204601-01588f351167?w=500&auto=format&fit=crop&q=60';
        if (file && file instanceof File) {
          fileUrl = URL.createObjectURL(file);
        }
        
        const newScan = {
          id: Date.now(),
          dr_prediction_level: 4, // Proliferative DR as requested
          regression_score: 4.01,  // 4.01 score as requested
          eye_side: 'left',       // Left eye side as requested
          raw_image_s3_url: fileUrl,
          gradcam_image_s3_url: '/images/gradcam_demo.jpg', // Path to the copied Grad-CAM image
          status: 'pending',
          created_at: new Date().toISOString(),
        };

        const scans = getScansFromStorage();
        scans.unshift(newScan); // Add to the top
        saveScansToStorage(scans);

        return { data: { id: newScan.id } };
      }
    }
    return apiInstance.post(url, data, config);
  },

  delete: async (url: string, config?: any) => {
    if (isDemoMode) {
      console.log(`[API Mock DELETE] ${url}`);
      if (url.startsWith('/scans/')) {
        const idStr = url.split('/').filter(Boolean).pop();
        const scans = getScansFromStorage();
        const filtered = scans.filter((s: any) => s.id !== Number(idStr));
        saveScansToStorage(filtered);
        return { data: { detail: 'Scan deleted successfully' } };
      }
    }
    return apiInstance.delete(url, config);
  },

  put: async (url: string, data?: any, config?: any) => {
    if (isDemoMode) {
      console.log(`[API Mock PUT] ${url}`, data);
    }
    return apiInstance.put(url, data, config);
  }
};

export default api;
