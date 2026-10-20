import type { HistoryItem, Metrics, Prediction } from '../types'
const base = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${base}${path}`, { headers: { 'Content-Type': 'application/json' }, ...options })
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `Request failed (${response.status})`) }
  return response.json() as Promise<T>
}
export const api = { predict: (text: string) => request<Prediction>('/api/predict', { method: 'POST', body: JSON.stringify({ text }) }), history: () => request<HistoryItem[]>('/api/history'), clearHistory: () => request<{message:string}>('/api/history', {method:'DELETE'}), metrics: () => request<Metrics>('/api/model/metrics'), info: () => request<{available:boolean; message:string; max_sequence_length?:number; vocabulary_size?:number}>('/api/model/info') }

