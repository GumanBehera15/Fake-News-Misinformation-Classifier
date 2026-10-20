export interface Prediction { prediction: string; fake_probability: number; real_probability: number; verification_probability: number; confidence: number; model_type: string; disclaimer: string; text_statistics: { characters: number; words: number; normalized_characters: number } }
export interface HistoryItem { id: number; preview: string; prediction: string; confidence: number; created_at: string }
export interface Metrics { accuracy: number; precision: number; recall: number; f1: number; confusion_matrix: number[][]; test_samples: number; labels: string[] }

