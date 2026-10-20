import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from './api'

describe('API client', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('sends analysis text and returns the server prediction', async () => {
    const response = { prediction: 'Likely Real', fake_probability: .1, verification_probability: .1, real_probability: .8, confidence: .8, model_type: 'BiLSTM', disclaimer: 'Estimate', text_statistics: { characters: 3, words: 1, normalized_characters: 3 } }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => response }))
    await expect(api.predict('text')).resolves.toEqual(response)
    expect(fetch).toHaveBeenCalledWith(expect.stringContaining('/api/predict'), expect.objectContaining({ method: 'POST' }))
  })

  it('uses API error detail when a request fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 503, json: async () => ({ detail: 'Model is not trained.' }) }))
    await expect(api.predict('text')).rejects.toThrow('Model is not trained.')
  })
})
