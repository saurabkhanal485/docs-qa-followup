// Cloud-friendly replacement for the original ollama.ts.
//
//  - Chat:       Groq (free, hosted, OpenAI-compatible /chat/completions API)
//  - Embeddings: @xenova/transformers, running locally inside the server
//                process. No API key, no external service, free forever.
//                Model weights are downloaded once and cached on disk.

import { pipeline, type FeatureExtractionPipeline } from '@xenova/transformers';

const GROQ_API_KEY = process.env.GROQ_API_KEY || '';
const GROQ_BASE_URL = 'https://api.groq.com/openai/v1';
const CHAT_MODEL = process.env.GROQ_CHAT_MODEL || 'llama-3.1-8b-instant';
const EMBEDDING_MODEL = 'Xenova/all-MiniLM-L6-v2';

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

/** Small retry helper for transient network/5xx errors. */
async function withRetry<T>(fn: () => Promise<T>, attempts = 3, baseDelayMs = 500): Promise<T> {
  let lastErr: unknown;
  for (let i = 0; i < attempts; i++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      if (i < attempts - 1) {
        await new Promise((r) => setTimeout(r, baseDelayMs * 2 ** i));
      }
    }
  }
  throw lastErr;
}

// The feature-extraction pipeline is expensive to create, so we build it
// once per server process and reuse it across requests.
let embedderPromise: Promise<FeatureExtractionPipeline> | null = null;
function getEmbedder(): Promise<FeatureExtractionPipeline> {
  if (!embedderPromise) {
    embedderPromise = pipeline('feature-extraction', EMBEDDING_MODEL) as Promise<FeatureExtractionPipeline>;
  }
  return embedderPromise;
}

export async function embed(text: string): Promise<number[]> {
  const embedder = await getEmbedder();
  const output = await embedder(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data as Float32Array);
}

/** Streams assistant text chunks as they arrive from Groq's chat completions API. */
export async function* streamChat(messages: ChatMessage[]): AsyncGenerator<string> {
  if (!GROQ_API_KEY) {
    throw new Error(
      'GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys and add it to your environment variables.',
    );
  }

  const res = await withRetry(() =>
    fetch(`${GROQ_BASE_URL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${GROQ_API_KEY}`,
      },
      body: JSON.stringify({ model: CHAT_MODEL, messages, stream: true }),
    }),
  );

  if (!res.ok || !res.body) {
    throw new Error(`Groq chat failed (${res.status}): ${await res.text().catch(() => '')}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split('\n');
    buffer = lines.pop() || ''; // keep the last (possibly incomplete) line

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice('data:'.length).trim();
      if (payload === '[DONE]') return;
      if (!payload) continue;

      const json = JSON.parse(payload) as {
        choices?: { delta?: { content?: string }; finish_reason?: string | null }[];
      };
      const token = json.choices?.[0]?.delta?.content;
      if (token) yield token;
    }
  }
}

/**
 * Follow-up handling — M5
 *
 * RAG retrieval only works well when the query stands on its own.
 * A raw follow-up like "how do I do that asynchronously?" embeds
 * poorly because it has no idea what "that" refers to.
 *
 * This rewrites the latest user message into a standalone search
 * query using the last few turns of conversation as context, via a
 * small, fast, non-streaming Groq call. If it's the first message in
 * the conversation, or anything goes wrong, we just fall back to the
 * original question — retrieval still works, it's just not
 * context-aware.
 */
export async function rewriteStandaloneQuery(
  history: ChatMessage[],
  question: string,
): Promise<string> {
  if (!GROQ_API_KEY || history.length === 0) return question;

  const historyText = history
    .map((m) => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`)
    .join('\n');

  const prompt: ChatMessage[] = [
    {
      role: 'system',
      content:
        'You rewrite follow-up questions into standalone search queries for a documentation ' +
        'search engine. Given the recent conversation and a new follow-up question, produce a ' +
        'single self-contained question that includes any context (subject, feature, object) ' +
        'implied by the conversation but missing from the follow-up. ' +
        'If the follow-up is already standalone, return it unchanged. ' +
        'Respond with ONLY the rewritten question, no explanation, no quotes.',
    },
    {
      role: 'user',
      content: `Conversation so far:\n${historyText}\n\nFollow-up question: ${question}\n\nStandalone question:`,
    },
  ];

  try {
    const res = await withRetry(
      () =>
        fetch(`${GROQ_BASE_URL}/chat/completions`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${GROQ_API_KEY}`,
          },
          body: JSON.stringify({
            model: CHAT_MODEL,
            messages: prompt,
            stream: false,
            temperature: 0,
            max_tokens: 120,
          }),
        }),
      2,
      300,
    );

    if (!res.ok) return question;

    const json = (await res.json()) as {
      choices?: { message?: { content?: string } }[];
    };
    const rewritten = json.choices?.[0]?.message?.content?.trim();
    return rewritten || question;
  } catch {
    return question;
  }
}

/**
 * Suggested follow-ups — M6
 *
 * After answering, suggest 2-3 natural next questions, grounded in the
 * last few turns of conversation plus the answer just given, so
 * suggestions actually build on where the conversation is rather than
 * being generic. One small, fast, non-streaming Groq call. On any
 * failure we just return no suggestions — this is a nicety, never a
 * blocker for the main answer.
 */
export async function generateFollowUps(
  history: ChatMessage[],
  question: string,
  answer: string,
): Promise<string[]> {
  if (!GROQ_API_KEY) return [];

  const historyText = [...history, { role: 'user', content: question } as ChatMessage]
    .map((m) => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`)
    .join('\n');

  const prompt: ChatMessage[] = [
    {
      role: 'system',
      content:
        'You suggest short follow-up questions for a FastAPI documentation chat assistant. ' +
        'Given the recent conversation and the answer just given, suggest 3 natural next ' +
        'questions the user might ask — things that build on this specific exchange, not ' +
        'generic FastAPI trivia. Keep each under 12 words, phrased as a question. ' +
        'Respond with ONLY a JSON array of 3 strings, no explanation, no markdown fences.',
    },
    {
      role: 'user',
      content: `Conversation:\n${historyText}\n\nAssistant's answer: ${answer}\n\nSuggest 3 follow-up questions as a JSON array:`,
    },
  ];

  try {
    const res = await withRetry(
      () =>
        fetch(`${GROQ_BASE_URL}/chat/completions`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${GROQ_API_KEY}`,
          },
          body: JSON.stringify({
            model: CHAT_MODEL,
            messages: prompt,
            stream: false,
            temperature: 0.4,
            max_tokens: 200,
          }),
        }),
      2,
      300,
    );

    if (!res.ok) return [];

    const json = (await res.json()) as {
      choices?: { message?: { content?: string } }[];
    };
    const raw = json.choices?.[0]?.message?.content?.trim() ?? '';
    // Strip accidental ```json fences before parsing.
    const cleaned = raw.replace(/^```(?:json)?/i, '').replace(/```$/, '').trim();
    const parsed = JSON.parse(cleaned);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((q): q is string => typeof q === 'string' && q.trim().length > 0)
      .map((q) => q.trim())
      .slice(0, 3);
  } catch {
    return [];
  }
}

export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
