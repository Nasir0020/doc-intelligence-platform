/**
 * api.ts
 * ======
 * A typed wrapper around every backend endpoint. Every function here
 * has an explicit return type matching the backend's Pydantic response
 * model EXACTLY — this is deliberate: if the backend's schema changes
 * (a field renamed, a type changed) and this file isn't updated to
 * match, TypeScript will flag every place in the UI that now has a
 * type mismatch, at COMPILE time, rather than the mismatch silently
 * causing a runtime bug discovered by a user.
 *
 * Why one central file for all API calls, instead of calling fetch()
 * directly from each component?
 *   Every component that needs backend data imports from here instead
 *   of duplicating fetch/error-handling/JSON-parsing logic. If the API
 *   base URL changes, or we need to add auth headers later, there's
 *   exactly one place to change it.
 */

// NEXT_PUBLIC_ prefix is REQUIRED for any environment variable that
// needs to be readable in browser-side code. Next.js only inlines
// env vars with this exact prefix into the client JavaScript bundle
// at build time — any env var WITHOUT this prefix stays server-only,
// which is a deliberate security boundary (so a backend secret in
// .env can never accidentally leak into code that ships to a user's
// browser).
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

// --- Types mirroring backend/app/api/routes/documents.py::DocumentSummary ---
export interface DocumentSummary {
  filename: string;
  total_pages: number;
  total_chunks: number;
  ocr_pages_used: number;
}

// --- Types mirroring backend/app/retrieval/schemas.py ---
export interface Citation {
  source_number: number;
  chunk_id: string;
  source_filename: string;
  page_numbers: number[];
}

export interface GeneratedAnswer {
  answer_text: string;
  citations: Citation[];
  grounded: boolean;
}

/**
 * A small helper that centralizes error handling: fetch() famously
 * does NOT throw on HTTP error statuses (a 404 or 500 response is
 * still a "successful" fetch as far as the Promise is concerned) —
 * only on genuine network failures. This helper explicitly checks
 * response.ok and throws a real Error with the backend's error detail
 * message, so callers can use normal try/catch instead of manually
 * checking status codes everywhere.
 */
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // Response body wasn't valid JSON — fall back to the plain
      // HTTP status text rather than letting a secondary parsing
      // error mask the original failure.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export async function uploadDocument(file: File): Promise<DocumentSummary> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/documents/upload`, {
    method: "POST",
    body: formData,
    // Deliberately NOT setting a Content-Type header here: the browser
    // sets it automatically to "multipart/form-data" WITH the correct
    // boundary string when the body is a FormData object. Setting it
    // manually would omit that boundary and break parsing on the
    // server side — a genuinely common mistake.
  });

  return handleResponse<DocumentSummary>(response);
}

export async function listDocuments(): Promise<DocumentSummary[]> {
  const response = await fetch(`${API_BASE_URL}/documents`);
  return handleResponse<DocumentSummary[]>(response);
}

export async function askQuestion(
  question: string,
  topK: number = 5
): Promise<GeneratedAnswer> {
  const response = await fetch(`${API_BASE_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK }),
  });

  return handleResponse<GeneratedAnswer>(response);
}
