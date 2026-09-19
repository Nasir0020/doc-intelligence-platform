"use client";

import { useState } from "react";
import { askQuestion, GeneratedAnswer } from "@/lib/api";

export default function ChatPanel() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<GeneratedAnswer | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    // preventDefault() stops the browser's DEFAULT behavior for a form
    // submission, which is a full page reload/navigation. We want to
    // handle the submission entirely in JavaScript (send an async
    // fetch, update state) instead, which is the standard pattern for
    // any React form that doesn't need a traditional page navigation.
    event.preventDefault();
    if (!question.trim()) return;

    setIsLoading(true);
    setError(null);
    setAnswer(null);

    try {
      const result = await askQuestion(question);
      setAnswer(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="rounded-lg border border-gray-200 p-6">
      <h2 className="text-lg font-semibold text-gray-900">Ask a question</h2>

      <form onSubmit={handleSubmit} className="mt-4 flex gap-2">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. What was the APAC growth percentage?"
          className="flex-1 rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-gray-500 focus:outline-none"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading || !question.trim()}
          className="rounded-md bg-gray-900 px-4 py-2 text-sm font-medium text-white hover:bg-gray-700 disabled:opacity-40"
        >
          {isLoading ? "Thinking..." : "Ask"}
        </button>
      </form>

      {error && (
        <p className="mt-4 text-sm text-red-600" role="alert">
          {error}
        </p>
      )}

      {answer && (
        <div className="mt-6 space-y-3">
          {!answer.grounded && (
            <p className="rounded-md bg-yellow-50 px-3 py-2 text-sm text-yellow-800">
              The indexed documents may not fully answer this question.
            </p>
          )}

          <p className="text-gray-900">{answer.answer_text}</p>

          {answer.citations.length > 0 && (
            <div className="flex flex-wrap gap-2 pt-2">
              {/* .map() over an array of citations renders one badge
                  per citation. The `key` prop is REQUIRED by React on
                  any list of elements produced this way — React uses
                  it internally to track which specific DOM element
                  corresponds to which array item across re-renders,
                  so it can efficiently update/reorder/remove exactly
                  the right elements instead of re-rendering the whole
                  list from scratch every time. chunk_id is a good key
                  here because it's guaranteed unique per citation. */}
              {answer.citations.map((citation) => (
                <span
                  key={citation.chunk_id}
                  className="rounded-full bg-gray-100 px-3 py-1 text-xs text-gray-700"
                  title={citation.chunk_id}
                >
                  {citation.source_filename} · p.{citation.page_numbers.join(", ")}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
