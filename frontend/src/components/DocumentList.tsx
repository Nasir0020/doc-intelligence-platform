"use client";

import { useEffect, useState } from "react";
import { listDocuments, DocumentSummary } from "@/lib/api";

export interface DocumentListHandle {
  refresh: () => void;
}

export default function DocumentList({ refreshKey }: { refreshKey: number }) {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  // useEffect runs side effects — anything that reaches OUTSIDE this
  // component's own rendering (a network call, in this case) after
  // React has finished rendering. The array at the end, [refreshKey],
  // is the DEPENDENCY ARRAY: React re-runs this effect whenever any
  // value inside that array changes between renders (using reference
  // equality). Passing an EMPTY array ([]) would mean "run once, only
  // on mount, never again" — we specifically want it to re-run every
  // time refreshKey changes (which the parent bumps after a successful
  // upload), so the list reflects newly indexed documents without a
  // full page reload.
  useEffect(() => {
    let cancelled = false;
    // The `cancelled` flag guards against a real, subtle bug: if this
    // component unmounts (or refreshKey changes again) WHILE a fetch
    // is still in flight, the old fetch's response would otherwise
    // still call setDocuments() on a component that's no longer
    // showing that data context — a classic React "state update on
    // unmounted/stale component" issue. Checking `cancelled` before
    // applying the result avoids acting on a stale, superseded request.
    listDocuments()
      .then((docs) => {
        if (!cancelled) setDocuments(docs);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load documents.");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  if (error) {
    return <p className="text-sm text-red-600">{error}</p>;
  }

  if (documents.length === 0) {
    return <p className="text-sm text-gray-500">No documents indexed yet.</p>;
  }

  return (
    <ul className="space-y-2">
      {documents.map((doc) => (
        <li
          key={doc.filename}
          className="flex items-center justify-between rounded-md bg-gray-50 px-3 py-2 text-sm"
        >
          <span className="font-medium text-gray-800">{doc.filename}</span>
          <span className="text-gray-500">
            {doc.total_pages} pages · {doc.total_chunks} chunks
            {doc.ocr_pages_used > 0 && ` · ${doc.ocr_pages_used} via OCR`}
          </span>
        </li>
      ))}
    </ul>
  );
}
