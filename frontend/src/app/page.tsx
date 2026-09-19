"use client";

import { useState } from "react";
import DocumentUpload from "@/components/DocumentUpload";
import DocumentList from "@/components/DocumentList";
import ChatPanel from "@/components/ChatPanel";
import { DocumentSummary } from "@/lib/api";

export default function HomePage() {
  // refreshKey is a simple, deliberate pattern for triggering
  // DocumentList's useEffect to re-run from a SIBLING component
  // (DocumentUpload) without those two components needing to know
  // about each other directly. Incrementing this number is the only
  // signal DocumentList's dependency array cares about — the actual
  // VALUE is meaningless, only that it CHANGED.
  const [refreshKey, setRefreshKey] = useState(0);

  function handleUploadSuccess(_summary: DocumentSummary) {
    setRefreshKey((prev) => prev + 1);
    // The functional updater form `(prev) => prev + 1`, rather than
    // `setRefreshKey(refreshKey + 1)`, is the safer pattern here: it
    // guarantees we're incrementing from React's actual latest state
    // at the moment the update is applied, rather than a `refreshKey`
    // value captured in this closure at render time — the two can
    // differ if multiple updates happen in quick succession.
  }

  return (
    <main className="mx-auto max-w-3xl space-y-8 px-4 py-12">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">
          Document Intelligence Platform
        </h1>
        <p className="mt-1 text-gray-500">
          Upload documents and ask questions, with page-level citations.
        </p>
      </div>

      <DocumentUpload onUploadSuccess={handleUploadSuccess} />

      <div>
        <h2 className="mb-3 text-lg font-semibold text-gray-900">
          Indexed documents
        </h2>
        <DocumentList refreshKey={refreshKey} />
      </div>

      <ChatPanel />
    </main>
  );
}
