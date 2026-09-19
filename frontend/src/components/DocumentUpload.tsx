"use client";
// The "use client" directive marks this file as a CLIENT COMPONENT.
// Next.js's App Router defaults every component to a SERVER component
// (rendered once on the server, sent to the browser as static HTML,
// with no JavaScript interactivity by default — good for performance).
// This component needs useState (to track upload progress) and an
// onClick handler (to respond to user interaction) — neither of those
// can exist in a server component, since there's no "server" to hold
// interactive state or respond to browser events on. "use client" opts
// this specific file INTO being shipped as interactive JavaScript that
// runs in the browser, exactly like a "normal" React app would.

import { useState } from "react";
import { uploadDocument, DocumentSummary } from "@/lib/api";

interface DocumentUploadProps {
  // A callback prop: the PARENT component (page.tsx) decides what
  // happens after a successful upload (e.g. refreshing the document
  // list) — this component's only job is uploading, not deciding what
  // happens next. This is a standard React pattern: child components
  // notify parents via callback props rather than reaching "up" into
  // parent state directly, which keeps data flow predictable
  // (React's core principle of "one-way data flow").
  onUploadSuccess: (summary: DocumentSummary) => void;
}

export default function DocumentUpload({ onUploadSuccess }: DocumentUploadProps) {
  // useState returns a [currentValue, setterFunction] pair. Calling
  // the setter (e.g. setIsUploading(true)) tells React "re-render this
  // component, and this time isUploading should be true" — React
  // doesn't mutate the old value in place, it schedules a fresh render
  // with the new one.
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    // Optional chaining (?.) + array index: files is a FileList,
    // which can be null (if the user cancels the file picker) or
    // empty. `?.[0]` safely returns undefined instead of throwing in
    // either case, rather than needing a separate explicit null check
    // first.
    if (!file) return;

    setIsUploading(true);
    setError(null);

    try {
      const summary = await uploadDocument(file);
      onUploadSuccess(summary);
    } catch (err) {
      // `err` is typed `unknown` in strict TypeScript catch blocks
      // (not `any`) — we can't assume it's an Error object without
      // checking, since JavaScript technically allows throwing any
      // value at all. The `instanceof` check narrows the type safely.
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setIsUploading(false);
      // Reset the input's value so selecting the SAME file again
      // still fires onChange (browsers don't fire a change event if
      // the selected file path hasn't changed since last time).
      event.target.value = "";
    }
  }

  return (
    <div className="rounded-lg border border-gray-200 p-6">
      <h2 className="text-lg font-semibold text-gray-900">Upload a document</h2>
      <p className="mt-1 text-sm text-gray-500">
        PDF files only. Scanned documents are automatically OCR&apos;d.
      </p>

      <label className="mt-4 flex cursor-pointer items-center justify-center rounded-md border-2 border-dashed border-gray-300 px-6 py-8 hover:border-gray-400">
        <input
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={handleFileChange}
          disabled={isUploading}
        />
        <span className="text-sm text-gray-600">
          {isUploading ? "Uploading and indexing..." : "Click to choose a PDF"}
        </span>
      </label>

      {error && (
        <p className="mt-2 text-sm text-red-600" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
