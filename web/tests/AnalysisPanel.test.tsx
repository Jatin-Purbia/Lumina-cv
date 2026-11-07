import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AnalysisPanel } from "../components/AnalysisPanel";
import type { Pixels } from "../lib/analysis";

function solid(level: number): Pixels {
  const data = new Uint8ClampedArray(16 * 16 * 4).fill(level);
  for (let i = 3; i < data.length; i += 4) data[i] = 255;
  return { data, width: 16, height: 16 };
}

vi.mock("../lib/pixels", () => ({
  loadPixels: vi.fn(async (blob: Blob) => solid(blob.type === "image/x-original" ? 20 : 90)),
}));

describe("AnalysisPanel", () => {
  it("shows the measurements, histogram and comparison once pixels are loaded", async () => {
    render(
      <AnalysisPanel
        original={new Blob(["a"], { type: "image/x-original" })}
        enhanced={new Blob(["b"], { type: "image/png" })}
      />,
    );
    expect(screen.getByText("Measuring the images…")).toBeInTheDocument();
    expect(await screen.findByText("What changed")).toBeInTheDocument();
    expect(screen.getByText("Overall brightness")).toBeInTheDocument();
    expect(screen.getByText("Shadow brightness")).toBeInTheDocument();
    expect(screen.getByText("Compared with plain brightening")).toBeInTheDocument();
    expect(screen.getByText("Lumina (this result)")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /histogram/i })).toBeInTheDocument();
    expect(screen.getByText(/descriptive measurements, not a quality score/)).toBeInTheDocument();
  });

  it("explains when the analysis cannot run", async () => {
    const pixels = await import("../lib/pixels");
    vi.mocked(pixels.loadPixels).mockRejectedValueOnce(new Error("no canvas"));
    render(<AnalysisPanel original={new Blob(["a"])} enhanced={new Blob(["b"])} />);
    expect(await screen.findByText(/could not be computed/)).toBeInTheDocument();
  });
});
