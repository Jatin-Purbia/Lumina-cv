import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { EnhanceApp } from "@/components/EnhanceApp";

interface Routes {
  health?: () => Response | Promise<Response>;
  options?: () => Response | Promise<Response>;
  enhance?: (body: FormData) => Response | Promise<Response>;
}

const okHealth = (): Response =>
  Response.json({ status: "ok", model_loaded: true, device: "cpu", version: "test" });
const okOptions = (): Response =>
  Response.json({
    flags: {},
    curve_scale: 1,
    max_upload_bytes: 2048,
    max_side: 1024,
    accepted_types: ["image/png", "image/jpeg"],
  });
const okEnhance = (): Response =>
  new Response(new Uint8Array([1, 2, 3, 4, 5]), {
    headers: { "Content-Type": "image/png", "X-Lumina-Elapsed-Ms": "87", "X-Lumina-Size": "320x200" },
  });
const apiError = (status: number, code: string, message = "nope"): Response =>
  new Response(JSON.stringify({ error: { code, message } }), {
    status,
    headers: { "Content-Type": "application/json" },
  });

function installFetch(routes: Routes): ReturnType<typeof vi.fn> {
  const fn = vi.fn(async (input: string, init?: RequestInit) => {
    if (input === "/api/v1/health") return (routes.health ?? okHealth)();
    if (input === "/api/v1/options") return (routes.options ?? okOptions)();
    if (input === "/api/v1/enhance") return (routes.enhance ?? okEnhance)(init?.body as FormData);
    throw new Error(`unexpected request ${input}`);
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

let created: string[];
let revoked: string[];

beforeEach(() => {
  created = [];
  revoked = [];
  let n = 0;
  URL.createObjectURL = vi.fn(() => {
    const url = `blob:test/${++n}`;
    created.push(url);
    return url;
  });
  URL.revokeObjectURL = vi.fn((url: string) => {
    revoked.push(url);
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const pngFile = (name = "night.png", size = 100) =>
  new File([new Uint8Array(size)], name, { type: "image/png" });

function fileInput(container: HTMLElement): HTMLInputElement {
  const input = container.querySelector<HTMLInputElement>('input[type="file"]');
  if (input === null) throw new Error("file input missing");
  return input;
}

describe("EnhanceApp", () => {
  it("selects a file, enhances it and shows the comparison, metadata and download", async () => {
    const fetchMock = installFetch({});
    const user = userEvent.setup();
    const { container } = render(<EnhanceApp />);

    await screen.findByText("Model loaded");
    await user.upload(fileInput(container), pngFile());

    expect(await screen.findByAltText("Preview of night.png")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Enhance" }));

    const slider = await screen.findByRole("slider", { name: "Comparison divider" });
    expect(slider).toBeInTheDocument();
    expect(screen.getByText("87 ms")).toBeInTheDocument();
    expect(screen.getByText(/320 × 200 px/)).toBeInTheDocument();

    const download = screen.getByRole("link", { name: "Download PNG" });
    expect(download).toHaveAttribute("download", "night-enhanced.png");
    expect(download.getAttribute("href")).toMatch(/^blob:test\//);

    const enhanceCall = fetchMock.mock.calls.find((c) => c[0] === "/api/v1/enhance");
    const form = (enhanceCall?.[1] as RequestInit).body as FormData;
    expect((form.get("file") as File).name).toBe("night.png");
    expect(form.has("curve_scale")).toBe(false);
  });

  it("shows the curve-scale control only when /options reports curve_scale and sends the choice", async () => {
    let sent: FormData | null = null;
    installFetch({
      enhance: (body) => {
        sent = body;
        return okEnhance();
      },
    });
    const user = userEvent.setup();
    const { container } = render(<EnhanceApp />);
    await screen.findByText("Model loaded");
    await user.upload(fileInput(container), pngFile());

    const group = await screen.findByRole("group", { name: "Estimation scale" });
    await user.click(within(group).getByLabelText("0.25"));
    await user.click(screen.getByRole("button", { name: "Enhance" }));
    await screen.findByRole("slider");
    expect((sent as FormData | null)?.get("curve_scale")).toBe("0.25");
  });

  it("hides the curve-scale control when /options does not report it", async () => {
    installFetch({ options: () => Response.json({ max_upload_bytes: 2048 }) });
    const user = userEvent.setup();
    const { container } = render(<EnhanceApp />);
    await screen.findByText("Model loaded");
    await user.upload(fileInput(container), pngFile());
    await screen.findByRole("button", { name: "Enhance" });
    expect(screen.queryByRole("group", { name: "Estimation scale" })).not.toBeInTheDocument();
  });

  it("rejects files that are the wrong type or over the /options limit without calling enhance", async () => {
    const fetchMock = installFetch({});
    const user = userEvent.setup({ applyAccept: false });
    const { container } = render(<EnhanceApp />);
    await screen.findByText("Model loaded");

    await user.upload(fileInput(container), new File(["x"], "anim.gif", { type: "image/gif" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/anim\.gif is not a supported image/);

    await user.upload(fileInput(container), pngFile("big.png", 4096));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/big\.png is 4\.0 KB\. The limit is 2\.0 KB/));

    expect(fetchMock.mock.calls.some((c) => c[0] === "/api/v1/enhance")).toBe(false);
    expect(screen.queryByRole("button", { name: "Enhance" })).not.toBeInTheDocument();
  });

  it("shows a specific message for 503 model_not_loaded and keeps the preview", async () => {
    installFetch({ enhance: () => apiError(503, "model_not_loaded") });
    const user = userEvent.setup();
    const { container } = render(<EnhanceApp />);
    await screen.findByText("Model loaded");
    await user.upload(fileInput(container), pngFile());
    await user.click(await screen.findByRole("button", { name: "Enhance" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("The model is not loaded yet");
    expect(screen.getByAltText("Preview of night.png")).toBeInTheDocument();
    expect(screen.queryByRole("slider")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Enhance" })).toBeEnabled();
  });

  it("shows a network-failure message when the API is unreachable", async () => {
    installFetch({
      health: () => {
        throw new TypeError("Failed to fetch");
      },
      options: () => {
        throw new TypeError("Failed to fetch");
      },
      enhance: () => {
        throw new TypeError("Failed to fetch");
      },
    });
    const user = userEvent.setup();
    const { container } = render(<EnhanceApp />);

    expect(await screen.findByText("Server unreachable")).toBeInTheDocument();
    await user.upload(fileInput(container), pngFile());
    await user.click(await screen.findByRole("button", { name: "Enhance" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Cannot reach the enhancement server");
  });

  it("reports when the server is up but no model is loaded", async () => {
    installFetch({
      health: () => Response.json({ status: "ok", model_loaded: false, device: "cpu", version: "t" }),
    });
    render(<EnhanceApp />);
    expect(await screen.findByText("Server up, no model loaded")).toBeInTheDocument();
  });

  it("re-checks health when the window regains focus", async () => {
    const fetchMock = installFetch({});
    render(<EnhanceApp />);
    await screen.findByText("Model loaded");
    const healthCalls = (): number => fetchMock.mock.calls.filter((c) => c[0] === "/api/v1/health").length;
    const before = healthCalls();
    window.dispatchEvent(new Event("focus"));
    await waitFor(() => expect(healthCalls()).toBe(before + 1));
  });

  it("revokes object URLs when the image is replaced and on unmount", async () => {
    installFetch({});
    const user = userEvent.setup();
    const { container, unmount } = render(<EnhanceApp />);
    await screen.findByText("Model loaded");

    await user.upload(fileInput(container), pngFile("one.png"));
    await user.click(await screen.findByRole("button", { name: "Enhance" }));
    await screen.findByRole("slider");
    expect(created).toHaveLength(2); // source + result
    expect(revoked).toHaveLength(0);

    await user.upload(fileInput(container), pngFile("two.png"));
    await screen.findByAltText("Preview of two.png");
    // Old source and old result are revoked; a new source exists.
    expect(revoked.sort()).toEqual([created[0], created[1]].sort());
    expect(created).toHaveLength(3);

    unmount();
    expect(revoked).toContain(created[2]);
    expect(new Set(revoked).size).toBe(created.length);
  });
});
